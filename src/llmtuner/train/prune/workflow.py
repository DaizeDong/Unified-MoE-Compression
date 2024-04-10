# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py
from typing import TYPE_CHECKING, List, Optional

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, DataCollatorWithPadding

from .io import save_sparse_model
from ..dpo.collator import DPODataCollatorWithPadding
from ..rm.collator import PairwiseDataCollatorWithPadding
from ...data import get_dataset
from ...extras.constants import IGNORE_INDEX
from ...model import load_model_and_tokenizer
from ...train.prune.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity_from_state_dict

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, ModelArguments, PruningArguments

DATA_AWARE_PRUNING_METHODS = ("wanda", "sparsegpt", "gradient-first", "gradient-zeroth")


# 🔍 Modified from src.train.pt.workflow.run_pt
def run_prune(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        pruning_args: "PruningArguments",  # 🔍 for pruning
        callbacks: Optional[List["TrainerCallback"]] = None,
):
    # 🔍 accelerator
    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")
    accelerator.print("Pruning Args:", pruning_args)
    accelerator.print("Model Args:", model_args)

    # 🔍 model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)

    if pruning_args.prune_method in DATA_AWARE_PRUNING_METHODS:
        # 🔍 dataset & data collator & dataloader
        dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage=pruning_args.prune_data_type)

        if pruning_args.prune_data_type == "pt":
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # concat all data to seq_length for each batch
        elif pruning_args.prune_data_type == "sft":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
                label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            )
        elif pruning_args.prune_data_type == "rm":
            data_collator = PairwiseDataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        elif pruning_args.prune_data_type == "ppo":
            tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        else:  # dpo
            data_collator = DPODataCollatorWithPadding(
                tokenizer=tokenizer,
                pad_to_multiple_of=8,
                label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            )

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=data_collator, num_workers=8)  # batch size must be 1

        accelerator.print("Total Sample Num:", len(dataset))
        accelerator.print("Total Used Sample Num:", pruning_args.n_calibration_samples)
        accelerator.print("Max sequence Length:", data_args.cutoff_len)
        accelerator.print(f"Example Data (len = {len(dataset[0]['input_ids'])}):", dataset[0])
        if pruning_args.n_calibration_samples > len(dataset):
            raise ValueError("Number of calibration samples is greater than the number of samples in the dataset!")

        # 🔍 Prepare model & dataloader
        model, dataloader = accelerator.prepare(model, dataloader)

        # 🔍 Distribute samples to each device for acceleration
        assert (pruning_args.n_calibration_samples % accelerator.num_processes == 0)  # have to be divided evenly
        num_samples_each_device = pruning_args.n_calibration_samples // accelerator.num_processes
        accelerator.print("Number of samples per device:", len(dataloader))
        accelerator.print("Number of used samples per device:", num_samples_each_device)

    elif AcceleratorState().deepspeed_plugin is not None:
        raise EnvironmentError("Data-independent pruning can only be done without DeepSpeed environment!")

    else:  # use no additional data for pruning, can be done on 1 GPU
        model = accelerator.prepare([model], device_placement=[False])[0]  # 🔍 Prepare model

    #######################################################################################################
    # TODO: Pruning at initialization.
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if pruning_args.sparsity_type != "unstructured":
        assert pruning_args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, pruning_args.sparsity_type.split(":"))

    if pruning_args.sparsity_ratio != 0:
        if pruning_args.prune_method == "wanda":
            update_state_dict = prune_wanda(pruning_args, model, dataloader, accelerator, num_samples_each_device, prune_n=prune_n, prune_m=prune_m)
        elif pruning_args.prune_method == "sparsegpt":
            update_state_dict = prune_sparsegpt(pruning_args, model, dataloader, accelerator, num_samples_each_device, prune_n=prune_n, prune_m=prune_m)
        elif pruning_args.prune_method == "gradient-first":
            raise NotImplementedError
        elif pruning_args.prune_method == "gradient-zeroth":
            raise NotImplementedError
        elif pruning_args.prune_method == "magnitude":
            update_state_dict = prune_magnitude(pruning_args, model, accelerator, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in pruning_args.prune_method:  ##### 这可以删掉吧 #####
            prune_ablate(pruning_args, model, tokenizer, prune_n=prune_n, prune_m=prune_m)  # TODO: adjust?
        else:
            raise NotImplementedError
    #######################################################################################################

    # 🔍 check sparsity
    if accelerator.is_main_process:
        accelerator.print("*" * 30)
        accelerator.print("Calculating sparsity for pruned params in the state dict...")
        sparsity_ratio = check_sparsity_from_state_dict(update_state_dict)
        accelerator.print(f"sparsity sanity check {sparsity_ratio:.4f}")
        accelerator.print("*" * 30)
    accelerator.wait_for_everyone()

    # 🔍 Save sparse model to disk
    if pruning_args.prune_model_save_path is not None:
        save_sparse_model(pruning_args.prune_model_save_path, model, tokenizer, accelerator, update_state_dict)

    # if training_args.do_eval:
    #     # TODO: eval on evaluation set (e.g. wikitext-2 calibration)
    #     pass

    accelerator.print("All done!")
