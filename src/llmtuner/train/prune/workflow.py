# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from copy import deepcopy
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, DataCollatorWithPadding
from .decompose import decompose_moe
from .io import save_sparse_model, save_update_state_dict, save_decomposed_model
from .prune import gate_remap
from ..dpo.collator import DPODataCollatorWithPadding
from ..rm.collator import PairwiseDataCollatorWithPadding
from ...data import get_dataset
from ...extras.constants import IGNORE_INDEX
from ...model import load_model_and_tokenizer, load_tokenizer
from ...train.prune.prune import prune_wanda_moe, prune_magnitude, prune_sparsegpt, prune_wanda

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, ModelArguments, PruningArguments

DATA_AWARE_PRUNING_METHODS = ("wanda", "sparsegpt", "gradient-first", "gradient-zeroth")


# 🔍 Modified from src.llmtuner.train.pt.workflow.run_pt
def run_prune(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        pruning_args: "PruningArguments",  # 🔍 for pruning
        callbacks: Optional[List["TrainerCallback"]] = None,
):
    """Workflow for pruning and decomposing."""
    # 🔍 accelerator
    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")
    accelerator.print("Pruning Args:", pruning_args)
    accelerator.print("Model Args:", model_args)

    # 🔍 model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    # tokenizer = load_tokenizer(model_args)
    
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
        print("Preparing model...")
        model, dataloader = accelerator.prepare(model, dataloader)

        # 🔍 Distribute samples to each device for acceleration
        assert (pruning_args.n_calibration_samples % accelerator.num_processes == 0)  # have to be divided evenly
        num_samples_each_device = pruning_args.n_calibration_samples // accelerator.num_processes
        accelerator.print("Number of samples per device:", len(dataloader))
        accelerator.print("Number of used samples per device:", num_samples_each_device)

    elif AcceleratorState().deepspeed_plugin is not None:
        raise EnvironmentError("Data-independent pruning can only be done without DeepSpeed environment!")

    else:  # use no additional data for pruning, can be done on 1 GPU
        print("Preparing model...")
        model = accelerator.prepare([model], device_placement=[False])[0]  # 🔍 Prepare model

    #######################################################################################################
    # TODO: Pruning at initialization.
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if pruning_args.sparsity_type != "unstructured":
        assert pruning_args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, pruning_args.sparsity_type.split(":"))

    if pruning_args.prune_method == "wanda":
        update_state_dict = prune_wanda(pruning_args, model, dataloader, accelerator, num_samples_each_device, prune_n=prune_n, prune_m=prune_m)
        # update_state_dict = prune_wanda_moe(pruning_args, model, dataloader, accelerator, num_samples_each_device, prune_n=prune_n, prune_m=prune_m)
    elif pruning_args.prune_method == "sparsegpt":
        update_state_dict = prune_sparsegpt(pruning_args, model, dataloader, accelerator, num_samples_each_device, prune_n=prune_n, prune_m=prune_m)
    elif pruning_args.prune_method == "gradient-first":
        raise NotImplementedError
    elif pruning_args.prune_method == "gradient-zeroth":
        raise NotImplementedError
    elif pruning_args.prune_method == "magnitude":
        update_state_dict = prune_magnitude(pruning_args, model, accelerator, prune_n=prune_n, prune_m=prune_m)  # Data-independent
    elif pruning_args.prune_method == "decompose_moe":
        pruning_args.sparsity_ratio = 1.0 - pruning_args.sparsity_ratio
        level = "layer"  # expert layer model
        has_sparse = True
        do_permute = True  # 🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍
        use_svd = True
        update_state_dict = decompose_moe(pruning_args, model, accelerator, level=level, has_sparse=has_sparse, do_permute=do_permute, use_svd=use_svd)  # Data-independent
    else:
        raise NotImplementedError
    #######################################################################################################

    # 🔍 Set config for low-rank decomposition.
    accelerator.print(f"model: {model}")

    # 🔍 Save sparse model to disk
    if pruning_args.prune_method == "decompose_moe":
        setattr(model.config, "decomposed", True)
        setattr(model.config, "has_sparse", has_sparse)
        if pruning_args.prune_model_save_path is not None:
            save_decomposed_model(pruning_args.prune_model_save_path, model, tokenizer, accelerator, update_state_dict)
    else:
        if pruning_args.prune_model_save_path is not None:
            save_sparse_model(pruning_args.prune_model_save_path, model, tokenizer, accelerator, update_state_dict, check_sparsity=True)

    # if training_args.do_eval:
    #     # TODO: eval on evaluation set (e.g. wikitext-2 calibration)
    #     pass

    accelerator.print("All done!")


def run_prune_remap_gate(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        pruning_args: "PruningArguments",  # 🔍 for pruning
        callbacks: Optional[List["TrainerCallback"]] = None,
):
    """Workflow for remapping the gate network."""
    # 🔍 accelerator
    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")
    accelerator.print("Pruning Args:", pruning_args)
    accelerator.print("Model Args:", model_args)

    if AcceleratorState().deepspeed_plugin is not None:
        raise EnvironmentError("Performing gate-remapping in DeepSpeed environment will result in errors! Use FSDP instead!")

    # 🔍 model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    model_pruned_args = deepcopy(model_args)
    model_pruned_args.model_name_or_path = pruning_args.pruned_model_path
    model_pruned, _ = load_model_and_tokenizer(model_pruned_args, finetuning_args, training_args.do_train)

    tokenizer = load
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
    print("Preparing model...")
    model, model_pruned, dataloader = accelerator.prepare(model, model_pruned, dataloader)

    # 🔍 Distribute samples to each device for acceleration
    assert (pruning_args.n_calibration_samples % accelerator.num_processes == 0)  # have to be divided evenly
    num_samples_each_device = pruning_args.n_calibration_samples // accelerator.num_processes
    accelerator.print("Number of samples per device:", len(dataloader))
    accelerator.print("Number of used samples per device:", num_samples_each_device)

    #######################################################################################################
    update_state_dict = gate_remap(model, model_pruned, dataloader, accelerator, num_samples_each_device)
    #######################################################################################################

    # Updating the parameters from the state_dict will cause errors in the FSDP environment.
    # So we need to initialize a new model to load it.
    # Here we save the state_dict first to avoid accidents.
    # 🔍 Save state_dict to disk
    if pruning_args.prune_model_save_path is not None:
        save_update_state_dict(pruning_args.prune_model_save_path, accelerator, update_state_dict)

    # 🔍 Reload state_dict and save model
    print("Reloading model and saving...")
    if accelerator.is_main_process:
        model_pruned, _ = load_model_and_tokenizer(model_pruned_args, finetuning_args, training_args.do_train)
        model_pruned.load_state_dict(update_state_dict, strict=False)
        model_pruned.save_pretrained(pruning_args.prune_model_save_path)
        tokenizer.save_pretrained(pruning_args.prune_model_save_path)
        # delete_file_or_dir(os.path.join(pruning_args.prune_model_save_path, "update_state_dict.pt"))
    accelerator.wait_for_everyone()

    accelerator.print("All done!")
