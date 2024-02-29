# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

from typing import TYPE_CHECKING, List, Optional

from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from .io import save_sparse_model
from ...data import get_dataset
from ...model import load_model_and_tokenizer
from ...train.prune.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity_from_state_dict

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, ModelArguments, PruningArguments

from accelerate import Accelerator


# 🔍
# Copied from src.train.pt.workflow.run_pt
def run_prune(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        pruning_args: "PruningArguments",  # 🔍 for pruning
        callbacks: Optional[List["TrainerCallback"]] = None,
):
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="pt")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=data_collator, num_workers=4)

    # 🔍 Prepare model & dataloader
    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")
    model, dataloader = accelerator.prepare(model, dataloader)

    accelerator.print("Total Data Num:", len(dataloader))
    accelerator.print("Target Used Data Num:", pruning_args.n_calibration_samples)
    accelerator.print("Sequence Length:", len(dataset[0]["input_ids"]))
    accelerator.print("Example Data:", dataset[0])

    # TODO: Pruning at initialization.

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if pruning_args.sparsity_type != "unstructured":
        assert pruning_args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, pruning_args.sparsity_type.split(":"))

    # 🔍 Distribute samples to each device for acceleration
    assert (pruning_args.n_calibration_samples % accelerator.num_processes == 0)  # have to be divided evenly
    num_samples_each_device = pruning_args.n_calibration_samples // accelerator.num_processes
    accelerator.print("Number of samples per device:", num_samples_each_device)

    if pruning_args.sparsity_ratio != 0:
        accelerator.print("pruning starts")
        if pruning_args.prune_method == "wanda":
            update_state_dict = prune_wanda(pruning_args, model, dataloader, accelerator, num_samples_each_device, data_args.cutoff_len, prune_n=prune_n, prune_m=prune_m)
        elif pruning_args.prune_method == "magnitude":
            prune_magnitude(pruning_args, model, tokenizer, prune_n=prune_n, prune_m=prune_m)  # TODO: adjust
        elif pruning_args.prune_method == "sparsegpt":
            update_state_dict = prune_sparsegpt(pruning_args, model, dataloader, accelerator, num_samples_each_device, data_args.cutoff_len, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in pruning_args.prune_method:
            prune_ablate(pruning_args, model, tokenizer, prune_n=prune_n, prune_m=prune_m)  # TODO: adjust

    ################################################################
    if accelerator.is_main_process:
        accelerator.print("*" * 30)
        sparsity_ratio = check_sparsity_from_state_dict(update_state_dict)  # 🔍 check sparsity
        accelerator.print(f"sparsity sanity check {sparsity_ratio:.4f}")
        accelerator.print("*" * 30)
    accelerator.wait_for_everyone()
    ################################################################

    save_sparse_model(pruning_args, model, tokenizer, accelerator, update_state_dict)  # 🔍 Save sparse model to disk
    accelerator.print("All done!")
