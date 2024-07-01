import os
from typing import TYPE_CHECKING

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling

from .block_drop import consecutive_block_dropping, discrete_block_dropping, post_block_drop
from .expert_drop import layerwise_pruning, global_pruning, post_experts_drop
from .io import save_sparse_model, save_expert_dropped_config, save_block_dropped_config, save_layer_dropped_config, load_json
from .layer_drop import discrete_layer_dropping, post_layers_drop
from ...data import get_dataset
from ...extras.constants import IGNORE_INDEX
from ...model import load_model_and_tokenizer
from ...compression.prune.prune import prune_magnitude, prune_sparsegpt, prune_wanda

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments
    from ...hparams import DataArguments, ModelArguments, PruningArguments

DATA_AWARE_PRUNING_METHODS = ("wanda", "sparsegpt", "expert_drop", "block_drop", "layer_drop")

EXPERT_DROP_METHODS_FUNC = {
    'layerwise_pruning': layerwise_pruning,
    'global_pruning': global_pruning,
}

LAYER_DROP_METHODS_FUNC = {
    'discrete': discrete_layer_dropping,
}

BLOCK_DROP_METHODS_FUNC = {
    'discrete': discrete_block_dropping,
    'consecutive': consecutive_block_dropping,
}


# 🔍 Modified from src.llmtuner.compression.pt.workflow.run_pt
def run_prune(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        pruning_args: "PruningArguments",  # 🔍 for pruning
):
    """Workflow for pruning and decomposing."""
    # 🔍 accelerator
    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")
    accelerator.print("Pruning Args:", pruning_args)
    accelerator.print("Model Args:", model_args)

    # 🔍 model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, training_args.do_train)

    if pruning_args.compress_method == "expert_drop" and pruning_args.expert_drop_method == "post_dropping":
        assert (os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) and (os.environ.get("ACCELERATE_USE_FSDP", "false"))
        config = load_json(os.path.join(pruning_args.compressed_model_save_path, "config.json"))
        accelerator.wait_for_everyone()
        post_experts_drop(pruning_args.compressed_model_save_path, model, tokenizer, config, accelerator, preserve_gate=pruning_args.preserve_gate)
        exit()

    if pruning_args.compress_method == "layer_drop" and pruning_args.layer_drop_method == "post_dropping":
        assert (os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) and (os.environ.get("ACCELERATE_USE_FSDP", "false"))
        reserved_layer_list = load_json(os.path.join(pruning_args.compressed_model_save_path, "reserved_layers.json"))
        post_layers_drop(pruning_args.compressed_model_save_path, model, tokenizer, reserved_layer_list, accelerator)
        exit()

    if pruning_args.compress_method == "block_drop" and pruning_args.block_drop_method == "post_dropping":
        assert (os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) and (os.environ.get("ACCELERATE_USE_FSDP", "false"))
        layer_id_mapping = load_json(os.path.join(pruning_args.compressed_model_save_path, "layer_mapping.json"))
        post_block_drop(pruning_args.compressed_model_save_path, model, tokenizer, layer_id_mapping, accelerator)
        exit()

    if pruning_args.compress_method in DATA_AWARE_PRUNING_METHODS:
        # 🔍 dataset & data collator & dataloader
        dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage=pruning_args.data_type)

        if pruning_args.data_type == "pt":
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # concat all data to seq_length for each batch
        elif pruning_args.data_type == "sft":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
                label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            )
        else:
            raise NotImplementedError

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=data_collator, num_workers=8)  # batch size must be 1

        accelerator.print("Total Sample Num:", len(dataset))
        accelerator.print("Total Used Sample Num:", pruning_args.n_compression_samples)
        accelerator.print("Max sequence Length:", data_args.cutoff_len)
        accelerator.print(f"Example Data (len = {len(dataset[0]['input_ids'])}):", dataset[0])
        if pruning_args.n_compression_samples > len(dataset):
            raise ValueError("Number of calibration samples is greater than the number of samples in the dataset!")

        # 🔍 Prepare model & dataloader
        print("Preparing model...")
        model, dataloader = accelerator.prepare(model, dataloader)

        # 🔍 Distribute samples to each device for acceleration
        assert (pruning_args.n_compression_samples % accelerator.num_processes == 0)  # have to be divided evenly
        num_samples_each_device = pruning_args.n_compression_samples // accelerator.num_processes
        accelerator.print("Number of samples per device:", len(dataloader))
        accelerator.print("Number of used samples per device:", num_samples_each_device)

    else:  # use no additional data for pruning, can be done on 1 GPU
        if (os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) or (os.environ.get("ACCELERATE_USE_FSDP", "false")):
            raise EnvironmentError("Data-independent pruning can only be done without DeepSpeed / FSDP environment!")
        print("Preparing model...")
        model = accelerator.prepare([model], device_placement=[False])[0]  # 🔍 Prepare model

    #######################################################################################################
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if pruning_args.sparsity_type != "unstructured" and ":" in pruning_args.sparsity_type:
        assert pruning_args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, pruning_args.sparsity_type.split(":"))

    if pruning_args.compress_method == "wanda":
        update_state_dict = prune_wanda(pruning_args, model, dataloader, accelerator, num_samples_each_device, prune_n=prune_n, prune_m=prune_m)
    elif pruning_args.compress_method == "sparsegpt":
        update_state_dict = prune_sparsegpt(pruning_args, model, dataloader, accelerator, num_samples_each_device, prune_n=prune_n, prune_m=prune_m)
    elif pruning_args.compress_method == "magnitude":
        update_state_dict = prune_magnitude(pruning_args, model, accelerator, prune_n=prune_n, prune_m=prune_m)  # Data-independent
    elif pruning_args.compress_method == "expert_drop":
        EXPERT_DROP_METHODS_FUNC[pruning_args.expert_drop_method](pruning_args, model, dataloader, accelerator, num_samples_each_device)
    elif pruning_args.compress_method == "layer_drop":
        dropped_layer_list = LAYER_DROP_METHODS_FUNC[pruning_args.layer_drop_method](pruning_args, model, dataloader, accelerator, num_samples_each_device)
    elif pruning_args.compress_method == "block_drop":
        dropped_layer_list = BLOCK_DROP_METHODS_FUNC[pruning_args.block_drop_method](pruning_args, model, dataloader, accelerator, num_samples_each_device)
    else:
        raise NotImplementedError

    #######################################################################################################
    accelerator.print(f"model: {model}")

    if pruning_args.compressed_model_save_path is not None:
        if pruning_args.compress_method == "expert_drop":
            # 🔍 only return the idx of remaining experts.
            save_expert_dropped_config(pruning_args.compressed_model_save_path, model, tokenizer, accelerator)
        elif pruning_args.compress_method == "layer_drop":
            save_layer_dropped_config(pruning_args.compressed_model_save_path, model, tokenizer, accelerator, dropped_layer_list=dropped_layer_list)
        elif pruning_args.compress_method == "block_drop":
            save_block_dropped_config(pruning_args.compressed_model_save_path, model, tokenizer, accelerator, dropped_layer_list=dropped_layer_list)
        else:  # wanda sparsegpt
            # 🔍 Save sparse model to disk
            save_sparse_model(pruning_args.compressed_model_save_path, model, tokenizer, accelerator, update_state_dict, check_sparsity=True)

    accelerator.print("All done!")
