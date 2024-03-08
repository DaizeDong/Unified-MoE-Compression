import os

from accelerate import Accelerator

from .prune import check_sparsity_from_state_dict


# 🔍
def save_sparse_model(prune_model_save_path, model, tokenizer, accelerator: Accelerator, update_state_dict):
    accelerator.print("Saving models... (may take minutes)")
    if accelerator.is_main_process:
        if not os.path.exists(prune_model_save_path):
            os.makedirs(prune_model_save_path)
    accelerator.wait_for_everyone()

    # get state dict for saving
    save_state_dict = accelerator.get_state_dict(model)

    if save_state_dict is not None:
        accelerator.print(f"State dict stored in CPU on process {accelerator.process_index}")

        # update state dict
        accelerator.print("save_state_dict", list(save_state_dict.keys()))
        accelerator.print("update_state_dict", list(update_state_dict.keys()))

        for name, param in save_state_dict.items():
            if name in update_state_dict:
                accelerator.print(f"Updating {name} (device = {save_state_dict[name].device})")
                save_state_dict[name] = update_state_dict[name]

        # check sparsity
        accelerator.print("*" * 30)
        accelerator.print("Calculating sparsity for all params in the model after update...")
        sparsity_ratio = check_sparsity_from_state_dict(save_state_dict)
        accelerator.print(f"sparsity sanity check {sparsity_ratio:.4f}")
        accelerator.print("*" * 30)

        # save updated state dict
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            prune_model_save_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=save_state_dict,
        )
        tokenizer.save_pretrained(prune_model_save_path)

    accelerator.wait_for_everyone()
    accelerator.print(f"Model saved to {prune_model_save_path}")
