import os

from accelerate import Accelerator


# 🔍
def save_sparse_model(args, model, tokenizer, accelerator: Accelerator, update_state_dict):
    if args.prune_model_save_path is not None:
        accelerator.print("Saving models... (may take minutes)")
        if accelerator.is_main_process:
            if not os.path.exists(args.prune_model_save_path):
                os.makedirs(args.prune_model_save_path)
        accelerator.wait_for_everyone()

        # update state dict
        save_state_dict = accelerator.get_state_dict(model)
        for name, param in save_state_dict.items():
            if name in update_state_dict:
                save_state_dict[name] = update_state_dict[name].to(save_state_dict[name].device)

        # save
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.prune_model_save_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=save_state_dict,
        )
        tokenizer.save_pretrained(args.prune_model_save_path)
        accelerator.print(f"Model saved to {args.prune_model_save_path}")
