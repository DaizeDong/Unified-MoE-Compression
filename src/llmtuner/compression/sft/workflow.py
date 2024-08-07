# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

import json
import os
from typing import TYPE_CHECKING, List, Optional

from llmtuner.compression.sft.metric import ComputeMetrics
from llmtuner.compression.sft.trainer import CustomSeq2SeqTrainer
from llmtuner.data import get_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.ploting import plot_loss
from llmtuner.model import load_model_and_tokenizer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import DataArguments, FinetuningArguments, ModelArguments


def run_sft(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        callbacks: Optional[List["TrainerCallback"]] = None,
):
    # with open(os.path.join(model_args.model_name_or_path, "config.json")) as f:
    #     config = json.load(f)
    #     config["mode"] = "dynamic"
    #
    # f = open(os.path.join(model_args.model_name_or_path, "config.json"), 'w')
    # config_to_save = json.dumps(config, indent=2, sort_keys=True)
    # f.write(config_to_save)
    # f.close()

    model, tokenizer = load_model_and_tokenizer(model_args, training_args.do_train, finetuning_args=finetuning_args)

    for name, params in model.named_parameters():
        if params.requires_grad:
            print(name)

    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")
    print(f"dataset: {dataset}")

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(
        dict(
            generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
            generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams,
        )
    )
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
