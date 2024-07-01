import logging
import os
import sys
from dataclasses import field, dataclass
from typing import Any, Dict, Optional, Tuple, Literal

import datasets
import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .compression_args import CompressionArguments
from .data_args import DataArguments
from .model_args import ModelArguments
from ..extras.logging import get_logger
from ..extras.packages import is_unsloth_available

logger = get_logger(__name__)


_COMPRESSION_ARGS = [ModelArguments, DataArguments, Seq2SeqTrainingArguments, CompressionArguments]  # 🔍
_COMPRESSION_CLS = Tuple[ModelArguments, DataArguments, Seq2SeqTrainingArguments, CompressionArguments]  # 🔍


def _parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return (*parsed_args,)


def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def _parse_compression_args(args: Optional[Dict[str, Any]] = None) -> _COMPRESSION_CLS:
    parser = HfArgumentParser(_COMPRESSION_ARGS)
    return _parse_args(parser, args)


def get_compression_args(args: Optional[Dict[str, Any]] = None) -> _COMPRESSION_CLS:
    model_args, data_args, training_args, compression_args = _parse_compression_args(args)

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    # Check arguments
    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True while training.")

    if training_args.do_train and model_args.use_unsloth and not is_unsloth_available:
        raise ValueError("Install Unsloth: https://github.com/unslothai/unsloth")

    if training_args.do_train and model_args.quantization_bit is not None and (not model_args.upcast_layernorm):
        logger.warning("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning("We recommend enable mixed precision training.")

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    # Post-process training arguments
    can_resume_from_checkpoint = True

    if (
            training_args.resume_from_checkpoint is None
            and training_args.do_train
            and os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
            and can_resume_from_checkpoint
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args_dict = training_args.to_dict()
            training_args_dict.update(dict(resume_from_checkpoint=last_checkpoint))
            training_args = Seq2SeqTrainingArguments(**training_args_dict)
            logger.info(
                "Resuming training from {}. Change `output_dir` or use `overwrite_output_dir` to avoid.".format(
                    training_args.resume_from_checkpoint
                )
            )

    # Post-process model arguments
    model_args.compute_dtype = (
        torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
    )
    model_args.model_max_length = data_args.cutoff_len

    # Log on each process the small summary:
    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}\n  distributed training: {}, compute dtype: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            str(model_args.compute_dtype),
        )
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, compression_args
