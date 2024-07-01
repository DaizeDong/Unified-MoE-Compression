from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .prune import run_prune
from .pt import run_pt
from ..extras.callbacks import LogCallback
from ..extras.logging import get_logger
from ..hparams import get_compression_args

if TYPE_CHECKING:
    from transformers import TrainerCallback

logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args,  compression_args = get_compression_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    if finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "prune":  # 🔍
        run_prune(model_args, data_args, training_args, compression_args)
    else:
        raise ValueError("Unknown task.")


if __name__ == "__main__":
    run_exp()
