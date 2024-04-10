from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Literal


@dataclass
class PruningArguments:
    r"""
    Arguments pertaining to specify the decoding parameters.
    """
    prune_seed: Optional[int] = field(
        default=42,
        metadata={"help": "Seed for sampling the calibration data."},
    )
    n_calibration_samples: Optional[int] = field(
        default=128,
        metadata={"help": "Number of calibration samples."},
    )
    sparsity_ratio: Optional[float] = field(
        default=0.,
        metadata={"help": "Sparsity Level."},
    )
    sparsity_type: Optional[Literal["unstructured", "4:8", "2:4"]] = field(
        default="unstructured",
        metadata={"choices": ["unstructured", "4:8", "2:4"]},
    )

    prune_method: Optional[
        Literal["wanda", "sparsegpt", "gradient-first", "gradient-zeroth", "magnitude",
        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"]
    ] = field(
        default="wanda",
        metadata={"choices": ["wanda", "sparsegpt", "gradient-first", "gradient-zeroth", "magnitude",
                              "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"]},
    )
    use_variant: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the variant for Wanda."},
    )

    # 🔍
    prune_model_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the pruned model."},
    )
    prune_data_type: Literal["pt", "sft", "rm", "ppo"] = field(
        default="sft",
        metadata={"choices": ["pt", "sft", "rm", "ppo"],
                  "help": "Path to save the pruned model."},
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args
