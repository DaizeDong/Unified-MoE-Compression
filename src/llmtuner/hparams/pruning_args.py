from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Literal

EXPERT_DROP_METHODS = ('layerwise_pruning', 'progressive_pruning', 'dynamic_skipping')


@dataclass
class PruningArguments:
    r"""
    Arguments pertaining to specify the decoding parameters.
    """
    prune_seed: Optional[int] = field(
        default=42,
        metadata={"help": "Seed for sampling the calibration data."},
    )
    prune_model_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the pruned model."},
    )
    n_calibration_samples: Optional[int] = field(
        default=128,
        metadata={"help": "Number of calibration samples."},
    )
    prune_data_type: Literal["pt", "sft", "rm", "ppo"] = field(
        default="sft",
        metadata={"choices": ["pt", "sft", "rm", "ppo"],
                  "help": "Path to save the pruned model."},
    )

    # 🔍 For pruning
    sparsity_ratio: Optional[float] = field(  # parameter_ratio when using decomposition
        default=0.5,
        metadata={"help": "Sparsity Level."},
    )
    sparsity_type: Optional[Literal["structured", "unstructured", "4:8", "2:4"]] = field(
        default="unstructured",
        metadata={"choices": ["structured", "unstructured", "4:8", "2:4"]},
    )
    prune_method: Optional[str] = field(
        default="wanda",
        metadata={"choices": ["wanda", "sparsegpt", "gradient-first", "gradient-zeroth", "magnitude", "remap_gate", "decompose_moe", "expert_drop"]},
    )
    use_variant: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the variant for Wanda."},
    )
    
    # 🔍 For expert drop
    expert_drop_method: Optional[str] = field(
        default="layerwise_pruning",
        metadata={"help": ' '.join(['Supported pruning methods:'] + list(EXPERT_DROP_METHODS)),
                  "choices": EXPERT_DROP_METHODS},
    )
    r: Optional[int] = field(
        default=4,
        metadata={"help": 'Number of experts to preserve'}
    )

    # 🔍 For decomposition
    level: Optional[str] = field(
        default="expert",
        metadata={"choices": ["expert", "layer", "model"]},
    )
    has_sparse: Optional[bool] = field(
        default=True,
    )
    do_permute: Optional[bool] = field(
        default=True,
    )
    use_svd: Optional[bool] = field(
        default=True,
    )

    # 🔍 For gate-remapping
    pruned_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pruned model. (Only for Gate-Remapping)"},
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args
