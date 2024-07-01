from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Literal

EXPERT_DROP_METHODS = ('global_pruning', 'layerwise_pruning', 'post_dropping')
LAYER_DROP_METHODS = ('discrete', 'post_dropping')
BLOCK_DROP_METHODS = ('discrete', 'consecutive', 'post_dropping')


@dataclass
class CompressionArguments:
    r"""
    Arguments about compression.
    """
    stage: Optional[Literal["prune", "pt"]] = field(
        default="prune",
        metadata={"help": "Which stage will be performed."},
    )
    compress_method: Optional[str] = field(
        default="wanda",
        metadata={"choices": ["magnitude", "wanda", "sparsegpt", "expert_drop", "block_drop", "layer_drop"]},
    )
    compressed_model_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the compressed model."},
    )
    n_compression_samples: Optional[int] = field(
        default=128,
        metadata={"help": "Number of samples to use for compression."},
    )
    data_type: Literal["pt", "sft"] = field(
        default="sft",
        metadata={"choices": ["pt", "sft"],
                  "help": "Type of the dataset. This decides the way that data are processed."
                          "(\"pt\": truncate & no padding. \"sft\": with padding.)"},
    )

    # 🔍 For pruning
    sparsity_ratio: Optional[float] = field(  # this term also denotes the "parameter_ratio" for decomposition
        default=0.5,
        metadata={"help": "Sparsity Level."},
    )
    sparsity_type: Optional[Literal["unstructured", "4:8", "2:4"]] = field(
        default="unstructured",
        metadata={"choices": ["unstructured", "4:8", "2:4"]},
    )
    exclude_prune_module_name: Optional[str] = field(
        default=None,
        metadata={"help": "Module names to exclude when pruning. (Use \",\" to separate different modules if excluding more than one module)"},
    )
    use_variant: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the variant for Wanda."},
    )

    # 🔍 For expert drop
    expert_drop_method: Optional[str] = field(
        default="layerwise_pruning",
        metadata={"help": ' '.join(['Supported dropping methods:'] + list(EXPERT_DROP_METHODS)),
                  "choices": EXPERT_DROP_METHODS},
    )
    preserve_n: Optional[int] = field(
        default=4,
        metadata={"help": 'Number of experts to preserve'}
    )
    preserve_gate: Optional[bool] = field(
        default=False,
        metadata={"help": 'Whether to preserve the dimension of the gate. (Available only in \"post_dropping\" mode) '
                          'If True, the gate weights of the corresponding experts will be re-ordered instead of being pruned. '
                          'The model will dynamically drop tokens according to the gate selections instead of redirecting to other experts.'},
    )
    reverse_drop: Optional[bool] = field(
        default=False,
        metadata={"help": 'Whether to drop the experts with the highest gating scores.'},
    )
    score_save_file: Optional[str] = field(
        default=None,
        metadata={"help": 'File to save the routing scores across layers for further analysis.'},
    )

    # 🔍 For layer drop & block drop
    layer_drop_method: Optional[str] = field(
        default="discrete",
        metadata={"help": ' '.join(['Supported dropping methods:'] + list(LAYER_DROP_METHODS)),
                  "choices": LAYER_DROP_METHODS},
    )
    layer_drop_norm: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to drop the LayerNorm before MoE."},
    )
    block_drop_method: Optional[str] = field(
        default="discrete",
        metadata={"help": ' '.join(['Supported dropping methods:'] + list(BLOCK_DROP_METHODS)),
                  "choices": BLOCK_DROP_METHODS},
    )
    drop_n: Optional[int] = field(
        default=4,
        metadata={"help": 'Number of blocks to drop'}
    )
    similarity_cache_file: Optional[str] = field(
        default=None,
        metadata={"help": 'Cached file storing the similarity scores across layers to reduce the computation consumption. '
                          'If the file does not exist, it will be created.'},
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args
