import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


@dataclass
class FreezeArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """

    name_module_trainable: Optional[str] = field(
        default=None,
        metadata={
            "help": """Name of trainable modules for partial-parameter (freeze) fine-tuning. \
                    Use commas to separate multiple modules. \
                    Use "all" to specify all the available modules. \
                    LLaMA choices: ["mlp", "self_attn"], \
                    BLOOM & Falcon & ChatGLM choices: ["mlp", "self_attention"], \
                    Qwen choices: ["mlp", "attn"], \
                    InternLM2 choices: ["feed_forward", "attention"], \
                    Others choices: the same as LLaMA."""
        },
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "The number of trainable layers for partial-parameter (freeze) fine-tuning."},
    )


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: Optional[str] = field(
        default=None,
        metadata={
            "help": """Name(s) of target modules to apply LoRA. \
                    Use commas to separate multiple modules. \
                    Use "all" to specify all the available modules. \
                    LLaMA choices: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    BLOOM & Falcon & ChatGLM choices: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], \
                    Baichuan choices: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    Qwen choices: ["c_attn", "attn.c_proj", "w1", "w2", "mlp.c_proj"], \
                    InternLM2 choices: ["wqkv", "wo", "w1", "w2", "w3"], \
                    Others choices: the same as LLaMA."""
        },
    )
    lora_bf16_mode: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to train lora adapters in bf16 precision."},
    )
    use_rslora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    create_new_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class FinetuningArguments(FreezeArguments, LoraArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    finetuning_type: Optional[Literal["lora", "freeze", "full"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    disable_version_checking: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable version checking."},
    )
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.name_module_trainable = split_arg(self.name_module_trainable)
        self.lora_alpha = self.lora_alpha or self.lora_rank * 2
        self.lora_target = split_arg(self.lora_target)
        self.additional_target = split_arg(self.additional_target)

        assert self.finetuning_type in ["lora", "freeze", "full"], "Invalid fine-tuning method."

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for the Freeze or LoRA method.")

    def save_to_json(self, json_path: str):
        r"""Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        r"""Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()

        return cls(**json.loads(text))
