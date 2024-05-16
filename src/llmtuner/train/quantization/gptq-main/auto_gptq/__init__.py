import sys

transformers_path = "/mnt/petrelfs/dongdaize.d/workspace/compression/src"
sys.path = [transformers_path] + sys.path

from .modeling import AutoGPTQForCausalLM, BaseQuantizeConfig
from .utils.exllama_utils import exllama_set_max_input_length
from .utils.peft_utils import get_gptq_peft_model

__version__ = "0.7.1"
