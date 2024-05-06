import sys
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os
import json

model_path = sys.argv[1]
quant_path = sys.argv[2]
bits = sys.argv[3]

# model_path = "/mnt/petrelfs/share_data/quxiaoye/models/Mistral-7B-v0.1/"
# model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
# quant_path = '/mnt/petrelfs/dongdaize.d/workspace/compression/src/llmtuner/train/quantization/AutoAWQ/mistral-instruct-v0.2-awq'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": int(bits), "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    use_fast=False, 
    trust_remote_code=True
    )

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
f = open(os.path.join(quant_path, "quantize_config.json"), 'w')
config_to_save = json.dumps(quant_config, indent=2, sort_keys=True)
f.write(config_to_save)
f.close()
print(f'Model is quantized and saved at "{quant_path}"')