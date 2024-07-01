# Evaluation by the Benchmark Performance

Here we use the [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for benchmarking the compressed models.

Make sure that you have installed the `./lm-evaluation-harness` directory by using the command `pip install -e .[dev]`.

The only change we make is updating the model classes in `./lm-evaluation-harness/lm_eval/models/huggingface.py`. If you want to add more models, please refer to this file.
