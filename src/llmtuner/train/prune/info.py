from torch import cuda


def print_gpu_memory(accelerator):
    if accelerator.is_local_main_process:  # 🔍
        for i in range(cuda.device_count()):
            used_memory = cuda.memory_allocated(0) // 1024 ** 2
            print(f"GPU {i} Used Memory: {used_memory}MB")
