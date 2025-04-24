import torch

def check_cuda_status():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"CUDA is available. {num_devices} GPU(s) detected.\n")

        for i in range(num_devices):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total memory: {round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)} GB")
            print(f"  Allocated memory: {round(torch.cuda.memory_allocated(i) / 1024**3, 2)} GB")
            print(f"  Cached memory: {round(torch.cuda.memory_reserved(i) / 1024**3, 2)} GB\n")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_cuda_status()

