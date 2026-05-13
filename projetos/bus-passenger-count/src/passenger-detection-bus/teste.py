import torch

def check_cuda_availability() -> None:
    f"""
    Checks if PyTorch is correctly configured to use the NVIDIA GPU via CUDA.
    Prints the result and the name of the GPU if available.
    """
    is_available: bool = torch.cuda.is_available()
    print(f"CUDA Available: {is_available}")
    
    if is_available:
        gpu_name: str = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {gpu_name}")
    else:
        print("Warning: CUDA is not available. PyTorch is using the CPU.")

if __name__ == "__main__":
    check_cuda_availability()