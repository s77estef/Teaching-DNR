import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / (1024 ** 3)
    print(f"Total VRAM: {total_vram_gb:.2f} GB")
else:
    print("No CUDA GPU detected.")

