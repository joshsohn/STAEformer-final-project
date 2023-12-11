import torch

# Display GPU memory summary
torch.cuda.memory_summary(device=0, abbreviated=False)
