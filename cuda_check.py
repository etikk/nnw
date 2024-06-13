import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU for computation.")
    # device = torch.device('cuda')
else:
    print("CUDA is not available. Using CPU for computation.")
    # device = torch.device('cpu')