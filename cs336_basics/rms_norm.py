import torch

class RMSNorm(torch.Module): 
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None): 
        super().init()
        self.d_model = d_model
        