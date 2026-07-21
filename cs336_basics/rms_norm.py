import torch

class RMSNorm(torch.nn.Module): 
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None, weights = None): 
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype
        self.device = device
        if weights is None: 
            gammas = torch.ones((d_model), dtype=dtype, device=device)
            self.gammas = torch.nn.Parameter(gammas)
        else: 
            weights.to(device)
            weights.to(dtype)
            self.gammas = torch.nn.Parameter(weights)
    
    #Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2) + self.eps)
        # x = x.to(x_dtype)
        x_rms_normed = torch.div(x, rms) * self.gammas
        return x_rms_normed.to(x_dtype)
        
        