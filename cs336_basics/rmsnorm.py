import torch

class RMSNorm(torch.nn.Module): 
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None, weights = None): 
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype
        self.device = device
        if weights is None: 
            weight = torch.ones((d_model), dtype=dtype, device=device)
            self.weight = torch.nn.Parameter(weight) # gamma
        else: 
            weights.to(device)
            weights.to(dtype)
            self.weight = torch.nn.Parameter(weights)
    
    #Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.to(torch.float32)
        r_rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # x = x.to(x_dtype)
        x_rms_normed = x * r_rms * self.weight
        return x_rms_normed.to(x_dtype)

# in_features = torch.Tensor([1, 2, 3])
# RMSLayer = RMSNorm(3, 1e-5, None, None)
# print(RMSLayer.forward(in_features))
    # raise NotImplementedError
        