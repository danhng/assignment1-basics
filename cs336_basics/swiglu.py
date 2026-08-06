from torch import nn  
from .linear import Linear
from torch import sigmoid

'''
Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)
Deliverable: Implement the SwiGLU feed-forward network, composed of a SiLU activation function and a GLU.
Note: in this particular case, you should feel free to use torch.sigmoid in your implementation for numerical stability.
You should set d_ff to approximately 8/3 x d_model in your implementation, while ensuring that the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware. 
To test your implementation against our provided tests, you will need to implement the test adapter at [adapters.run_swiglu] . Then, run uv run pytest -k test_swiglu to test your implementation
'''
class SwiGLU_FFN(nn.Module): 
    # 1. Upsize - Linear 1 (W1: d_ff * d_model)
    # 2. SILU 1
    # 3. Upsize - Linear 3 (W3: d_ff * d_model)
    # 4. Downsize - Linear 2 (W2: d_model * d_ff)
    
    def __init__(self, d_ff, d_model, dtype, device, w1 = None, w2 = None, w3 = None): 
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.dtype = dtype
        self.device = device
        self.w1 = Linear(d_model, d_ff, device, dtype, w1)
        self.w3 = Linear(d_model, d_ff, device, dtype, w3)
        self.w2 = Linear(d_ff, d_model, device, dtype, w2)
    
    def forward(self, x): 
        h1 = self.w1.forward(x) # 1, d_model * d_model,d_ff = 1 * d_ff
        a1 = h1 * sigmoid(h1) # 1 * d_ff 
        h3 = self.w3.forward(x)  # 1 * d_ff todo: could we do h1 and h3 in parallel
        h2 = a1 * h3 # 1 * d_ff
        a2 = self.w2.forward(h2) # 1,d_ff * d_ff,d_model = 1,d_model
        return a2