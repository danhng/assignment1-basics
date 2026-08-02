from torch import nn 

'''
Let's begin by assembling the Transformer block (it will be helpful to refer back to Figure 2). A 
Transformer block contains two 'sub-layers', one for the multihead self attention, and another for the 
SwiGLU feed-forward network. In each sub-layer, we first perform RMSNorm, then the main operation 
(MHA/FF), finally adding in the residual connection.
To be concrete, the first half (the first 'sub-layer') of the Transformer block should be implementing the 
following set of updates to produce an output y from an input x,
y = x + MultiHeadSelfAttention(RMSNorm(x))
'''
class TransformerBlock(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, dtype, device, weights = None, use_rope = False, max_seq_len = 4096, theta = 10000): 
        