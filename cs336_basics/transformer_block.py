from torch import nn 
from .multihead_self_attention_layers import Multihead_Self_Attention_Layers
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU_FFN

import logging
logger = logging.getLogger('TransformerBlock')


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
    def __init__(self, block_index, d_model, num_heads, d_ff, dtype=None, device=None, weights = None, use_rope = False, max_seq_len = 4096, theta = 10000): 
        super().__init__()
        # First half
        # LayerNorm1
        # rmsnorm_1 = weights["ln1"]
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        # self.rms_norm1.load_state_dict(rmsnorm_1)
          # MHA
        # attn = weights["attn"]
        self.attn = Multihead_Self_Attention_Layers(d_model=d_model, num_heads=num_heads, dtype=dtype, device=device, use_rope=use_rope, max_seq_length=max_seq_len, theta=theta)
        # self.attn.load_state_dict(attn)
        
        # Second half
        # LayerNorm2
        # rmsnorm_2 = weights["ln2"]
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        # self.rms_norm2.load_state_dict(rmsnorm_2)
        # FFN
        # ffn = weights["ffn"]
        self.ffn = SwiGLU_FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        # self.ffn.load_state_dict(ffn)
        if (weights is not None): 
            self.load_state_dict(weights)
        self.block_index = block_index

    def get_block_index(self):
        return self.block_index
    '''
     Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    '''
    def forward(self, in_features): 
        logger.debug(f"Transformer Block{self.block_index}: In feature: {in_features.size()}" )
        out_1_rmsnorm = self.ln1(in_features) # rms norm 1
        out_1_mha = self.attn(out_1_rmsnorm) # mha
        out_1 = in_features + out_1_mha # residual connection
        
        out_2_rms_norm = self.ln2(out_1) # rms_norm 2
        out_2_ffn = self.ffn(out_2_rms_norm) # swiglu
        out_2 = out_1 + out_2_ffn # residual connection
        logger.debug(f"Transformer Block{self.block_index}: output size:  {out_2.size()}")
        return out_2