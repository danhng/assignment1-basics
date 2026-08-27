import math
import torch 
from .utils import scaled_dot_product_attention
from .rope import RotaryPositionalEmbedding
from .linear import Linear

# MHA that uses layer class instead of raw weights
class Multihead_Self_Attention_Layers(torch.nn.Module): 
    #max_seq_length: optional, if supplied then rope will be created as per max_seq_length, otherwise rope will not be used. 
    def __init__(self, d_model: None, num_heads: None, q_proj_weight = None, k_proj_weight = None, v_proj_weight = None, o_proj_weight = None, dtype = None, device = None, use_rope = False, max_seq_length = 4096, theta = 10000):
        super().__init__()
        # init q, k, v, o project weights
        self.q_proj = Linear(in_features=d_model, out_features=d_model, dtype=dtype, device=device, weight=q_proj_weight)
        self.k_proj = Linear(in_features=d_model, out_features=d_model, dtype=dtype, device=device, weight=k_proj_weight)
        self.v_proj = Linear(in_features=d_model, out_features=d_model, dtype=dtype, device=device, weight=v_proj_weight)
        self.output_proj = Linear(in_features=d_model, out_features=d_model, dtype=dtype, device=device, weight=o_proj_weight)
        self.num_heads = num_heads
        self.d_model = d_model
        self.rope = None
        self.device = device
        if use_rope:
            self.max_seq_length = max_seq_length
            self.rope = RotaryPositionalEmbedding(theta, int(d_model/num_heads), max_seq_length, device) # rope is applied per head so d_k of rope = d_model/num_heads
                
    # x: (... sequence_length d_model)
    def forward(self, x, token_positions = None): 
        '''
        1. Calculate Q = Wq*x, K=Wk*x, V=Wv*x
        2. Split Qi, Ki, Vi 
        3. Calculate Attention(Qi, Ki, Vi) 
        4. Concate h(Qi, Ki, Vi) 
        5. Calcuate multi head attentions Wo(4) 
        '''
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # divide the d_model by num_heads and transpose num_heads (-2) with seq_length (-3). So we have the seq_len, d_k pair as trailing dimensions. 
        Q_heads = Q.unflatten(dim=-1, sizes=(self.num_heads, int(self.d_model/self.num_heads))).transpose(-2, -3)
        K_heads = K.unflatten(dim=-1, sizes=(self.num_heads, int(self.d_model/self.num_heads))).transpose(-2, -3)
        V_heads = V.unflatten(dim=-1, sizes=(self.num_heads, int(self.d_model/self.num_heads))).transpose(-2, -3)

        # todo: add RoPE if rope layer is on 
        seq_length = x.shape[-2]
        if (self.rope is not None):
            if token_positions is None:
                token_positions = torch.arange(0, seq_length, dtype=torch.int) # if token position is not on
            Q_heads = self.rope(Q_heads, token_positions)
            K_heads = self.rope(K_heads, token_positions)
        # create the mask from triu along the seq_len
        mask = torch.ones((seq_length, seq_length), dtype=torch.bool, device=self.device)
        mask = torch.tril(mask)

        attention_heads = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)
        # concat attention_heads (... seq_length d_model)
        attention_heads_concat = attention_heads.transpose(-2, -3).flatten(-2, -1)
        # multi_head_self_attention = attention_heads_concat @ self.w_o.mT
        multi_head_self_attention = self.output_proj(attention_heads_concat)
        return multi_head_self_attention
    
t = torch.Tensor([[1, 2],[3, 4], [5, 6], [7, 8]]) # 4 x 2
trailing_shape = t.shape[1:] 
new_b = t.shape[0]/2 
print(t.reshape(2, int(new_b), *trailing_shape))