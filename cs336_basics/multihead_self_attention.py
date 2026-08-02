import math
import torch 
from .utils import scaled_dot_product_attention
from .rope import RotaryPositionalEmbedding

class Multihead_Self_Attention(torch.nn.Module): 
    #max_seq_length: optional, if supplied then rope will be created as per max_seq_length, otherwise rope will not be used. 
    def __init__(self, d_model: None, num_heads: None, q_proj_weight = None, k_proj_weight = None, v_proj_weight = None, o_proj_weight = None, dtype = None, device = None, max_seq_length = None, theta = None):
        super().__init__()
        # init project weights
        if q_proj_weight is not None: 
            self.w_q = q_proj_weight
        else: 
            self.w_q = self._init_linear_weight(d_model, d_model, dtype, device)
        
        if k_proj_weight is not None: 
            self.w_k = k_proj_weight
        else: 
            self.w_k = self._init_linear_weight(d_model, d_model, dtype, device)
            
        if v_proj_weight is not None: 
            self.w_v = v_proj_weight
        else: 
            self.w_v = self._init_linear_weight(d_model, d_model, dtype, device)
            
        if o_proj_weight is not None: 
            self.w_o = o_proj_weight
        else: 
            self.w_o = self._init_linear_weight(d_model, d_model, dtype, device)
            
        self.num_heads = num_heads
        self.d_model = d_model
        self.rope = None
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
            self.rope = RotaryPositionalEmbedding(theta, int(d_model/num_heads), max_seq_length, device) # rope is applied per head so d_k of rope = d_model/num_heads
            
    
    def _init_linear_weight(d_in, d_out, d_type, device): 
        std_init_weight = math.sqrt(2/(d_in+d_out))
        w_o = torch.empty(size = (d_out, d_in), dtype=d_type, device=device) # initialize based on Xavier
        torch.nn.init.trunc_normal_(w_o, mean=0, std=std_init_weight, a=-3 * std_init_weight, b = 3*std_init_weight)
        return torch.nn.Parameter(w_o)
            
    # x: (... sequence_length d_model)
    def forward(self, x, token_positions = None): 
        '''
        1. Calculate Q = Wq*x, K=Wk*x, V=Wv*x
        2. Split Qi, Ki, Vi 
        3. Calculate Attention(Qi, Ki, Vi) 
        4. Concate h(Qi, Ki, Vi) 
        5. Calcuate multi head attentions Wo(4) 
        '''
        Q = torch.matmul(x, self.w_q.mT) 
        K = torch.matmul(x, self.w_k.mT) 
        V = torch.matmul(x, self.w_v.mT)
        
        # divide the d_model by num_heads and transpose num_heads (-2) with seq_length (-3). So we have the seq_len, d_k pair as trailing dimensions. 
        Q_heads = Q.unflatten(dim=-1, sizes=(self.num_heads, int(self.d_model/self.num_heads))).transpose(-2, -3)
        K_heads = K.unflatten(dim=-1, sizes=(self.num_heads, int(self.d_model/self.num_heads))).transpose(-2, -3)
        V_heads = V.unflatten(dim=-1, sizes=(self.num_heads, int(self.d_model/self.num_heads))).transpose(-2, -3)

        # todo: add RoPE if rope layer is on and token_positions is supplied
        if (self.rope is not None):
            if token_positions is None:
                token_positions = torch.range(0, self.max_seq_length-1)
            Q_heads = self.rope.forward(Q_heads, token_positions)
            K_heads = self.rope.forward(K_heads, token_positions)
        
        seq_length = x.shape[-2]
        print("seq length:" , seq_length)
        # create the mask from triu along the seq_len
        mask = torch.ones((seq_length, seq_length), dtype=torch.bool)
        mask = torch.tril(mask)

        attention_heads = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)
        # concat attention_heads (... seq_length d_model)
        attention_heads_concat = attention_heads.transpose(-2, -3).flatten(-2, -1)
        multi_head_self_attention = attention_heads_concat @ self.w_o.mT
        return multi_head_self_attention
    
t = torch.Tensor([[1, 2],[3, 4], [5, 6], [7, 8]]) # 4 x 2
trailing_shape = t.shape[1:] 
new_b = t.shape[0]/2 
print(t.reshape(2, int(new_b), *trailing_shape))
