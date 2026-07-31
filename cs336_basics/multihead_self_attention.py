import math
import torch 
import einops 

class Multihead_Self_Attention(torch.nn.Module): 
    def __init__(self, d_model: int, num_heads: int, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, dtype = None, device = None):
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
    
    def _init_linear_weight(d_in, d_out, d_type, device): 
        std_init_weight = math.sqrt(2/(d_in+d_out))
        w_o = torch.empty(size = (d_out, d_in), dtype=d_type, device=device) # initialize based on Xavier
        torch.nn.init.trunc_normal_(w_o, mean=0, std=std_init_weight, a=-3 * std_init_weight, b = 3*std_init_weight)
        return torch.nn.Parameter(w_o)
            
    # x: (... sequence_length d_model)
    def forward(self, x): 
        '''
        1. Calculate Q = Wq*x, K=Wk*x, V=Wv*x
        2. Split Qi, Ki, Vi 
        3. Calculate Attention(Qi, Ki, Vi)
        4. Concate h(Qi, Ki, Vi)
        5. Calcuate multi head attentions Wo(4)
        '''
        
        Q = torch.matmul(x, self.w_q.T) 
        K = torch.matmul(x, self.w_k.T) 
        V = torch.matmul(x, self.w_v.T)
        
        Q_heads = Q.reshape(*Q.shape[:-2], self.num_heads, self.Q.shape[-2], int(self.d_model/self.num_heads)) # divide the d_model and transpose num_heads with seq_length. So we have the seq_len, d_k pair as trailing dimensions. 
        K_heads = K.reshape(*K.shape[:-2], self.num_heads, self.K.shape[-2], int(self.d_model/self.num_heads))
        V_heads = V.reshape(*V.shape[:-2],  self.num_heads, self.V.shape[-2], int(self.d_model/self.num_heads))
        
        attention_Qi = None
        attention_Ki = None
        attention_Vi = None
        
        attention_concat_Q = None
        attention_concat_K = None
        attention_concat_V = None
        
        multi_head_self_attention = None
        return multi_head_self_attention
    
t = torch.Tensor([[1, 2],[3, 4], [5, 6], [7, 8]]) # 4 x 2
# trailing_shape = t.shape[1:] 
# new_b = t.shape[0]/2 
# print(t.reshape(2, int(new_b), *trailing_shape))
