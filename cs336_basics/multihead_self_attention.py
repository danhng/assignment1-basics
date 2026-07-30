import math
import torch 

class Multihead_Self_Attention(torch.nn.Module): 
    def __init__(self, d_model: int, num_heads: int, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, dtype = None, device = None):
        super().__init__()
        std_init_weight = math.sqrt(2/(d_model+d_model))
        
        # init project weights
        if q_proj_weight is not None: 
            self.w_q = q_proj_weight
        else: 
            w_q = torch.empty(size = (d_model, d_model), dtype=dtype, device=device) # initialize based on Xavier
            torch.nn.init.trunc_normal_(w_q, mean=0, std=std_init_weight, a=-3 * std_init_weight, b = 3*std_init_weight)
            self.w_q = torch.nn.Parameter(w_q)
        
        if k_proj_weight is not None: 
            self.w_k = k_proj_weight
        else: 
            w_k = torch.empty(size = (d_model, d_model), dtype=dtype, device=device) # initialize based on Xavier
            torch.nn.init.trunc_normal_(w_k, mean=0, std=std_init_weight, a=-3 * std_init_weight, b = 3*std_init_weight)
            self.w_k = torch.nn.Parameter(w_k)
            
        if v_proj_weight is not None: 
            self.w_v = v_proj_weight
        else: 
            w_v = torch.empty(size = (d_model, d_model), dtype=dtype, device=device) # initialize based on Xavier
            torch.nn.init.trunc_normal_(w_v, mean=0, std=std_init_weight, a=-3 * std_init_weight, b = 3*std_init_weight)
            self.w_v = torch.nn.Parameter(w_v)
            
        if o_proj_weight is not None: 
            self.w_o = o_proj_weight
        else: 
            w_o = torch.empty(size = (d_model, d_model), dtype=dtype, device=device) # initialize based on Xavier
            torch.nn.init.trunc_normal_(w_o, mean=0, std=std_init_weight, a=-3 * std_init_weight, b = 3*std_init_weight)
            self.w_o = torch.nn.Parameter(w_o)
    0
    def forward(self, in_features): 
        '''
        1. Calculate Q, K, V
        '''
        pass