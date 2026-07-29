import math
import torch 

class Multihead_Self_Attention(torch.nn.Module): 
    def __init__(self, d_model: int, num_heads: int, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, dtype = None, device = None):
        super().__init__()
        if q_proj_weight is not None: 
            self.w_q = q_proj_weight
        else: 
            w_q = torch.empty(size = (d_model, d_model), dtype=dtype, device=device) # initialize based on Xavier
            std_init_weight = math.sqrt(2/(d_model+d_model))
            torch.nn.init.trunc_normal_(w_q, mean=0, std=std_init_weight, a=-3 * std_init_weight, b = 3*std_init_weight)
            self.w_q = torch.nn.Parameter(w_q)