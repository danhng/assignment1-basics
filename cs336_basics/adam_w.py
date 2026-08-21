import torch.optim as optim
from collections.abc import Callable, Iterable
from typing import Optional
import math
import torch

class AdamW(optim.Optimizer): 
    
    '''
    1. torch.optim.Adam
    The defaults dictionary for Adam typically contains:
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9,0.999))
        eps: Term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay / L2 penalty (default: 0)
        amsgrad: Whether to use the AMSGrad variant (default: False)
        maximize: Maximize the params based on the objective, instead of minimizing (default: False)
        foreach: Whether foreach implementation is used (default: None)
        capturable: Whether the optimizer is safe to capture in CUDA graphs (default: False)
        differentiable: Whether autograd should occur through the optimizer step (default: False)
        fused: Whether the fused implementation is used (default: None)
    '''
    # def __init__(self, params, lr, beta, epsilon, weight_decay_rate): 
    def __init__(self, param_groups: list,  lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8): 
        #validate lr
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps":eps, "weight_decay": weight_decay}
        super().__init__(params=param_groups, defaults=defaults)
    
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for param_group in self.param_groups: 
            #get all the coefficients: betas, eps, weight_decay_rate
            lr = param_group["lr"] # get the learning rate            
            betas = param_group["betas"] # get the learning rate            
            eps = param_group["eps"] # get the learning rate            
            weight_decay_rate = param_group["weight_decay"] # get the learning rate            
            beta_m, beta_v = betas[0], betas[1]
            # for each Parameter tensor. 
            for param in param_group["params"]: 
                state = self.state[param]
                step = state.get("step", 1) # first step is 1
                lr_warm_up = lr * math.sqrt(1 - math.pow(beta_v, step))/(1-math.pow(beta_m, step))
                
                # step 1. learning rate warm up (bias correction)
                # step 2. calculate the m moment
                g_raw = param.grad.data
                
                # step 3. First moment
                m_past = state.get("m", 0) # optimizer state memory
                m_next = g_raw*(1-beta_m)+m_past*beta_m
 
                # step 4. Second moment (Square)
                v_past = state.get("v", 0) # optimizer state memory
                g_raw_square = torch.pow(g_raw, 2)
                v_next = beta_v*v_past + (1-beta_v)*g_raw_square
                
                # step 5. weight decay
                weight_decay = lr * weight_decay_rate * param.data # note: weight_decay should not have anything todo with warmup, just a L2 Loss (depends solely on the weight magnitude)
                
                # step 6. final param update: w_next = w_past - lr_warm_up*m_next/(sqrt(v_next) + epsilon) - weight_decay
                param.data -= weight_decay + lr_warm_up * m_next/(torch.sqrt(v_next) + eps)
                
                # step 7. update states:
                state["step"] = step + 1
                state["m"] = m_next
                state["v"] = v_next
        return loss
                
            
            
            