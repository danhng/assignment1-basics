import math
from torch.optim.lr_scheduler import LRScheduler

class CosineAnnealingLR(LRScheduler): 
    def __init__(self, optimizer, max_iters, warmup_iter_ratio, lr_min, cosine_iter_ratio): 
        self.max_iters = max_iters
        self.lr_min = lr_min
        self.warmup_iters = int(max_iters * warmup_iter_ratio)
        self.cosine_iters = int(max_iters * cosine_iter_ratio + self.warmup_iters)
        super().__init__(optimizer=optimizer, last_epoch=-1)
        
    def get_lr(self): 
        return [CosineAnnealingLR.single_lr(t=self.last_epoch, lr_max=base_lr, lr_min=self.lr_min, T_w=self.warmup_iters, T_c=self.cosine_iters) 
             for base_lr in self.base_lrs]
    
    '''
    The cosine annealing learning rate schedule takes (i) the current iteration t, (ii) the maximum learningrate lr_max, the minimum (final) learning rate lr_min, 
    the number of warm-up iterations T_w, and the final iteration of cosine annealing T_c. 
    '''
    @classmethod
    def single_lr(cls, t, lr_max, lr_min, T_w, T_c): 
        assert T_w <= T_c and t >= 0
        if t < T_w: 
            return t/T_w * lr_max
        if t > T_c: 
            return lr_min
        return lr_min + 1/2*(lr_max-lr_min)*(1+math.cos(math.pi * (t-T_w)/(T_c - T_w)))    
