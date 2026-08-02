import torch
from einops import repeat

class RotaryPositionalEmbedding(torch.nn.Module): 
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, rotate_adjacent = True): 
        super().__init__()
        self.theta = theta # the theta of the RoPE
        self.d_k = d_k # dimension of query and key vector
        self.max_seq_len = max_seq_len # maximum input sequence length
        self.device = device # Device to store the buffer on
        self.rotated_adjacent = rotate_adjacent # Device to store the buffer on
        # todo: calculate the cos, sin cache (size max_sequence_len, d_k) (cos(i, k) = cos(i*theta^(-2*k/d)))
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/9
        self.cos_cache = torch.zeros(max_seq_len, d_k//2)
        range_d_k = torch.arange(0, d_k, 2) # size d_k/2
        range_d_k = range_d_k.unsqueeze(0) # size 1, d_k/2
        
        range_token_positions_i = torch.arange(max_seq_len) 
        range_token_positions_i = range_token_positions_i.unsqueeze(1) # size max_seq_len, 1
        
        range_exp_angle = range_token_positions_i * torch.pow(theta, -range_d_k.float()/d_k) # max_seq_len, d_k/2
        self.cos_cache = torch.cos(range_exp_angle)
        self.sine_cache = torch.sin(range_exp_angle)
        self.register_buffer('cos_cache_buffer', self.cos_cache, persistent=False)
        self.register_buffer('sine_cache_buffer', self.sine_cache, persistent=False)
        self.to(device)
        # we choose the rotate_half strategy
        '''
        Example: 
        ROTATE_HALF
            x1' = x1.cos(token_pos,k=1 (pair 1), theta) - x3.sin(token_pos, k=1 (pair 1), theta)
            x2' = x2.cos(token_pos,k=2 (pair 2), theta) - x4.sin(token_pos, k=2 (pair 2), theta)
            x3' = x3.cos(token_pos,k=1 (pair 1), theta) + x1.sin(token_pos, k=1 (pair 1), theta)
            x4' = x4.cos(token_pos,k=2 (pair 2), theta) + x2.sin(token_pos, k=2 (pair 2), theta)
            First half: x1, x2
            Second half: -x3, -x4
        
         
        '''

    # [1, 2, 3, 4] -> [-3, -4, 1, 2]
    def _rotate_half(self, x: torch.Tensor): 
        x_split_second_half = torch.neg(x[...,self.d_k//2:]) # (seq_length, d_k/2) x_split_second_half = [-3, -4]
        x_rotated = torch.cat((x_split_second_half,x[...,:self.d_k//2]), dim=-1) # (seq_length, d_k), x_rotated = [-3, -4, 1, 2]
        return x_rotated
    
    # [1, 2, 3, 4] -> [1, -2, 3, -4]
    def _rotate_adjacent(self, x: torch.Tensor): 
        even = x[..., 0::2] # 1, 3
        odd = -x[..., 1::2] # -2, -4
        return torch.stack((odd, even), dim=-1).flatten(-2)

    '''
    Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. 
    Note that you should tolerate x with an arbitrary number of batch dimensions. 
    You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension.
    You should use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension.
    '''
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor): 
        # token positions: the true token positions - or m, used to multiplied by theta to get the angle. if we don't pass, the model will treat the token pos as 0, 1, .., seq-1. This might not be desirable as token positions might not be like that during inference and when packed multiple sequences.
        cosines = self.cos_cache_buffer[token_positions] # seq_length, d_k/2
        # cosines = torch.index_select(self.cos_cache_buffer, dim=-2, index=token_positions) # seq_length, d_k/2
        sines = self.sine_cache_buffer[token_positions] # seq_length, d_k/2
        # sines = torch.index_select(self.sine_cache_buffer, dim=-2, index=token_positions) # seq_length, d_k/2
       
        # ex. x = [1, 2, 3, 4]
        x.to(self.device)
        # rotate adjacent
        if (self.rotated_adjacent): 
            x_rotated = self._rotate_adjacent(x)
            cosines_repeated = repeat(cosines, "... d_k -> ... (d_k 2)") # seq_length, d_k 
            sines_repeated = repeat(sines, "... d_k -> ... (d_k 2)")# seq_length, d_k
        #rotate_half
        else: 
            x_rotated = self._rotate_half(x)
            cosines_repeated = repeat(cosines, "... d_k -> ... (2 d_k)") # seq_length, d_k 
            sines_repeated = repeat(sines, "... d_k -> ... (2 d_k)")# seq_length, d_k
        rope = x * cosines_repeated + x_rotated * sines_repeated # 1'= 1*cos1 + (-3)*sin1, 3' = 3*cos1 + 1*sin1
        return rope