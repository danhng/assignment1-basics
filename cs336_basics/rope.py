import torch

class RotaryPositionalEmbedding(torch.nn.Module): 
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None): 
        self.theta = theta # the theta of the RoPE
        self.d_k = d_k # dimension of query and key vector
        self.max_seq_len = max_seq_len # maximum input sequence length
        self.device = device # Device to store the buffer on
        # todo: calculate the cos, sin cache (size max_sequence_len, d_k) (cos(i, k) = cos(i*theta^(-2*k/d)))
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/9
        self.rotated_angles_cache = torch.zeros(max_seq_len, d_k)
        # we choose the rotate_half strategy
        '''
        x1' = x1.cos(theta) - x2.sin(theta)
        x2' = x2.cos(theta) + x1.sin(theta)
        rotate_half 
        -> q_rotated = x.cos(i, k, theta) + x_negated.sin(i, k, theta)
        x_negated = negate_second_half(x) + first_half(x)
        cos(i,k_theta) = cache_cos[i, k] (k = [0,d/2-1])
        sin(i, k, theta) = cache_sin[i, k] (k = [0,d/2-1])
        
        First half [0, 1, ..., k/2-1] -> cos(i_k_theta)     
        '''

    '''
    Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. 
    Note that you should tolerate x with an arbitrary number of batch dimensions. 
    You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension.
    You should use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension.
    '''
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor): 
        # token positions: the true token positions - or m, used to multiplied by theta to get the angle. if we don't pass, the model will treat the token pos as 0, 1, .., seq-1. This might not be desirable as token positions might not be like that during inference and when packed multiple sequences.
        # calculate the 
        pass 