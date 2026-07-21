import math
import torch

class Embedding(): 
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None, weights=None): 
        super().__init__()
        self.num_embeddings = num_embeddings # vocab size
        self.d_model = embedding_dim # d_model
        self.device = device # device to store embedding
        self.dtype = dtype # type of embedding params
        if weights is None: 
            embeddings = torch.empty(size = (self.num_embeddings, self.d_model), dtype=dtype) 
            torch.nn.init.trunc_normal_(embeddings, 0, std=1, a=-3, b=3)
            self.embedding_weights = torch.nn.Parameter(embeddings)
        else: 
            self.embedding_weights = weights
    
    def forward(self, token_ids: torch.Tensor): 
        return self.embedding_weights[token_ids]
        