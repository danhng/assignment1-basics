import torch
from .embedding import TokenEmbedding
from .transformer_block import TransformerBlock
from .transformer_block import RMSNorm
from .linear import Linear
from .utils import softmax

'''
Token Embedding 
Transformer Block
Norm 
FFN
Softmax
Output Probabilities
'''
class Transformer_LM(torch.nn.Module): 
    """
    vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
    context_length: int The maximum context length, necessary for determining the dimensionality of the RoPE sin and cos buffer.
    num_layers: int The number of Transformer blocks to use.
    
    vocab size will affect embedding
    """
    def __init__(self,vocab_size, context_length, num_layers, d_model, num_heads, d_ff, dtype=None, device=None, weights = None, use_rope = False, theta = 10000): 
        super().__init__()
        # input: vocab_size, d_model, 
        # output: seq_len, d_model
        self.token_embeddings = TokenEmbedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype) 
        
        # input: seq_len, d_model
        # output: seq_len, d_model
        layers = [TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dtype=dtype, device=device, weights=None, use_rope=use_rope, max_seq_len=context_length, theta=theta) for _ in range(num_layers)]
        self.layers = torch.nn.ModuleList(layers)
        self.num_layers = num_layers
        
        # input: seq_len, d_model
        # output: seq_len, d_model
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        
        # input: seq_len, d_model
        # weight: vocab_size, d_model
        # output: seq_len, vocab_size (output probablity)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
        self.load_state_dict(weights)
        self._verify_weights(weights)
        
        
    def _verify_weights(self, weights):
        """Helper method to dynamically verify weights without hardcoding layers."""
        print("assert begins")
        assert torch.equal(self.token_embeddings.weight.detach(), weights["token_embeddings.weight"])
        
        for i in range(self.num_layers):
            assert torch.equal(self.layers[i].attn.q_proj.weight.detach(), weights[f"layers.{i}.attn.q_proj.weight"])
            assert torch.equal(self.layers[i].attn.k_proj.weight.detach(), weights[f"layers.{i}.attn.k_proj.weight"])
            assert torch.equal(self.layers[i].attn.v_proj.weight.detach(), weights[f"layers.{i}.attn.v_proj.weight"])
            assert torch.equal(self.layers[i].attn.output_proj.weight.detach(), weights[f"layers.{i}.attn.output_proj.weight"])
            
            assert torch.equal(self.layers[i].ln1.weight.detach(), weights[f"layers.{i}.ln1.weight"])
            assert torch.equal(self.layers[i].ln2.weight.detach(), weights[f"layers.{i}.ln2.weight"])
            
            assert torch.equal(self.layers[i].ffn.w1.weight.detach(), weights[f"layers.{i}.ffn.w1.weight"])
            assert torch.equal(self.layers[i].ffn.w2.weight.detach(), weights[f"layers.{i}.ffn.w2.weight"])
            assert torch.equal(self.layers[i].ffn.w3.weight.detach(), weights[f"layers.{i}.ffn.w3.weight"])
            
        assert torch.equal(self.ln_final.weight.detach(), weights["ln_final.weight"])
        assert torch.equal(self.lm_head.weight.detach(), weights["lm_head.weight"])
        print("assert ends")
    
    def forward(self, in_indices): 
        print("Model architecture:", self)
        print("Input size:", in_indices.size())
        output_token_embeddings = self.token_embeddings(in_indices)
        output_transformer_block = output_token_embeddings
        for layer in self.layers: 
            output_transformer_block = layer(output_transformer_block)
        output_ln_final = self.ln_final(output_transformer_block)
        output_lm_head = self.lm_head(output_ln_final) 
        # output_softmax = softmax(output_lm_head, -1)
        return output_lm_head