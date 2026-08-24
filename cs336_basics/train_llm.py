import torch 
from .tokenizer import FastTokenizer
from .transformer_lm import Transformer_LM
from .adam_w import AdamW

"""
Input: 
vocab_filepath, merges_filepath, special_tokens=None, inputFormatJson=True
vocab_size, context_length, num_layers, d_model, num_heads, d_ff, dtype=None, device=None, weights = None, use_rope = False, theta = 10000
- 
"""
def train_llm(tokenizer_vocab_file, tokenizer_merges_file, tokenizer_special_tokens, tokenizer_vocab_size, 
              model_context_length, model_num_transformer_blocks, model_num_mha_heads, model_d_model, model_d_ff, model_dtype_weight, model_device, model_rope_use_rope=True, model_rope_theta=10000, 
              optim_lr_max=1e-3, optim_weight_decay=0.01, optim_betas=(0.9, 0.999), optim_eps=1e-8, optim_lr_min=1e-8, optim_iter_warmup_end_ratio=0.05, optim_lr_cosine_end_ratio=0.8, optim_gradient_clipping_on = False, optim_gradient_clipping_max_l2_norm = 1,
              training_batch_size=10, training_checkpoint_every_x_batch=5, training_epoch=1, 
              **kwargs): 
    
    """
    1. construct the tokenizer 
    2. construct the model 
    3. construct the optimizer
    4. Training loop 
        while iteration < max_iter or validation_loss > certain threshold
            Get the batches (batch_size, sequence)
            run through the model.forward to calculate predicted targets logits
            calculate the cross entropy loss between predicted targets logits and ground truth targets
            optimize
                clip gradients
                optimize weights using moment, rmsprop, weight decay
            update the lr using cosine annealing
    """
    
    #Step 1: Construct the tokenizer
    tokenizer = FastTokenizer.from_files(vocab_filepath=tokenizer_vocab_file, merges_filepath=tokenizer_merges_file, special_tokens=tokenizer_special_tokens)
    model = Transformer_LM(vocab_size=tokenizer_vocab_size, context_length=model_context_length, num_layers=model_num_transformer_blocks, 
                           num_heads=model_num_mha_heads, d_ff=model_d_ff, dtype=model_dtype_weight, 
                           device=model_device, weights=None, use_rope=model_rope_use_rope, theta=model_rope_theta, d_model=model_d_model)
    optimizer = AdamW(model.parameters(), lr=optim_lr_max,weight_decay=optim_weight_decay,betas=optim_betas,eps=optim_betas)
    
    
    
    pass
    