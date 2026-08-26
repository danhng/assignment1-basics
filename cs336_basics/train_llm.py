import datetime
import math
import torch
import tqdm
from .tokenizer import FastTokenizer
from .transformer_lm import Transformer_LM
from .adam_w import AdamW
from types import SimpleNamespace
from .checkpoint import SimpleNamespace
from .dataloader import get_batch
from .utils import cross_entropy
from .utils import cosine_annealing_lr

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('train_llm')

STATE_DICT_MODEL_STATE = 'model'
STATE_DICT_OPTIM_STATE = 'optim'
STATE_DICT_TRAINING_STATE = 'training_state'
STATE_DICT_TRAINING_CONFIG = 'training_config'

"""
Input: 
vocab_filepath, merges_filepath, special_tokens=None, inputFormatJson=True
vocab_size, context_length, num_layers, d_model, num_heads, d_ff, dtype=None, device=None, weights = None, use_rope = False, theta = 10000
- 
"""
"""
self, tokenizer_vocab_file, tokenizer_merges_file, tokenizer_special_tokens, tokenizer_vocab_size,
                model_context_length, model_num_transformer_blocks, model_num_mha_heads, model_d_model, model_d_ff, model_dtype_weight, model_device, model_rope_use_rope=True, model_rope_theta=10000,
                optim_lr_max=1e-3, optim_weight_decay=0.01, optim_betas=(0.9, 0.999), optim_eps=1e-8, optim_lr_min=1e-8, optim_iter_warmup_end_ratio=0.05, optim_lr_cosine_end_ratio=0.8, optim_gradient_clipping_on=False, optim_gradient_clipping_max_l2_norm=1,
                training_batch_size=10, training_checkpoint_every_x_iter=5, training_epoch=1, training_max_iteration=10, training_use_chincilla_law=False, training_chincilla_token_param_ratio=20,
                **kwargs
"""
class X75_Trainer(): 

    def save_checkpoint(self, out): 
        state = {
            STATE_DICT_MODEL_STATE:self.model.state_dict(), 
            STATE_DICT_OPTIM_STATE:self.optimizer.state_dict(), 
            STATE_DICT_TRAINING_STATE:self.training_state, 
            STATE_DICT_TRAINING_CONFIG:self.training_config, 
        }
        torch.save(obj=state, f=out)
        return state

    
    def __init__(self, **kwargs): 
        #  PARAMS
        self.training_config = SimpleNamespace()
        self.training_state = SimpleNamespace() # init with empty state
        
        self.training_config.tokenizer_vocab_file = kwargs.get('tokenizer_vocab_file')
        self.training_config.tokenizer_merges_file = kwargs.get('tokenizer_merges_file')
        self.training_config.tokenizer_special_tokens = kwargs.get('tokenizer_special_tokens')
        self.training_config.tokenizer_vocab_size = kwargs.get('tokenizer_vocab_size')
        
        #MODEL HYPER PARAMS
        self.training_config.model_context_length = kwargs.get('model_context_length')
        self.training_config.model_num_transformer_blocks = kwargs.get('model_num_transformer_blocks')
        self.training_config.model_num_mha_heads = kwargs.get('model_num_mha_heads')
        self.training_config.model_d_ff = kwargs.get('model_d_ff')
        self.training_config.model_d_model = kwargs.get('model_d_model')
        self.training_config.model_dtype_weight = kwargs.get('model_dtype_weight')
        self.training_config.model_device = kwargs.get('model_device')
        self.training_config.model_rope_use_rope = kwargs.get('model_rope_use_rope', True)
        self.training_config.model_rope_theta = kwargs.get('model_rope_theta')
        
        self.training_config.optim_lr_max = kwargs.get('optim_lr_max')
        self.training_config.optim_weight_decay = kwargs.get('optim_weight_decay')
        self.training_config.optim_betas = kwargs.get('optim_betas')
        self.training_config.optim_eps = kwargs.get('optim_eps')
        self.training_config.optim_lr_min = kwargs.get('optim_lr_min')
        self.training_config.optim_lr_cosine_end_ratio = kwargs.get('optim_lr_cosine_end_ratio')
        self.training_config.optim_gradient_clipping_on = kwargs.get('optim_gradient_clipping_on')
        self.training_config.optim_iter_warmup_end_ratio = kwargs.get('optim_iter_warmup_end_ratio')
        self.training_config.optim_gradient_clipping_max_l2_norm = kwargs.get('optim_gradient_clipping_max_l2_norm')
        
        self.training_config.training_batch_size = kwargs.get('training_batch_size')
        self.training_config.training_checkpoint_every_x_iter = kwargs.get('training_checkpoint_every_x_iter')
        self.training_config.training_max_iterations = kwargs.get('training_max_iterations')
        self.training_config.training_use_chincilla_law = kwargs.get('training_use_chincilla_law')
        self.training_config.training_chincilla_token_param_ratio = kwargs.get('training_chincilla_token_param_ratio')
        self.training_config.training_checkpoint_path = kwargs.get('training_checkpoint_path')
        
        # init states
        self.training_state.current_iteration = 1
        self.training_state.current_tokens_processed = 0 # for chincilla law
        self.training_state.current_validation_loss = math.inf # for chincilla law
        
        # model, optim, tokenizer
        self.tokenizer = FastTokenizer.from_files(vocab_filepath=self.training_config.tokenizer_vocab_file, merges_filepath=self.training_config.tokenizer_merges_file, special_tokens=self.training_config.tokenizer_special_tokens)
        self.model = Transformer_LM(vocab_size=self.training_config.tokenizer_vocab_size, context_length=self.training_config.model_context_length, num_layers=self.training_config.model_num_transformer_blocks,
                            num_heads=self.training_config.model_num_mha_heads, d_ff=self.training_config.model_d_ff, dtype=self.training_config.model_dtype_weight,
                            device=self.training_config.model_device, weights=None, use_rope=self.training_config.model_rope_use_rope, theta=self.training_config.model_rope_theta, d_model=self.training_config.model_d_model)
        self.optimizer = AdamW(self.model.parameters(), lr=self.training_config.optim_lr_max,
                        weight_decay=self.training_config.optim_weight_decay, betas=self.training_config.optim_betas, eps=self.training_config.optim_betas)
        
    
    """
    Load a checkpoint from src (path or file-likeobject), and then recover the model and optimizer states from that checkpoint. 
    Your function should return the iteration number that was saved to the checkpoint. You can use torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
    load_state_dict method in both the model and optimizer to return them to their previous states
    """
    @classmethod
    def load_model_from_file(src): 
        # Load the checkpoint dictionary
        checkpoint = torch.load(src)
        trainer = X75_Trainer(checkpoint[STATE_DICT_TRAINING_CONFIG])
        # Restore the model and optimizer states
        trainer.model.load_state_dict(checkpoint[STATE_DICT_MODEL_STATE])
        trainer.optimizer.load_state_dict(checkpoint[STATE_DICT_OPTIM_STATE])    
        trainer.training_state = checkpoint[STATE_DICT_TRAINING_STATE]
        return trainer
    
    def _get_checkpoint_name(self): 
        now = datetime.now()
        string_format = now.strftime("%y%m%d%H%M%S")
        return f"{hash(str(self))}-{string_format}-{self.self.training_state.current_iteration}.pt"        

    """
    Train the model
    Params: 
        - training_tokens: the memmap training tokens numpy array
        
    """
    def train_llm(self, training_tokens):
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

        # Step 1: Construct the tokenizer
        # training loop
        current_iteration = self.training_state.current_iteration
        validation_loss = self.training_state.current_validation_loss
        total_params = sum(p.numel() for p in self.model.parameters())
        tokens_trained = self.training_state.current_tokens_processed
        
        # todo: validate the training tokens passed is the one used in the checkpoint. 

        """
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=1000)
        for t in range(100):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.
        """
        for current_iteration in tqdm(range(current_iteration, self.training_config.training_max_iterations + 1), desc="Training"):
            inputs, targets = get_batch(training_tokens, self.training_config.training_batch_size, self.training_config.model_context_length, self.training_config.model_device) # move input and target to device
            logger.info(f"Step {current_iteration}, loaded inputs, targets of size: {inputs.shape}")
            
            self.model.zero_grad() # reset gradients of all parameters before calculating
            logits = self.model(inputs) # model forward pass
            validation_loss = cross_entropy(logits=logits, target=targets) # loss calculation
            validation_loss.backward() # model backward pass
            self.optimizer.step() # optimize the weights
            
            # update state after each iteration
            self.training_state.current_iteration = current_iteration + 1
            self.training_state.validation_loss = validation_loss
            self.training_state.tokens_trained = tokens_trained + inputs.numel()
            logger.info(f"After step {current_iteration}, tokens trained: {self.training_state.tokens_trained}, lr: {self.optimizer.get_lr}, loss: {validation_loss}")

            #checkpointing
            if (current_iteration % self.training_config.training_checkpoint_every_x_iter == 0): 
                path = self._get_checkpoint_name()
                self.save_checkpoint(path)
                logger.info(f"saved to save checkpoint: {path}")
                

## Usage
if __name__ == '__main__':
    # translate the training text to binary token file
    # tokenizer = FastTokenizer.from_files(vocab_filepath=)
    training_config = dict()
    training_config[""]