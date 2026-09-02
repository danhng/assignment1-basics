from datetime import datetime
import math
import sys
import mlflow
import numpy as np
import torch
from tqdm import tqdm
from cs336_basics import utils
from cs336_basics.tokenizer import FastTokenizer
from cs336_basics.transformer_lm import Transformer_LM
from cs336_basics.adam_w import AdamW
from types import SimpleNamespace
from cs336_basics.dataloader import get_batch
from cs336_basics.utils import cross_entropy
from cs336_basics.cosine_annealing_lr import CosineAnnealingLR
import tomllib
import matplotlib.pyplot as plt

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data/log/{datetime.now().strftime("%y%m%d%H%M%S")}.log', mode='a'),  # Log to file
        logging.StreamHandler(sys.stdout)          # Log to console
    ]
)
logger = logging.getLogger('train_llm')

STATE_DICT_MODEL_STATE = 'model'
STATE_DICT_OPTIM_STATE = 'optim'
STATE_DICT_TRAINING_STATE = 'training_state'
STATE_DICT_TRAINING_CONFIG = 'training_config'

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("X75_Trainer")

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
            STATE_DICT_TRAINING_STATE:vars(self.training_state), 
            STATE_DICT_TRAINING_CONFIG:vars(self.training_config), 
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
        # self.training_config.tokenizer_vocab_size = kwargs.get('tokenizer_vocab_size')
        
        #MODEL HYPER PARAMS
        self.training_config.model_context_length = kwargs.get('model_context_length')
        self.training_config.model_num_transformer_blocks = kwargs.get('model_num_transformer_blocks')
        self.training_config.model_num_mha_heads = kwargs.get('model_num_mha_heads')
        self.training_config.model_d_ff = kwargs.get('model_d_ff')
        self.training_config.model_d_model = kwargs.get('model_d_model')
        dtype_val = kwargs.get('model_dtype_weight', 'float32')
        # If it's a string (fresh start), look it up in torch
        if isinstance(dtype_val, str):
            self.training_config.model_dtype_weight = getattr(torch, dtype_val)
        # If it's already a dtype (loaded from checkpoint), assign it directly
        else:
            self.training_config.model_dtype_weight = dtype_val
        # self.training_config.model_dtype_weight = getattr(torch, kwargs.get('model_dtype_weight','float32'))
        self.training_config.model_device = torch.device(kwargs.get('model_device'))
        self.training_config.model_rope_use_rope = kwargs.get('model_rope_use_rope', True)
        self.training_config.model_rope_theta = kwargs.get('model_rope_theta', 10000)
        
        self.training_config.optim_lr_max = kwargs.get('optim_lr_max', 1e-3)
        self.training_config.optim_weight_decay = kwargs.get('optim_weight_decay', 1e-1)
        self.training_config.optim_betas = tuple(kwargs.get('optim_betas', (0.9, 0.999)))
        self.training_config.optim_eps = kwargs.get('optim_eps', 1e-8)
        self.training_config.optim_lr_min = kwargs.get('optim_lr_min',0)
        self.training_config.optim_lr_cosine_ratio = kwargs.get('optim_lr_cosine_ratio', 1)
        self.training_config.optim_gradient_clipping_enable = kwargs.get('optim_gradient_clipping_enable', True)
        self.training_config.optim_lr_warmup_ratio = kwargs.get('optim_lr_warmup_ratio', 0.05)
        self.training_config.optim_gradient_clipping_max_l2_norm = kwargs.get('optim_gradient_clipping_max_l2_norm', 1)
        
        self.training_config.training_batch_size = kwargs.get('training_batch_size',1)
        self.training_config.training_checkpoint_every_x_iter = kwargs.get('training_checkpoint_every_x_iter',1000)
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
        self.model = Transformer_LM(vocab_size=self.tokenizer.get_vocab_size(), context_length=self.training_config.model_context_length, num_layers=self.training_config.model_num_transformer_blocks,
                            num_heads=self.training_config.model_num_mha_heads, d_ff=self.training_config.model_d_ff, dtype=self.training_config.model_dtype_weight,
                            device=self.training_config.model_device, weights=None, use_rope=self.training_config.model_rope_use_rope, theta=self.training_config.model_rope_theta, d_model=self.training_config.model_d_model)
        self.optimizer = AdamW(self.model.parameters(), lr=self.training_config.optim_lr_max,
                        weight_decay=self.training_config.optim_weight_decay, betas=self.training_config.optim_betas, eps=self.training_config.optim_eps)
        # todo scheduler
        self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer, max_iters=self.training_config.training_max_iterations, 
                                              warmup_iter_ratio=self.training_config.optim_lr_warmup_ratio, 
                                            cosine_iter_ratio=self.training_config.optim_lr_cosine_ratio, 
                                            lr_min=self.training_config.optim_lr_min)
        
    """
    Load a checkpoint from src (path or file-likeobject), and then recover the model and optimizer states from that checkpoint. 
    Your function should return the iteration number that was saved to the checkpoint. You can use torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
    load_state_dict method in both the model and optimizer to return them to their previous states
    """
    @classmethod
    def load_model_from_file(cls, src): 
        # Load the checkpoint dictionary
        checkpoint = torch.load(src, weights_only=False)
        trainer = X75_Trainer(**checkpoint[STATE_DICT_TRAINING_CONFIG])
        # Restore the model and optimizer states
        trainer.model.load_state_dict(checkpoint[STATE_DICT_MODEL_STATE])
        trainer.optimizer.load_state_dict(checkpoint[STATE_DICT_OPTIM_STATE])    
        trainer.training_state = checkpoint[STATE_DICT_TRAINING_STATE]
        return trainer
    
    def _get_checkpoint_name(self, time, iteration, name): 
        return self.training_config.training_checkpoint_path + f"{name}-{hash(str(self))}-{time}-{iteration}.pt"        

    """
    Train the model
    Params: 
        - training_tokens: the memmap training tokens numpy array
        
    """
    def train_llm(self, training_tokens):
        mlflow.enable_system_metrics_logging()
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
        now = datetime.now()
        time = now.strftime("%y%m%d%H%M%S")
        
        # Step 1: Construct the tokenizer
        # training loop
        current_iteration = self.training_state.current_iteration
        validation_loss = self.training_state.current_validation_loss
        total_params = sum(p.numel() for p in self.model.parameters())
        model_name=f"X75-{int(total_params/1e6)}M"
        logger.info(f"Model params: {total_params}")
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
        with mlflow.start_run():
            mlflow.log_params(self.training_config.__dict__)
            for current_iteration in tqdm(range(current_iteration, self.training_config.training_max_iterations + 1), desc="Training"):
                inputs, targets = get_batch(training_tokens, self.training_config.training_batch_size, self.training_config.model_context_length, self.training_config.model_device) # move input and target to device
                logger.info(f"Step {current_iteration}, loaded inputs, targets of size: {inputs.shape}")
                self.model.zero_grad() # reset gradients of all parameters before calculating
                logits = self.model(inputs) # model forward pass
                # logger.info(f"Forward pass result size  {logits.shape}")
                validation_loss = cross_entropy(logits=logits, target=targets) # loss calculation
                validation_loss.backward() # model backward pass
                self.optimizer.step() # optimize the weights
                self.lr_scheduler.step() # step the scheduler to update the lr 
                
                # gemini added 
                current_lr = self.optimizer.get_lr()
                if isinstance(current_lr, list):
                    current_lr = current_lr[0]

                # update state after each iteration
                self.training_state.current_iteration = current_iteration + 1
                self.training_state.validation_loss = validation_loss
                self.training_state.current_tokens_processed = self.training_state.current_tokens_processed + inputs.numel()
                logger.info(f"After step {current_iteration}, tokens trained: {self.training_state.current_tokens_processed}, lr: {self.optimizer.get_lr()}, loss: {validation_loss}")

                #checkpointing
                if (current_iteration % self.training_config.training_checkpoint_every_x_iter == 0): 
                    path = self._get_checkpoint_name(time=time, iteration=current_iteration, name=model_name)
                    self.save_checkpoint(path)
                    logger.info(f"MEMORY FOOTPRINT AT STEP: {current_iteration}")
                    if (self.training_config.model_device.type == 'cuda'):
                        logger.info(torch.cuda.memory_summary())
                    logger.info(f"saved to save checkpoint: {path}")
                
                mlflow.log_metric("loss", validation_loss, step=current_iteration)
                mlflow.log_metric("learning rate", current_lr, step=current_iteration)
            mlflow.pytorch.log_model(self.model, name=f"X75 {int(total_params/1e6)}M", serialization_format="pickle")
                
    """
        Generate text
    """
    def generate(self, input, max_tokens_generated, temperature, p_sampling_threshold): 
        # Step 0. turn input text to tensor
        input_tensor = torch.tensor([self.tokenizer.encode(str) for str in input], dtype=torch.int32, device=self.training_config.model_device)
        # Step 1. Set model in eval mode
        # Step 2. Iterate through max_tokens_generated
        # Step 2.1 run the forward pass
        # Step 2.2. reduce to only the last token' probs
        # Step 2.2 calculate the last output token's softmax (with temperature scaled logits)
        # Step 2.3. sample next token from the top p candidate output tokens
        # Step 2.4. append token ids to output, and input (making new input)
        self.model.eval()
        # # 1. Get all dimensions except the last one using slicing [:-1]
        # base_shape = input.size()[:-1]
        # # 2. Unpack the base shape and add the new last dimension
        # new_shape = (*base_shape, max_tokens_generated)
        # # 3. Initialize the new tensor (e.g., with zeros)
        # # It is highly recommended to match the dtype and device of your input
        # output = torch.empty(new_shape, dtype=input.dtype, device=input.device)
        output = []
        input_appended = input_tensor
        for i in range(max_tokens_generated):
            logger.debug(f"Input appended: {input_appended}")
            logits = self.model(input_appended) # output of size seq_len, vocab_size
            last_token_logits = logits[:, -1:, :] # size batch, 1, vocab_size
            # todo - high: implement top k. 
            last_token_logits = self.top_k(last_token_logits, 50)
            logits = logits / temperature # scale the logits
            softmaxes_last_token = utils.softmax(last_token_logits, -1) # batch, 1, vocab_size
            #softmaxes_last_token_top_k = 
            output_token_id = self.sample_top_p(softmaxes_last_token, p_sampling_threshold) # size batch, 1,1
            output_int = output_token_id.squeeze().item()
            logging.debug(f"iteration {i} output: {output_token_id.item()} -> {self.tokenizer.vocab_id_word[output_int]}")
            # append token ids to output and input appended
            output.append(output_int)
            if (output_int in self.tokenizer.get_special_tokens_ids()): 
                break
            else:
                input_appended = torch.cat((input_appended, output_token_id), dim=-1)
        return output
        
    
    def sample_top_k (logits, k): 
        return logits
    """
    Input: 1, vocab_size
    Output: token id
    """
    def sample_top_p(self, softmaxes, p=0.9): 
        if softmaxes.dim() == 3:
            softmaxes = softmaxes.squeeze(1)
        # Step 1. get sampling_tokens: the minimum numbers of tokens that have cummulative probs > p
        sorted_probs, sorted_ids = torch.sort(softmaxes, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        ids_remove = cum_probs > p
        ids_remove[..., 1:] = ids_remove[..., :-1].clone() # shift by 1 to exclude the sample that pushes the cum over p from ids to remove
        ids_remove[..., 0] = 0 # make sure the first sample is never to be removed
        sorted_probs[ids_remove] = 0.0
        # normalize probs (make sure all probs add up to 1 again) so we could use the multinomial method later 
        sorted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=False)
        # Step 2. sample 1 sample from sampling_tokens
        sorted_sampled_token = torch.multinomial(sorted_probs, num_samples=1) # batch, 1
        # Step 3. Gather chosen index (indices) along the last dim
        token_ids = torch.gather(sorted_ids, -1, sorted_sampled_token)
        return token_ids

def doTrain(input_data_set_path, config_training_path): 
    mlflow.enable_system_metrics_logging()
    # translate the training text to binary token file
    # tokenizer = FastTokenizer.from_files(vocab_filepath=)
    # Open the file in binary mode ("rb")
    with open(config_training_path, "rb") as file:
        training_config = tomllib.load(file)
    logger.info(f"===Training config==")
    logger.info(training_config)
    trainer = X75_Trainer(**training_config)
    #load numpy array in memmap mode
    inputs = np.load(input_data_set_path, mmap_mode='r')
    logger.info(f"input token length {inputs.size}")
    trainer.train_llm(inputs)

## Training script
if __name__ == '__main__':
    #"data/output/tinystories_sample_5M.txt-encoded-darwin.npy"
    # trainer = X75_Trainer.load_model_from_file("data/checkpoint/X75-14M-6388329099797006412-260831203714-200.pt")
    # input = ["Jenny was a very proud human"]
    # response = trainer.generate(input=input, max_tokens_generated=20, temperature=1, p_sampling_threshold=0.9)
    # logging.info(f"Generated response: {trainer.tokenizer.decode(response)}")
    doTrain(input_data_set_path="data/output/TinyStoriesV2-GPT4-train.txt-encoded-linux.npy", config_training_path="config/training_config.toml")
    