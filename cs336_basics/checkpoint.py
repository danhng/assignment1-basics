# import torch 
# from .train_llm

# """
# Dump all the state from the model, optimizer and iteration into the file-like object out. 
# You can use the state_dict method of both the model and the optimizer to get their relevant states and use torch.save(obj, out) to dump obj into out (PyTorch supports either a path or a file-like
# object here). 
# A typical choice is to have obj be a dictionary, but you can use whatever format you want as long as you can load your checkpoint later
# """
# STATE_DICT_MODEL_STATE = 'model'
# STATE_DICT_OPTIM_STATE = 'optim'
# STATE_DICT_TRAINING_STATE = 'training_state'
# STATE_DICT_HYPER_PARAMS = 'hyper_params'

# def save_checkpoint(, out): 
#     state = {
#         STATE_DICT_MODEL_STATE:model.state_dict(), 
#         STATE_DICT_OPTIM_STATE:optimizer.state_dict(), 
#     }
#     torch.save(obj=state, f=out)
#     return state

# """
# Load a checkpoint from src (path or file-likeobject), and then recover the model and optimizer states from that checkpoint. 
# Your function should return the iteration number that was saved to the checkpoint. You can use torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
# load_state_dict method in both the model and optimizer to return them to their previous states
# """
# def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer): 
#     states = torch.load(src)
#     model.load_state_dict(state_dict=states[STATE_DICT_MODEL_STATE])
#     optimizer.load_state_dict(state_dict=states[STATE_DICT_OPTIM_STATE])
#     return states[STATE_DICT_ITERATION]