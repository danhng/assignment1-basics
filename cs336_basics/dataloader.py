import torch
import numpy as np

"""
Deliverable: Write a function that takes a numpy array x (integer array with token IDs), a batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), 
and returns a pair of tensors: the sampled input sequences and the corresponding next-token targets. 
Both tensors should have shape (batch_size, context_length)
"""
def get_batch(x, batch_size, context_length, device): 
    # Step 1. get the correct starting indices of sample sequences (0, x.len-context.length-1). The " - 1" is due to we could only sample to the token prior to the last token so we could get the targets (which is advanced by one token)
    end_idx_sample = len(x) - context_length - 1
    start_indices = np.random.randint(0, end_idx_sample + 1, size=batch_size).reshape(-1, 1)
    offsets = np.arange(context_length).reshape(1, -1)
    
    # Step 2. get batch_size random indices from the range of step 1 
    full_batch_indices_sample = start_indices + offsets # broadcasting operations
    full_batch_indices_target = full_batch_indices_sample + 1 # advances all sample indices by 1
    
    # Step 3. Get and put the samples and target tensors to device
    inputs = torch.from_numpy(x[full_batch_indices_sample]).to(device=device, dtype=torch.long)
    targets = torch.from_numpy(x[full_batch_indices_target]).to(device=device, dtype=torch.long)
    return (inputs, targets)