import os
import torch

def save_weights(path, model):
    """
    Save model weights
    """
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    # Write first to the temporary file
    temp_path = path + '.tmp'
    dir_path = os.path.dirname(temp_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    torch.save(state_dict, temp_path)
    # Safely rename to the main file
    os.rename(temp_path, path)