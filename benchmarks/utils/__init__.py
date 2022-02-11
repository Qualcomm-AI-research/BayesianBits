# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


import torch


def load_model(model, model_path, location=None):
    state_dict = torch.load(model_path, map_location=location)

    # Error check one, if state_dict is a key
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    # Error check if model was saved as Data parallel
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict)
    return model
