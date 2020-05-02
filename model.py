import torch
import pickle
import numpy as np
import cv2

from models import *

from functools import partial
from collections import OrderedDict

model_dict = {
    "c3d": partial(c3d.C3DasVGG, train=False),
    "resnet3d_10": partial(resnet3d.generate_model, model_depth=10,train=False),
    "resnet3d_18": partial(resnet3d.generate_model, model_depth=18,train=False),
    "resnet3d_34": partial(resnet3d.generate_model, model_depth=34,train=False),
    "resnet3d_50": partial(resnet3d.generate_model, model_depth=50,train=False),
    "resnet3d_101": partial(resnet3d.generate_model,model_depth=101,train=False),
    "resnet3d_152": partial(resnet3d.generate_model,model_depth=152,train=False),
    "resnet3d_200": partial(resnet3d.generate_model,model_depth=200,train=False),
    "slowfast_4x16": partial(slowfast.slowfast_4x16,train=False)
}

def get_model(architecture, num_classes,weights_path="", **kwargs):
    model = model_dict[architecture](num_classes=num_classes, **kwargs)
    if weights_path:
        load_model(model,weights_path,**kwargs)
    return model

def load_model(model,weights_path,is_caffe=False):
    is_parallel = hasattr(model,"module")
    if len(weights_path) > 0:
        if is_caffe:
            with open(weights_path,"rb") as f:
                weights = pickle.load(f,encoding="latin1")
                weights = weights["blobs"]
        else:
            weights = dict(torch.load(weights_path))
        nweights = {}
        if "state_dict" in weights:
            weights = dict(weights["state_dict"])
        for k, v in weights.items():
            if "module" in k and not is_parallel:
                nweights[k[7:]] = torch.tensor(v).clone()
            else:
                nweights[k] = torch.tensor(v).clone()
        model.load_state_dict(nweights,strict=False)
    return model
