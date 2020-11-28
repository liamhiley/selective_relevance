import torch
import pickle
import numpy as np
import cv2

from .models import *

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
    "resneXt3d_50": partial(resnet3d.generate_model, model_depth=50,cardinality=32,in_planes=64,train=False),
    "resneXt3d_101": partial(resnet3d.generate_model,model_depth=101,cardinality=32,in_planes=64,train=False),
    "mars": partial(resnet3d.generate_model,model_depth=101,cardinality=32,in_planes=64,train=False),
    "resneXt3d_152": partial(resnet3d.generate_model,model_depth=152,cardinality=32,in_planes=64,train=False),
    "slowfast_4x16_R50": partial(slowfast.slowfast_4x16,train=False)
}

def get_model(arch, num_classes,weights_path="", **kwargs):
    if "slowfast" in arch["name"]:
        arch_name = f"{arch['name']}_{arch['frame_sampling']}_{arch['base_net']}"
    else:
        arch_name = arch['name']
    model = model_dict[arch_name](num_classes=num_classes, **kwargs)
    if weights_path:
        load_model(model,weights_path,**kwargs)
    return model

def load_model(model,weights_path,is_caffe=False, map_loc="cpu", is_parallel=False, **kwargs):
    if len(weights_path) > 0:
        if is_caffe:
            with open(weights_path,"rb") as f:
                weights = pickle.load(f,encoding="latin1")
                weights = weights["blobs"]
        else:
            weights = dict(torch.load(weights_path,map_location=map_loc))
        nweights = {}
        if "state_dict" in weights:
            weights = dict(weights["state_dict"])
        for k, v in weights.items():
            if "module" in k and not is_parallel:
                nweights[k[7:]] = v.clone().detach()
            else:
                nweights[k] = v.clone().detach()
        model.load_state_dict(nweights,strict=False)
    return model
