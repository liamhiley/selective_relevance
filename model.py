import torch
import numpy as np
import cv2
from models import c3d
from functools import partial

model_dict = {
        "c3d": partial(c3d.C3DasVGG, train=False)
}

def get_model(architecture, num_classes,weights_path="", **kwargs):
    model = model_dict[architecture](num_classes, **kwargs)
    if len(weights_path) > 0:
        nweights = {}
        weights = dict(torch.load(weights_path))
        if "state_dict" in weights:
            weights = dict(weights["state_dict"])
        for k, v in weights.items():
            if "module" in k:
                nweights[k[7:]] = v
            else:
                nweights[k] = v
        model.load_state_dict(nweights,strict=False)
    return model

def get_input(path, shape=None, mean=[0,0,0], std=[1,1,1],sample_len=16):
    """
    Read in video from file and store in a torch.Tensor.

    Args:
        path (str): the path to the video file.
        shape (tuple/list,optional): the desired shape to resize each frame to.
        mean (tuple/list,optional): channel-wise mean to center the channels of each frame around, in RGB format.
        std (tuple/list,optional): channel-wise standard deviation to scale the channels of each frame by, in RGB format.
        sample_len (int,optional): the length in frames of each sample expected by the model, the video will be split
            into batches accordingly.
    """
    rdr = cv2.VideoCapture(path)
    offsets = int(rdr.get(cv2.CAP_PROP_FRAME_COUNT)//sample_len)
    if shape is None:
        shape = (rdr.get(cv2.CAP_PROP_FRAME_HEIGHT),rdr.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        shape = tuple(shape)
    vid = torch.zeros((offsets,3,sample_len) + shape).requires_grad_()
    batch = []
    for o in range(offsets):
        frames = []
        f_idx = 0
        while True:
            if f_idx > sample_len-1:
                break
            r, frame = rdr.read()
            frames.append(frame)
            if not r:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(shape[1],shape[0]))
            frame = frame.transpose(2,0,1)
            frame = torch.from_numpy(frame).float()
            # if len(frames):
            #     print((frame==frames[-1]).all())
            for c in range(3):
                frame[c,...] -= mean[c]
                frame[c,...] /= std[c]
            vid[o][:,f_idx,:,:] = frame
            f_idx += 1
        if f_idx < sample_len:
            vid[o,:,f_idx:sample_len,:,:] = vid[o,:,f_idx:f_idx+1,:,:]
        batch.append(frames)
    return vid,batch

def get_exp(inp,mdl,target=-1):
    """
    For a given model, forward pass an input, and backwards pass the output to gain the explanation for that
        input.

    Args:
        inp (torch.Tensor): the input batch of samples.
        mdl (torch.nn.Module): the torchexplain model.
        target (int): the index of the target class to be explained.
    """
    print("Forward pass...")
    out = mdl(inp)
    if target < 0:
        prob,target = out.topk(1,1)
    filter_out=torch.zeros_like(out)
    filter_out[:,target] = 1
    print("Backward pass...")
    exp = torch.autograd.grad(out,inp, grad_outputs=filter_out,retain_graph=True)[0]
    exp_vis = exp.sum(dim=(1))
    for e in range(exp_vis.shape[0]):
        exp_vis[e,...] /= abs(exp_vis[e,...]).max()
    # exp_vis /= abs(exp_vis).max()
    if target < 0:
        return exp_vis, target
    return exp_vis

def show_image(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
