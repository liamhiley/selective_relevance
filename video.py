import os
import torch
import numpy as np
import cv2
import imageio

def get_input(path, shape=None, mean=[0,0,0], std=[1,1,1],sample_len=16):
    """
    Read in video from file and store in a torch.Tensor.

    Args:
        path (str): the path to the video file.
        shape (tuple/list of int,optional): the desired shape to resize each frame to.
        mean (tuple/list of int,optional): channel-wise mean to center the channels of each frame around, in RGB format.
        std (tuple/list of int,optional): channel-wise standard deviation to scale the channels of each frame by, in RGB format.
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

def get_class_dict_from_index_file(class_file_path):
    class_dict = {}
    with open(class_file_path,"r") as f:
        class_file = f.readlines()
    for line in class_file:
        ind, action = line.split()
        class_dict[ind - 1] = action
    return class_dict


def write_video(path,frames,**kargs):
    """
    Write to an .mp4 file from a list of frames

    Args:
        path (str): the path to the video file.
        frames (list of np.ndarrays): list of frames to write to the video
    """
    with imageio.get_writer(path, mode='I', **kargs) as writer:
        for frame in frames:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            writer.append_data(frame)
