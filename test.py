import model
import video
import selection
import torch
import numpy as np
import cv2

mdl = model.get_model("c3d",101,"models/weights/c3d_ucf101.pth",range=(-101.41,164.75)).cuda()
inp, frames = video.get_input("/media/datasets/Video/UCF_CRIME/Katie/RoadAccidents036_x264.mp4",shape=(112,112),mean=[90.25,96.77,101.41])
exp = []
for i in range(0,inp.shape[0],8):
    samp = inp[i:i+8,...]
    exp_samp = model.get_exp(samp.cuda(),mdl)
    exp.append(exp_samp)
exp = torch.cat(exp,0).cpu()
selector = selection.SelectiveRelevance(3)
save_path = "/media/datasets/Video/UCF_CRIME/Katie/Fighting006/sel_dtd_{}.png"
# save_path = "/media/datasets/Video/UCF_CRIME/Katie/RoadAccidents036/sel_dtd_{}.png"
sel_exp = []
for i,e in enumerate(exp):
    sel_exp.append(selector.selective_relevance(e,"gist_heat",frames[i]))

sel_exp = [i for s in sel_exp for i in s]
video.write_video("/media/datasets/Video/UCF_CRIME/Katie/RoadAccidents036/sel_dtd.mp4",sel_exp)
for i,se in enumerate(sel_exp):
    cv2.imwrite(save_path.format(i),se)
