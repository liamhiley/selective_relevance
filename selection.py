import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SelectiveRelevance:
    """
    Object for applying the Selective Relevance method to a video explanation. Basically a wrapper for a torch Sobel operator based masking algorithm.

    Attributes:
        sobel (torch.Tensor): The Sobel kernel applied to the explanation to obtain the relevance gradient.
        sig (int): The number of standard deviations to threshold the relevance gradient at. Positions for where the gradient is larger than
            sig*rel_grad.std() are considered to be relevant due to motion.
    """
    def __init__(self, sig):
        """
        Args:
            sig (int): See Attributes
        """
        self.sig = sig
        self.sobel = torch.tensor([[[1,2,1],[2,4,2],[1,2,1]],[[0,0,0],[0,0,0],[0,0,0]],[[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]]])
        self.sobel = self.sobel.reshape((1,1,3,3,3))

    def get_exp_from_file(self,path):
        """
        Read explanation from video file into torch.Tensor
        Args:
            path (str): Absolute/relative path to explanation video file
        """
        # Assume path is to video file
        rdr = cv2.VideoCapture(path)
        frames = []
        while True:
            r, frame = rdr.read()
            if frame is None:
                break
            # convert to grey-scale map
            if len(frame.shape) > 2:
                frame = frame.sum(axis=2)
            frame = torch.from_numpy(frame.astype(np.float))
            frames.append(frame)
        expl_vid = torch.stack(frames,0)
        return expl_vid

    def selective_relevance(self,expl_tensor,cmap=None,input=None):
        """
        Apply mask to explanation based on it's gradient over time, i.e. how quickly relevance changes in a region.

        Args:
            expl_tensor (torch.Tensor): Grey-scale magnitude representation of relevance for an input video. Each pixels value should be the
                amount of relevance at that position. Overlaying the explanation on the input, or putting it through some colour map will cause
                incorrect results.
            cmap (str, optional): Name of matplotlib.pyplot colourmap to apply to the resulting Selective Relevance map.
            input ((list of str)/str,optional): The input video on which to overlay the resulting Selective Relevance map.
        Returns:
            sel_expl (list of numpy.ndarrays): List of frames for the Selective Relevance map.
        """
        if cmap:
            cmap = plt.get_cmap(cmap)

        if expl_tensor.min() < 0:
            expl_tensor = (expl_tensor - expl_tensor.min())/(expl_tensor.max() - expl_tensor.min())


        # sobel operator expects a batch and channel dimension, it also requires padding to fit to
        #     even dimensions but this can be altered.
        deriv_t = F.conv3d(expl_tensor[None][None].float(), self.sobel.float(), padding=(1, 1, 1))[0, 0, ...]
        # this is the selective process in essentially one line: constructing the mask and applying it
        temp_vis = expl_tensor * (deriv_t > (deriv_t.std() * self.sig)).float()
        temp_vis = temp_vis.numpy()
        sel_expl = []
        for f_idx in range(temp_vis.shape[0]):
            sel_frame = temp_vis[f_idx]
            if cmap:
                sel_frame = cmap(sel_frame)
            sel_frame = (sel_frame*255).astype(np.uint8)
            sel_frame = cv2.cvtColor(sel_frame,cv2.COLOR_BGRA2RGB)
            if input:
                in_frame = input[f_idx][...,0:3]
                sel_frame = cv2.resize(sel_frame,(in_frame.shape[1],in_frame.shape[0]))
                sel_frame = cv2.addWeighted(sel_frame,0.5,in_frame,0.5,0)
            sel_expl.append(sel_frame)
        return sel_expl
