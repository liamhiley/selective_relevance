import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.ndimage import zoom

import pdb

def flatten(name, module, flattened_modules):
    if len(module._modules) > 0:
        for c, child in module._modules.items():
            flatten(name+"."+c, child, flattened_modules)
    else:
        flattened_modules.append((name,module))

def _lrp(samples,mdl,target,target_layers="fc",proportion=False):
    output = mdl(samples)
    if target < 0:
        avg = output.mean(dim=0)
        prob, target = torch.topk(avg, 1)
        target = target.float().mean().int()
        print(f"Explaining for class {target}")
    out_mask = torch.zeros_like(output)
    out_mask[:, target] += 1
    # Modified backward pass
    if isinstance(samples,(list,tuple)):
        grad = [torch.autograd.grad(output, sample, out_mask, retain_graph=True)[0] for sample in samples]
    else:
        grad = torch.autograd.grad(output, samples, out_mask, )[0]
    return grad, target

def _grad_cam(samples,mdl,target,target_layers='avg_pool'):
    features = []
    gradients = []
    def save_out(mod,in_,out_):
        features.append(out_)
    modules = []
    flatten("model",mdl, modules)
    for mod in modules:
        name, module = mod
        if name in target_layers:
            module.register_forward_hook(save_out)
    output = mdl(samples)
    if target < 0:
        avg = output.mean(dim=0)
        prob, target = torch.topk(avg, 1)
        target = target.float().mean().int()
        print(f"Explaining for class {target}")
    out_mask = torch.zeros_like(output)
    out_mask[:, target] = 1
    mdl.zero_grad()
    feature_grads = torch.autograd.grad(output, features, out_mask, retain_graph=True) # calculate gradient of output w.r.t features
    grads = []

    if isinstance(features,(list,tuple)):
        for stream in range(len(features)):
            sample_stream = samples[stream]
            feature_stream = features[stream].data.numpy()
            grad = feature_grads[stream].data.numpy()
            weights = np.mean(grad, axis=tuple(range(2,len(grad.shape)))) # average over kernel
            batch_cam = torch.zeros_like(sample_stream)
            for sample_idx in range(samples[0].shape[0]):
                sample_feature_stream = feature_stream[sample_idx,...]
                sample_weight = weights[sample_idx,:]
                cam = np.zeros(feature_stream.shape[1:], dtype=np.float32)
                for i, w in enumerate(sample_weight):
                    cam += w * sample_feature_stream[i,...]
                cam = np.max(cam,0)
                if len(cam.shape) > 2:
                    nshape = [sample_stream.shape[2]/cam.shape[1],sample_stream.shape[3]/cam.shape[2],sample_stream.shape[4]/cam.shape[3]]
                else:
                    nshape = [sample_stream.shape[2]/cam.shape[1],sample_stream.shape[3]/cam.shape[2]]
                cam = zoom(cam,nshape)
                cam = cam - np.min(cam)
                cam = cam / np.max(cam)
                batch_cam[sample_idx] = cam
            grads.append(cam)

method_dict = {
    "lrp": _lrp,
    "grad-cam": _grad_cam
}

class SelectiveRelevanceExplainer:
    """
    Object for applying the Selective Relevance method to a video explanation. Basically a wrapper for a torch Sobel operator based masking algorithm.

    Attributes:
        sobel (torch.Tensor): The Sobel kernel applied to the explanation to obtain the relevance gradient.
        sig (int): The number of standard deviations to threshold the relevance gradient at. Positions for where the gradient is larger than
            sig*rel_grad.std() are considered to be relevant due to motion.
        device (str, default:cpu): torch device to move model and samples to
    """
    def __init__(self, sig_val, device, **_):
        """
        Args:
            sig_val (int): See Attributes
        """
        self.sig = sig_val
        self.sobel = torch.tensor([[[1,2,1],[2,4,2],[1,2,1]],[[0,0,0],[0,0,0],[0,0,0]],[[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]]])
        self.sobel = self.sobel.reshape((1,1,3,3,3)).to(device)
        self.device = device

    def get_exp(
            self,
            samples,
            mdl,
            target=-1,
            base_method="lrp",
            target_layers="",
            keep_channels=False,
            **_):
        """
        Backprop relevance onto an input sample for a given model
        Args:
            samples (torch.Tensor): Batch of samples to run through the model, to then backpropagate relevance onto
            mdl (torch.nn.Module): torchexplain model to explain
            target (int, default:1): the class to generate relevance towards from the input.
            cmap (plt.Cmap, default:None): return raw gradient tensor vs normalise and map to palette first
            target_layers (str/list of (str), default:""): the layer(s) to start the backwards process from.
            keep_channels (bool, default:False): if false, sums along the channel dimension of the tensor
        Returns:
            grad (torch.Tensor): Relevance in shape of samples
            target (torch.Tensor): One value tensor containing the target class index
        """
        if mdl.training:
            mdl = mdl.eval().to(self.device)
        # Forward pass
        if type(samples) == list:
            samples = [sample.cuda().requires_grad_() for sample in samples]
        else:
            samples = samples.cuda().requires_grad_()
        grad, target = method_dict[base_method](samples,mdl,target,target_layers)
        return grad,target

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

    def selective_relevance(
            self,
            expl_tensor,
            **_
    ):
        """
        Apply mask to explanation based on it's gradient over time, i.e. how quickly relevance changes in a region.

        Args:
            expl_tensor (torch.Tensor): Grey-scale magnitude representation of relevance for an input video. Each pixels value should be the
                amount of relevance at that position. Overlaying the explanation on the input, or putting it through some colour map will cause
                incorrect results.
        Returns:
            sel_expl (list of numpy.ndarrays): Explanation tensor post Selective process.
        """

        expl_tensor = expl_tensor.to(self.device)
        if len(expl_tensor.shape)>3:
            expl_tensor = expl_tensor.sum(0)

        if expl_tensor.max() != 1:
            expl_tensor = self.normalise(expl_tensor)
        # if expl_tensor.min() < 0:
        #     expl_tensor = (expl_tensor - -1)/(1 - -1)



        # sobel operator expects a batch and channel dimension, it also requires padding to fit to
        #     even dimensions but this can be altered.
        deriv_t = abs(F.conv3d(expl_tensor[None][None].float(), self.sobel.float(), padding=(1, 1, 1))[0, 0, ...])
        # this is the selective process in essentially one line: constructing the mask and applying it
        temp_vis = expl_tensor * (deriv_t > (deriv_t.std() * self.sig)).float()
        temp_vis = temp_vis[None,...].repeat([3,1,1,1])
        return temp_vis

    def compare_with_flow(self,exp_frames, flow_frames):
        """
        For a list of frames of an explanation, and a list of optical flow fields calculated between the
        input frames of said explanation, calculate the pixelwise overlap (precision) so as to evaluate the explanations bias
        towards regions of motion.

        Args:
            exp_frames (list of np.ndarrays): A list of frames of an explanation for a 3D CNN
            flow_frames (list of np.ndarrays): A list of optical flow fields, each calculated as the motion between
            neighbouring frames at it's corresponding position in the input to the 3D CNN
        Returns:
            The percentage overlap between positive valued pixels in frames of both lists.
        """
        true_pos = 0
        false_pos = 0
        for ex, fl in zip(exp_frames,flow_frames):
            if ex.shape != fl.shape:
                ex = cv2.resize(ex,fl.shape[1::-1])
            ex[ex < 10] = 0
            fl_mask = cv2.cvtColor(fl,cv2.COLOR_BGR2GRAY)
            fl_mask = fl_mask > fl_mask.std()

            ex = ex > 0

            true_pos += (ex & fl_mask).sum()
            false_pos += (ex & (~fl_mask)).sum()

        if true_pos or false_pos:
            precision = true_pos / (true_pos+false_pos)
        else:
            precision = 0
        return precision

    def compile_batches(self, batches, lbls, streams=1, **_):
        clip = []
        if streams == 1:
            clip = torch.cat(batches,0)
        else:
            for s in range(streams):
                clip.append(
                    torch.cat(
                        [c[s] for c in batches],
                        0
                    )
                )
        lbls = torch.stack(lbls)
        return clip, lbls

    def normalise(
            self,
            expl_tensor
    ):
        if expl_tensor.min() < 0:
            expl_tensor = abs(expl_tensor)
        expl_tensor = (expl_tensor - expl_tensor.min())/(expl_tensor.max() - expl_tensor.min())
        return expl_tensor

    def visualise(
            self,
            expl_tensor,
            cmap=None,
            inp=None
    ):
        # Args:
        #     expl_tensor (torch.Tensor): Grey-scale magnitude representation of relevance for an input video. Each pixels value should be the
        #         amount of relevance at that position. Overlaying the explanation on the input, or putting it through some colour map will cause
        #         incorrect results.
        #     cmap (str, optional): Name of matplotlib.pyplot colourmap to apply to the resulting Selective Relevance map.
        #     inp (list of (numpy.ndarray),optional): The input video on which to overlay the resulting Selective Relevance map.
        # Returns:
        #     expl (list of numpy.ndarrays): List of frames for the Selective Relevance map.
        if cmap:
            cmap = plt.get_cmap(cmap)
        expl_tensor = expl_tensor.cpu().numpy()
        expl = []
        if len(expl_tensor.shape) > 4:
            expl_tensor = expl_tensor.sum(1)
        expl_tensor = self.normalise(expl_tensor)

        for batch_idx, batch in enumerate(expl_tensor):
            batch_sz = batch.shape[0]
            for f_idx in range(batch.shape[0]):
                frame = batch[f_idx]
                if cmap:
                    frame = cmap(frame)
                    frame = frame[...,(2,1,0)]
                frame = (frame*255).astype(np.uint8)
                if inp:
                    in_frame = inp[batch_idx*batch_sz+f_idx]
                    frame = cv2.resize(frame,(in_frame.shape[1],in_frame.shape[0]))
                    frame = cv2.addWeighted(frame,0.5,in_frame,0.5,0)
                expl.append(frame)
        return expl
