import os
import torch
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import imageio
import cv2
import random
# import imageio
from . import robust_pca
import pdb
from tqdm import tqdm

supported_imgs = [
    '.png',
    '.jpg'
]

supported_vids = [
    '.mp4',
    '.avi'
]

def get_video_list(
        dataset_path,
        num_vids=-1,
        cache_file="",
        class_list=[],
        sampling_method="uniform",
        extension='.mp4',
        sample_len=16,
        train=False,
        **kwargs
):
    """
    Randomly generate a list of videos for a given dataset.
    Args:
        dataset_path (str): the absolute path to the dataset folder on your
            system.
        num_vids (int): the desired length of the return list
        cache_file (str, default:""): a list of video names saved to disk,
            these are prepended to the list before generating
        class_list (list of str, default:[]): if your dataset folder has
            subfolders that aren't classes, then provide a list of class names with this arg.
        sampling_method (str, default="uniform"): The method by which the
            number of samples for each class is calculated. Viable
            options are uniform/proportional
    Returns:
        vid_list (list of str): list of absolute file names to read videos from when generating results.
    """
    vid_list = []
    if cache_file:
        # prepend cache list to list of samples
        cache_list = open(cache_file,"r").readlines()
        cache_list = [c[:-1] for c in cache_list]
        if cache_list:
            vid_list += cache_list
    if not class_list:
        # default class_list
        class_list = os.listdir(dataset_path)
    # get abs paths for class folders
    class_list = [os.path.join(dataset_path,c) for c in class_list if os.path.isdir(os.path.join(dataset_path,c))]
    # remainder of videos to be sampled randomly
    if isinstance(eval(vid_list[0]),tuple):
        if num_vids == len(set([eval(v)[0] for v in vid_list])):
            return vid_list
    if num_vids == len(vid_list):
        return vid_list
    dataset_size = 0
    sample_list = []
    for c in class_list:
        samples = os.listdir(c)
        # get absolute path of all video files in class folder
        if extension in supported_vids:
            samples = [os.path.join(c,s) for s in samples if s.endswith(extension)]
        else:
            new_samples = []
            for s in samples:
                abs_path = os.path.join(c,s)
                if os.path.isdir(abs_path):
                    for f in os.listdir(abs_path):
                        if f.endswith(extension):
                            new_samples.append(abs_path)
                            break
            samples = new_samples

        num_samples = len(samples)
        sample_list.append(samples)
        dataset_size += num_samples
    if num_vids == -1:
        num_vids = dataset_size
    if sampling_method == "proportional":
        # get weighted number of samples for each class, according to their
        # percentage size within the entire dataset
        # for cl, samples in zip(class_list,sample_list):
        #     pdb.set_trace()
        #     print(cl,samples)
        weights = []
        for samples in sample_list:
            weight = round(
                (num_vids-len(vid_list))*len(samples)/dataset_size
            )
            weights.append(weight)
    else:
        weights = [1]*len(class_list)

    if num_vids < dataset_size:
        sample_key = sorted(random.choices(list(range(len(weights))),weights=weights,k=num_vids-len(vid_list)))

        for c in sample_key:
            sample = random.choice(sample_list[c])
            while sample in vid_list:
                sample = random.choice(sample_list[c])
            vid_list += [sample]
    else:
        vid_list = [s for c in sample_list for s in c]
    if train:
        n_vid_list = []
        for v in tqdm(vid_list):
            if extension in supported_imgs:
                len_vid = len(os.listdir(v))
            elif extension in supported_vids:
                len_vid = cv2.VideoCapture(v).get(cv2.CAP_PROP_FRAME_COUNT)
            offsets = math.ceil(len_vid / sample_len)
            for f_idx in range(0,len_vid,sample_len):
                n_vid_list.append((v,f_idx,f_idx+sample_len))
        vid_list = n_vid_list
    with open("/mnt/hdd/datasets/PHAV/phav_v2-img/cache.txt",'w') as f:
        for v in vid_list:
            f.write(str(v)+'\n')
    return vid_list

class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        class_list,
        shape=None,
        mean=[0,0,0],
        std=[1,1,1],
        sample_len=16,
        streams=1,
        sample_rate=1,
        extension='.mp4',
        device='cpu',
        **kwargs
    ):
        self.dataset_path = dataset_path
        self.vid_list = get_video_list(dataset_path, class_list=class_list, extension=extension, sample_len=sample_len, **kwargs)
        self.classes = class_list
        self.shape = shape
        self.mean = mean
        self.std = std
        self.sample_len = sample_len
        self.streams = streams
        self.sample_rate = sample_rate
        self.extension = extension
        self.kwargs = kwargs

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        start, stop = (0,-1)
        path = self.vid_list[idx]
        if isinstance(path,str):
            path = eval(path)
        if isinstance(path,tuple):
            path, start, stop = path
        activity = self.classes.index(path.split('/')[-2])
        if self.extension in supported_vids:
            return get_input(f"{path}", self.shape, self.mean, self.std,
                  self.sample_len, self.streams, self.sample_rate, start, stop,
                  **self.kwargs), (activity, path)
        elif self.extension in supported_imgs:
            return get_input_from_frames(f"{path}", shape=self.shape, mean=self.mean, std=self.std,
                  sample_len=self.sample_len, streams=self.streams, sample_rate=self.sample_rate,
                  start=start, stop=stop, **self.kwargs), (activity, path)

    def collate(self,batch):
        inp, labels = list(zip(*batch))
        vid, frames = list(zip(*inp))
        gts, path = list(zip(*labels))
        n_gts = []
        for v, g in zip(vid,gts):
            n_gts += [g]*v.shape[0]
        vid = torch.cat(vid,dim=0)
        n_gts = torch.tensor(n_gts)
        return (vid, frames), (n_gts, path)


class FlowDataset(VideoDataset):
    def __init__(
        self,
        dataset_path,
        class_list,
        shape=None,
        mean=[0,0,0],
        std=[1,1,1],
        sample_len=16,
        streams=1,
        sample_rate=1,
        extension='.mp4',
        motion_compensation = False,
        device='cpu',
        channels=3,
        **kwargs
    ):
        super().__init__(
            dataset_path,
            class_list,
            shape,
            mean,
            std,
            sample_len,
            streams,
            sample_rate,
            extension,
            **kwargs
        )
        self.motion_compensation = motion_compensation
        self.channels=channels

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self,idx):
        start, stop = (0,-1)
        path = self.vid_list[idx]
        if isinstance(path,str):
            path = eval(path)
        if isinstance(path,tuple):
            path, start, stop = path
        activity = self.classes.index(path.split('/')[-2])
        if self.extension in supported_vids:
            vid, frames = get_input(f"{path}", self.shape, self.mean, self.std,
                  self.sample_len, self.streams, self.sample_rate, start, stop,
                  **self.kwargs)
            shape = vid.shape
            del vid
            flow_frames =  generate_optical_flow(frames,self.motion_compensation,self.streams,self.sample_rate,channels=self.channels)
            flow = [torch.from_numpy(f.transpose(2,0,1)) for f in flow_frames]
            pad_len = shape[0]*shape[2] - len(flow)
            flow = torch.stack(flow+[torch.zeros_like(flow[-1])]*pad_len).reshape(shape).float()
            for c in range(3):
                flow[:,c,...] -= self.mean[c]
                flow[:,c,...] /= self.std[c]
        elif self.extension in supported_imgs:
            flow, flow_frames = get_flow_from_frames(f"{path}", shape=self.shape, mean=self.mean, std=self.std,
                  sample_len=self.sample_len, streams=self.streams, sample_rate=self.sample_rate,
                  start=start, stop=stop, channels=self.channels, **self.kwargs)
        return (flow, flow_frames), (activity, path)

    def collate(self,batch):
        inp, labels = list(zip(*batch))
        vid, frames = list(zip(*inp))
        gts, path = list(zip(*labels))
        n_gts = []
        for v, g in zip(vid,gts):
            n_gts += [g]*v.shape[0]
        vid = torch.cat(vid,dim=0)
        n_gts = torch.tensor(n_gts)
        return (vid, frames), (n_gts, path)

def get_input(path, shape=None, mean=[0,0,0], std=[1,1,1],sample_len=16, streams=1, sample_rate=1, start=0, stop=-1, **kwargs):
    """
    Read in video from file and store in a torch.Tensor.

    Args:
        path (str): the path to the video file.
        shape (tuple/list of int,optional): the desired shape to resize each frame to.
        mean (tuple/list of int,optional): channel-wise mean to center the channels of each frame around, in RGB format.
        std (tuple/list of int,optional): channel-wise standard deviation to scale the channels of each frame by, in RGB format.
        sample_len (int,optional): the length in frames of each sample expected by the model, the video will be split
            into batches accordingly.
        streams (int, default: 1): the number of streams the target model will have, we repeat the input accordingly so
            that each stream has a copy of the input.
        sample_rate (int/tuple/list, default: 1): frames for the video will be sampled at a rate of 1/sample_rate, if multiple
            streams, then a sample_rate must be provided for each of them
    Returns:
        vid (torch.Tensor): The processed video with BxCxTxHxW shape.
        frames (list of (np.ndarray)): A list of frames
    """
    rdr = cv2.VideoCapture(path)
    offsets = math.ceil(rdr.get(cv2.CAP_PROP_FRAME_COUNT)/sample_len)
    if shape is None:
        shape = (int(rdr.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(rdr.get(cv2.CAP_PROP_FRAME_WIDTH)))
    else:
        shape = tuple(shape)
    vid = torch.zeros((offsets,3,sample_len) + shape)
    frames = []
    for o in range(offsets):
        f_idx = 0
        while True:
            if f_idx < start or f_idx > stop:
                f_idx += 1
                continue
            if f_idx > sample_len-1:
                break
            r, frame = rdr.read()
            if not r:
                break
            frame = cv2.resize(frame,(shape[1],shape[0]))
            frames.append(frame)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = frame.transpose(2,0,1)
            frame = torch.from_numpy(frame).float()
            # if len(frames):
            #     print((frame==frames[-1]).all())
            for c in range(3):
                frame[c,...] -= mean[c]
                frame[c,...] /= std[c]
            vid[o,:,f_idx,:,:] = frame
            f_idx += 1
        if f_idx < sample_len:
            vid[o,:,f_idx:sample_len,:,:] = vid[o,:,f_idx:f_idx+1,:,:]
    if not len(frames):
        return False, False
    # some models work with multiple streams, e.g. slowfast, where you need input to each stream   
    if streams > 1:
        if isinstance(sample_rate,int):
            sample_rate = [sample_rate]*streams
        stream_copies = [vid.clone()] * streams
        # resample each copy of the input at it's respective sample rate
        for s in range(streams):
            stream_copies[s] = stream_copies[s][:,:,::sample_rate[s],...]
        vid = stream_copies
    else:
        vid = vid[:,:,::sample_rate,...]
    return vid,frames

def get_input_from_frames(path, shape=None, mean=[0,0,0], std=[1,1,1],sample_len=16, streams=1, sample_rate=1, extension='', start=0, stop=-1, **kwargs):
    """
    Read images from directory and compile into 4D tensor
    Args:
        path (str): the path to the frame directory.
        shape (tuple/list of int,optional): the desired shape to resize each frame to.
        mean (tuple/list of int,optional): channel-wise mean to center the channels of each frame around, in RGB format.
        std (tuple/list of int,optional): channel-wise standard deviation to scale the channels of each frame by, in RGB format.
        sample_len (int,optional): the length in frames of each sample expected by the model, the video will be split
            into batches accordingly.
        streams (int, default: 1): the number of streams the target model will have, we repeat the input accordingly so
            that each stream has a copy of the input.
        sample_rate (int/tuple/list, default: 1): frames for the video will be sampled at a rate of 1/sample_rate, if multiple
            streams, then a sample_rate must be provided for each of them
    Returns:
        vid (torch.Tensor): The processed video with BxCxTxHxW shape.
        frames (list of (np.ndarray)): A list of frames
    """
    frame_files = os.listdir(path)
    frame_files = [f"{path}/{f}" for f in frame_files if f.endswith(extension)]
    frame_files = sorted(frame_files)

    if stop:
        frame_files = frame_files[start:stop]
    offsets = math.ceil(len(frame_files)/sample_len)
    if shape is None:
        first = cv2.imread(frame_files[0])
        shape = first.shape[:-1]
    else:
        shape = tuple(shape)
    vid = torch.zeros((offsets,3,sample_len) + shape)
    frame_files = iter(frame_files)
    frames = []
    for o in range(offsets):
        f_idx = 0
        for frame in frame_files:
            if f_idx > sample_len-1:
                break
            frame = cv2.imread(frame)
            frame = cv2.resize(frame,(shape[1],shape[0]))
            frames.append(frame)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = frame.transpose(2,0,1)
            frame = torch.from_numpy(frame).float()
            # if len(frames):
            #     print((frame==frames[-1]).all())
            for c in range(3):
                frame[c,...] -= mean[c]
                frame[c,...] /= std[c]
            vid[o,:,f_idx,:,:] = frame
            f_idx += 1
        if f_idx < sample_len:
            vid[o,:,f_idx:sample_len,:,:] = vid[o,:,f_idx:f_idx+1,:,:]
    if not len(frames):
        return False, False
    # some models work with multiple streams, e.g. slowfast, where you need input to each stream   
    if streams > 1:
        if isinstance(sample_rate,int):
            sample_rate = [sample_rate]*streams
        stream_copies = [vid.clone()] * streams
        # resample each copy of the input at it's respective sample rate
        for s in range(streams):
            stream_copies[s] = stream_copies[s][:,:,::sample_rate[s],...]
        vid = stream_copies
    else:
        vid = vid[:,:,::sample_rate,...]
    return vid,frames

def get_flow_from_frames(path, shape=None, mean=[0,0,0], std=[1,1,1],sample_len=16, streams=1, sample_rate=1, extension='', start=0, stop=-1, channels=3, **kwargs):
    """
    Read images from directory and compile into 4D tensor
    Args:
        path (str): the path to the frame directory.
        shape (tuple/list of int,optional): the desired shape to resize each frame to.
        mean (tuple/list of int,optional): channel-wise mean to center the channels of each frame around, in RGB format.
        std (tuple/list of int,optional): channel-wise standard deviation to scale the channels of each frame by, in RGB format.
        sample_len (int,optional): the length in frames of each sample expected by the model, the video will be split
            into batches accordingly.
        streams (int, default: 1): the number of streams the target model will have, we repeat the input accordingly so
            that each stream has a copy of the input.
        sample_rate (int/tuple/list, default: 1): frames for the video will be sampled at a rate of 1/sample_rate, if multiple
            streams, then a sample_rate must be provided for each of them
    Returns:
        vid (torch.Tensor): The processed video with BxCxTxHxW shape.
        frames (list of (np.ndarray)): A list of frames
    """
    frame_files = os.listdir(path)
    frame_files = [f"{path}/{f}" for f in frame_files if f.endswith(extension)]
    x_files = [f for f in frame_files if 'x' in f]
    y_files = [f for f in frame_files if 'y' in f]
    x_files = sorted(x_files)
    y_files = sorted(y_files)

    if stop:
        frame_files = frame_files[start:stop]
    offsets = math.ceil(len(frame_files)/sample_len)
    if shape is None:
        first = cv2.imread(frame_files[0])
        shape = first.shape[:-1]
    else:
        shape = tuple(shape)
    flow = torch.zeros((offsets,channels,sample_len) + shape)
    flow_frames = []
    for o in range(offsets):
        f_idx = 0
        for x,y in zip(x_files,y_files):
            if f_idx > sample_len-1:
                break
            x = cv2.imread(x)
            x = cv2.resize(x,(shape[1],shape[0]))
            x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
            y = cv2.imread(y)
            y = cv2.resize(y,(shape[1],shape[0]))
            y = cv2.cvtColor(y,cv2.COLOR_BGR2GRAY)


            if channels == 2:
                flow_frame = np.zeros(shape+(2,))
                flow_frame[...,0] = x
                flow_frame[...,1] = y
            if channels == 3:
                mag, ang = cv2.cartToPolar(x,y)
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                flow_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                flow_frame = cv2.cvtColor(flow_frame, cv2.COLOR_BGR2GRAY)
                flow_frame = cv2.cvtColor(flow_frame, cv2.COLOR_GRAY2BGR)

            flow_frames.append(flow_frame)

            flow_frame = flow_frame.transpose(2,0,1)
            flow_frame = torch.from_numpy(flow_frame).float()
            # if len(frames):
            #     print((frame==frames[-1]).all())
            for c in range(channels):
                flow_frame[c,...] -= mean[c]
                flow_frame[c,...] /= std[c]
            flow[o,:,f_idx,:,:] = flow_frame
            f_idx += 1
        if f_idx < sample_len:
            flow[o,:,f_idx:sample_len,:,:] = flow[o,:,f_idx:f_idx+1,:,:]
    if not len(flow_frames):
        return False, False
    # some models work with multiple streams, e.g. slowfast, where you need input to each stream   
    if streams > 1:
        if isinstance(sample_rate,int):
            sample_rate = [sample_rate]*streams
        stream_copies = [flow.clone()] * streams
        # resample each copy of the input at it's respective sample rate
        for s in range(streams):
            stream_copies[s] = stream_copies[s][:,:,::sample_rate[s],...]
        flow = stream_copies
    else:
        flow = flow[:,:,::sample_rate,...]
    return flow,flow_frames

def show_image(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def write_video(path,frames,**kwargs):
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

def generate_optical_flow(frames=[], motion_compensation=False, streams=1, sample_rate=1, **kwargs):
    """
    For a list of frames from a video, generate the Dense (Farneback) optical flow fields between each

    Args:
        frames (list of np.ndarrays): list of n frames
        motion_compensation (float/bool, default: False): if float, estimate and negate optical flow due to camera
            motion using this value as a threshold.
        streams (int, default: 1): the number of streams the target model will have, we repeat the optical flow accordingly so
            that each stream has a copy.
        sample_rate (int/tuple/list, default: 1): frames for the video will be sampled at a rate of 1/sample_rate, if multiple
            streams, then a sample_rate must be provided for each of them (NB: We suggest against using optical flow precision
            on streams that resample the video (!=1))

    Returns:
        flow (list of np.ndarrays): list of n-1 optical flow fields between frames
    """
    if streams > 1:
        assert isinstance(sample_rate,(list,tuple))
        assert len(sample_rate) == streams
        flow = []
        for r in sample_rate:
            resampled = [f for i, f in enumerate(frames) if not (i % r)]
            if motion_compensation:
                s_flow, metric = generate_optical_flow(
                    resampled,
                    motion_compensation=motion_compensation,
                    streams=1,
                    sample_rate=1
                )
                flow.append(s_flow)
            else:
                flow.append(generate_optical_flow(
                    resampled,
                    motion_compensation=motion_compensation,
                    streams=1,
                    sample_rate=1
                ))
    else:
        assert isinstance(sample_rate,int)
        frames = frames[::sample_rate]
        flow = []
        f1 = frames[0]
        hsv = np.zeros_like(f1)
        f1 = cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)
        for f_idx in range(1,len(frames)):
            f2 = cv2.cvtColor(frames[f_idx],cv2.COLOR_BGR2GRAY)
            flow.append(cv2.calcOpticalFlowFarneback(f1,f2,None,0.5,3,15,3,5,1.2,0))
        if motion_compensation:
            flow, metric = remove_cam_motion(flow,motion_compensation,**kwargs)
        for i,cart in enumerate(flow):
            mag, ang = cv2.cartToPolar(cart[...,0], cart[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            flow[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            flow[i] = cv2.cvtColor(flow[i], cv2.COLOR_BGR2GRAY)
            flow[i] = cv2.cvtColor(flow[i], cv2.COLOR_GRAY2BGR)
            f1 = f2
    if motion_compensation:
        return flow, metric
    return flow

def remove_cam_motion(
        flow,
        thresh=1,
        retain=False,
        **kwargs
):
    """
    Camera motion can be suppressed in an optical flow field, via particle advection and rank optimisation.
    Args:
        frames (list of np.ndarrays): list of n frames
        flow (list of np.ndarrays): list of n-1 optical flow fields between frames represented as [X,Y] (cv2 cart)
        thresh (float): a value to threshold what constitutes as 'camera motion'. The threshold is indirectly proportional
            to the proportion of the screen that has to be moving in one direction for you to consider it camera motion.
        retain (bool): if True, the degree of camera motion will be returned, otherwise it will only return the detection.
    Returns:
        flow (list of np.ndarrays): list of n-1 optical flow fields between frames where each has had flow
            due to camera motion removed.
    References:
        [1]: Shandong Wu, Omar Oreifej, and Mubarak Shah,
            Action Recognition in Videos Acquired by a Moving Camera Using Motion Decomposition of Lagrangian Particle Trajectories,
            International Conference on Computer Vision, November 2011, Barcelona, Spain.
    """
    fshape = (120,120)
    x = np.arange(fshape[1]) # x component of clip position vector
    x = np.repeat(x[None,...],fshape[0],axis=0)
    y = np.arange(fshape[0]) # y component of clip position vector
    y = np.repeat(y[...,None],fshape[1],axis=1)
    m = np.zeros((len(flow)*2,np.prod(fshape))) # particle trajectory set for clip
    m[0,:] = x.reshape(-1)
    m[1,:] = y.reshape(-1)
    for f,cart in enumerate(flow[:-1]):
        # We use particle advection to estimate the position of each particle (pixel) in the frame in the next frame
        cart = cv2.resize(cart,fshape)
        u = cart[...,0].reshape(-1)
        v = cart[...,1].reshape(-1)
        m[(1+f)*2] = m[f*2] + u
        m[1+(1+f)*2] = m[1+f*2] + v
    # # Perform low rank optimisation on M to find A and E s.t. M=A+E using r-pca
    rpca = robust_pca.R_pca(m)
    # Given the low rank representation, we can assume from [1] that:
    # A accounts for camera and rigid body motion
    # E accounts for articulated motion
    A, E = rpca.fit(max_iter=1, iter_print=1000)
    # Further from [1] if we assume camera motion IS present then
    # A_c, a subset of A containing camera motion, is dominant in A.
    U, S, V = torch.svd(A)
    # By approximating this dominance to the top 3 sing. values of S, S* we can reconstruct A_c (UxS*xV')
    # Since we only want to find the ratio A_c / A = USV'/US*V'
    # We can instead compute sum(S*)/sum(S), therefore describing the amount of energy in A accounted for by A_c
    metric = S[:1].sum() / S.sum()
    if metric < thresh:
        # We assume this is camera motion
        S_star = torch.cat((S[:1],torch.zeros_like(S[1:])))
        A_c = torch.matmul(U*S_star,V.t())
        # Construct a mask to remove camera motion
        E_t = E + A - A_c
        E_t = E_t.reshape(E_t.shape[0:1]+fshape)
        E_t = torch.stack([E_t[::2],E_t[1::2]],-1)
        # Remap E_t to the shape (T,H,W)
        E_t = E_t.cpu().numpy()
        for idx, f in enumerate(E_t):
            f = cv2.resize(f,flow[0].shape[0:2])
            mask = np.zeros_like(flow[0])
            # f is trajectories for each frame
            mask[...,:] += (f > 0)
            flow[idx] = flow[idx] * mask
    if retain:
        return flow, metric
    return flow, metric<thresh
