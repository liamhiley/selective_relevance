#!/usr/bin/env python3
import torch
from torch import nn
from .. import torchexplain
import matplotlib.pyplot as plt

import pdb

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}


class ResNetBasicStem(nn.Module):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            kernel,
            stride,
            padding,
            inplace_relu=True,
            eps=1e-5,
            bn_mmt=0.1,
            norm_module=nn.BatchNorm3d,
            range=(0,255),
            train=False,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(ResNetBasicStem, self).__init__()
        self.training = train
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # Construct the stem layer.
        self.range=range
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        if self.training:
            self.conv = self.lib.Conv3d(
                dim_in,
                dim_out,
                self.kernel,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            )
        else:
            self.conv = self.lib.Conv3d(
                dim_in,
                dim_out,
                self.kernel,
                stride=self.stride,
                padding=self.padding,
                bias=False,
                range=self.range
            )
        self.bn = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )
        self.relu = self.lib.ReLU(self.inplace_relu)
        self.pool_layer = self.lib.MaxPool3d(
            kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
        )

    def forward(self, inp):
        x = self.conv(inp)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool_layer(x)

        return x

class VideoModelStem(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            kernel,
            stride,
            padding,
            inplace_relu=True,
            eps=1e-5,
            bn_mmt=0.1,
            norm_module=nn.BatchNorm3d,
            range=(0,255),
            train=False
    ):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, Slow
        and etc), list size of 2 for two pathway models (SlowFast).
        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(VideoModelStem, self).__init__()

        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(kernel),
                    len(stride),
                    len(padding),
                }
            )
            == 1
        ), "Input pathway dimensions are not consistent."
        self.training = train
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.range=range
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        for pathway in range(len(dim_in)):
            stem = ResNetBasicStem(
                dim_in[pathway],
                dim_out[pathway],
                self.kernel[pathway],
                self.stride[pathway],
                self.padding[pathway],
                self.inplace_relu,
                self.eps,
                self.bn_mmt,
                norm_module,
                range=self.range,
                train=self.training
            )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def forward(self, inp):
        x = []
        for pathway, i in enumerate(inp):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x.append(m(i))
        return x

class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
            self,
            dim_in,
            fusion_conv_channel_ratio,
            fusion_kernel,
            alpha,
            eps=1e-5,
            bn_mmt=0.1,
            inplace_relu=True,
            norm_module=nn.BatchNorm3d,
            train=False
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.training = train
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain
        self.conv_f2s = self.lib.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = self.lib.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]

def get_trans_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {
        "bottleneck_transform": BottleneckTransform,
        "basic_transform": BasicTransform,
    }
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]

class BasicTransform(nn.Module):
    """
    Basic transformation: Tx3x3, 1x3x3, where T is the size of temporal kernel.
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner=None,
            num_groups=1,
            stride_1x1=None,
            inplace_relu=True,
            eps=1e-5,
            bn_mmt=0.1,
            norm_module=nn.BatchNorm3d,
            train=False
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (None): the inner dimension would not be used in
                BasicTransform.
            num_groups (int): number of groups for the convolution. Number of
                group is always 1 for BasicTransform.
            stride_1x1 (None): stride_1x1 will not be used in BasicTransform.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BasicTransform, self).__init__()
        self.training = train
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(dim_in, dim_out, stride, norm_module)

    def _construct(self, dim_in, dim_out, stride, norm_module):
        # Tx3x3, BN, ReLU.
        self.a = self.lib.Conv3d(
            dim_in,
            dim_out,
            kernel_size=[self.temp_kernel_size, 3, 3],
            stride=[1, stride, stride],
            padding=[int(self.temp_kernel_size // 2), 1, 1],
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = self.lib.ReLU(inplace=self._inplace_relu)
        # 1x3x3, BN.
        self.b = self.lib.Conv3d(
            dim_out,
            dim_out,
            kernel_size=[1, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 1, 1],
            bias=False,
        )
        self.b_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )

        self.b_bn.transform_final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        x = self.b(x)
        x = self.b_bn(x)
        return x


class BottleneckTransform(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=False,
            inplace_relu=True,
            eps=1e-5,
            bn_mmt=0.1,
            dilation=1,
            norm_module=nn.BatchNorm3d,
            train=False
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BottleneckTransform, self).__init__()
        self.training=train
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        if str1x1 > 1:
            print("here")
        # Tx1x1, BN, ReLU.
        self.a = self.lib.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = self.lib.ReLU(inplace=self._inplace_relu)

        # 1x3x3, BN, ReLU.
        self.b = self.lib.Conv3d(
            dim_inner,
            dim_inner,
            [1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        self.b_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu = self.lib.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = self.lib.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        # Explicitly forward every layer.
        # Branch2a.
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        # Branch2b.
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups=1,
            stride_1x1=False,
            inplace_relu=True,
            eps=1e-5,
            bn_mmt=0.1,
            dilation=1,
            norm_module=nn.BatchNorm3d,
            train=False
    ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(ResBlock, self).__init__()
        self.training = train
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = self.lib.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            norm_module=norm_module,
            train=self.training,
        )
        self.relu = self.lib.ReLU(self._inplace_relu)

    def forward(self, x):
        if hasattr(self, "branch1"):
            if self.training:
                x = self.branch1_bn(self.branch1(x)) + self.branch2(x)
            else:
                x = self.lib.add(self.branch1_bn(self.branch1(x)), self.branch2(x))
        else:
            if self.training:
                x = x + self.branch2(x)
            else:
                x = self.lib.add(x, self.branch2(x))
        x = self.relu(x)
        return x

class Nonlocal(nn.Module):
    """
    Builds Non-local Neural Networks as a generic family of building
    blocks for capturing long-range dependencies. Non-local Network
    computes the response at a position as a weighted sum of the
    features at all positions. This building block can be plugged into
    many computer vision architectures.
    More details in the paper: https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(
            self,
            dim,
            dim_inner,
            pool_size=None,
            instantiation="softmax",
            zero_init_final_conv=False,
            zero_init_final_norm=True,
            norm_eps=1e-5,
            norm_momentum=0.1,
            norm_module=nn.BatchNorm3d,
            train=False
    ):
        """
        Args:
            dim (int): number of dimension for the input.
            dim_inner (int): number of dimension inside of the Non-local block.
            pool_size (list): the kernel size of spatial temporal pooling,
                temporal pool kernel size, spatial pool kernel size, spatial
                pool kernel size in order. By default pool_size is None,
                then there would be no pooling used.
            instantiation (string): supports two different instantiation method:
                "dot_product": normalizing correlation matrix with L2.
                "softmax": normalizing correlation matrix with Softmax.
            zero_init_final_conv (bool): If true, zero initializing the final
                convolution of the Non-local block.
            zero_init_final_norm (bool):
                If true, zero initializing the final batch norm of the Non-local
                block.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(Nonlocal, self).__init__()
        self.training=train
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain
        self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.use_pool = (
            False
            if pool_size is None
            else any((size > 1 for size in pool_size))
        )
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(
            zero_init_final_conv, zero_init_final_norm, norm_module
        )

    def _construct_nonlocal(
        self, zero_init_final_conv, zero_init_final_norm, norm_module
    ):
        # Three convolution heads: theta, phi, and g.
        self.conv_theta = self.lib.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi = self.lib.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_g = self.lib.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )

        # Final convolution output.
        self.conv_out = self.lib.Conv3d(
            self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0
        )
        # Zero initializing the final convolution output.
        self.conv_out.zero_init = zero_init_final_conv

        # TODO: change the name to `norm`
        self.bn = norm_module(
            num_features=self.dim,
            eps=self.norm_eps,
            momentum=self.norm_momentum,
        )
        # Zero initializing the final bn.
        self.bn.transform_final_bn = zero_init_final_norm

        # Optional to add the spatial-temporal pooling.
        if self.use_pool:
            self.pool = self.lib.MaxPool3d(
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding=[0, 0, 0],
            )

    def forward(self, x):
        x_identity = x
        N, C, T, H, W = x.size()

        theta = self.conv_theta(x)

        # Perform temporal-spatial pooling to reduce the computation.
        if self.use_pool:
            x = self.pool(x)

        phi = self.conv_phi(x)
        g = self.conv_g(x)

        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)

        # (N, C, TxHxW) * (N, C, TxHxW) => (N, TxHxW, TxHxW).
        theta_phi = torch.einsum("nct,ncp->ntp", (theta, phi))
        # For original Non-local paper, there are two main ways to normalize
        # the affinity tensor:
        #   1) Softmax normalization (norm on exp).
        #   2) dot_product normalization.
        if self.instantiation == "softmax":
            # Normalizing the affinity tensor theta_phi before softmax.
            theta_phi = theta_phi * (self.dim_inner ** -0.5)
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == "dot_product":
            spatial_temporal_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_temporal_dim
        else:
            raise NotImplementedError(
                "Unknown norm type {}".format(self.instantiation)
            )

        # (N, TxHxW, TxHxW) * (N, C, TxHxW) => (N, C, TxHxW).
        theta_phi_g = torch.einsum("ntg,ncg->nct", (theta_phi, g))

        # (N, C, TxHxW) => (N, C, T, H, W).
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, T, H, W)

        p = self.conv_out(theta_phi_g)
        p = self.bn(p)
        if self.training:
            return x_identity + p
        else:
            return self.lib.add(x_identity, p)

class ResStage(nn.Module):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, Slow), and multi-pathway (SlowFast) cases.
        More details can be found here:
        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "SlowFast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            stride,
            temp_kernel_sizes,
            num_blocks,
            dim_inner,
            num_groups,
            num_block_temp_kernel,
            nonlocal_inds,
            nonlocal_group,
            nonlocal_pool,
            dilation,
            instantiation="softmax",
            trans_func_name="bottleneck_transform",
            stride_1x1=False,
            inplace_relu=True,
            norm_module=nn.BatchNorm3d,
            train=False
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            nonlocal_inds (list): If the tuple is empty, no nonlocal layer will
                be added. If the tuple is not empty, add nonlocal layers after
                the index-th block.
            dilation (list): size of dilation for each pathway.
            nonlocal_group (list): list of number of p nonlocal groups. Each
                number controls how to fold temporal dimension to batch
                dimension before applying nonlocal transformation.
                https://github.com/facebookresearch/video-nonlocal-net.
            instantiation (string): different instantiation for nonlocal layer.
                Supports two different instantiation method:
                    "dot_product": normalizing correlation matrix with L2.
                    "softmax": normalizing correlation matrix with Softmax.
            trans_func_name (string): name of the the transformation function apply
                on the network.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(ResStage, self).__init__()
        assert all(
            (
                num_block_temp_kernel[i] <= num_blocks[i]
                for i in range(len(temp_kernel_sizes))
            )
        )
        self.training=train
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[: num_block_temp_kernel[i]]
            + [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temp_kernel_sizes),
                    len(stride),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                    len(num_block_temp_kernel),
                    len(nonlocal_inds),
                    len(nonlocal_group),
                }
            )
            == 1
        )
        self.num_pathways = len(self.num_blocks)
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            nonlocal_inds,
            nonlocal_pool,
            instantiation,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        trans_func_name,
        stride_1x1,
        inplace_relu,
        nonlocal_inds,
        nonlocal_pool,
        instantiation,
        dilation,
        norm_module,
    ):
        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                # Retrieve the transformation function.
                trans_func = get_trans_func(trans_func_name)
                # Construct the block.
                res_block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    stride[pathway] if i == 0 else 1,
                    trans_func,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=norm_module,
                    train=self.training
                )
                self.add_module("pathway{}_res{}".format(pathway, i), res_block)
                if i in nonlocal_inds[pathway]:
                    nln = Nonlocal(
                        dim_out[pathway],
                        dim_out[pathway] // 2,
                        nonlocal_pool[pathway],
                        instantiation=instantiation,
                        norm_module=norm_module,
                        train=self.training
                    )
                    self.add_module(
                        "pathway{}_nonlocal{}".format(pathway, i), nln
                    )

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m(x)
                if hasattr(self, "pathway{}_nonlocal{}".format(pathway, i)):
                    nln = getattr(
                        self, "pathway{}_nonlocal{}".format(pathway, i)
                    )
                    b, c, t, h, w = x.shape
                    if self.nonlocal_group[pathway] > 1:
                        # Fold temporal dimension into batch dimension.
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(
                            b * self.nonlocal_group[pathway],
                            t // self.nonlocal_group[pathway],
                            c,
                            h,
                            w,
                        )
                        x = x.permute(0, 2, 1, 3, 4)
                    x = nln(x)
                    if self.nonlocal_group[pathway] > 1:
                        # Fold back to temporal dimension.
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(b, t, c, h, w)
                        x = x.permute(0, 2, 1, 3, 4)
            output.append(x)

        return output

class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
            self,
            dim_in,
            num_classes,
            pool_size,
            dropout_rate=0.0,
            act_func="softmax",
            train=False
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].
        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.training=train
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            avg_pool = self.lib.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = self.lib.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = False
            # raise NotImplementedError(
            #     "{} is not supported as an activation"
            #     "function.".format(act_func)
            # )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout") and self.training:
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            if self.act:
                x = self.act(x)
                x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x

class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.
    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self,
                 train=False,
                 **kwargs
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            **kwargs
        """
        super(SlowFast, self).__init__()
        self.training=train
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain
        self.norm_module = self.lib.BatchNorm3d
        self.enable_detection = False
        self.num_pathways = 2
        self._construct_network(**kwargs)

    def _construct_network(self,
                           resnet,
                           slowfast,
                           non_local,
                           data,
                           mdl

    ):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            resnet (dict): config dict for resnet specs.
            slowfast (dict): config dict for slowfast specs.
            non_local (dict): config dict for non_local specs.
            data (dict): config dict for data specs.
            mdl (dict): config dict for mdl specs.
        """
        pool_size = [[1, 1, 1], [1, 1, 1]]

        assert len({len(pool_size), self.num_pathways}) == 1
        assert resnet["depth"] in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[resnet["depth"]]

        dim_inner = resnet["num_groups"] * resnet["width_per_group"]
        out_dim_ratio = (
            slowfast["beta_inv"] // slowfast["fusion_conv_ch_ratio"]
        )

        # Basis of temporal kernel sizes for each of the stage.
        temp_kernel = [
            [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
            [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
            [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
        ]

        self.s1 = VideoModelStem(
            dim_in=data["in_channels"],
            dim_out=[resnet["width_per_group"], resnet["width_per_group"] // slowfast["beta_inv"]],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
            range=data["range"],
            train=self.training,
        )
        self.s1_fuse = FuseFastToSlow(
            resnet["width_per_group"] // slowfast["beta_inv"],
            slowfast["fusion_conv_ch_ratio"],
            slowfast["fusion_kernel_sz"],
            slowfast["alpha"],
            norm_module=self.norm_module,
            train=self.training,
        )

        self.s2 = ResStage(
            dim_in=[
                resnet["width_per_group"] + resnet["width_per_group"] // out_dim_ratio,
                resnet["width_per_group"] // slowfast["beta_inv"],
            ],
            dim_out=[
                resnet["width_per_group"] * 4,
                resnet["width_per_group"] * 4 // slowfast["beta_inv"],
            ],
            dim_inner=[dim_inner, dim_inner // slowfast["beta_inv"]],
            temp_kernel_sizes=temp_kernel[1],
            stride=resnet["spatial_strides"][0],
            num_blocks=[d2] * 2,
            num_groups=[resnet["num_groups"]] * 2,
            num_block_temp_kernel=resnet["n_block_temp_kernel"][0],
            nonlocal_inds=non_local["loc"][0],
            nonlocal_group=non_local["group"][0],
            nonlocal_pool=non_local["pool"][0],
            instantiation=non_local["instantiation"],
            trans_func_name=resnet["trans_func"],
            dilation=resnet["spatial_dilations"][0],
            norm_module=self.norm_module,
            train=self.training,
        )
        self.s2_fuse = FuseFastToSlow(
            resnet["width_per_group"] * 4 // slowfast["beta_inv"],
            slowfast["fusion_conv_ch_ratio"],
            slowfast["fusion_kernel_sz"],
            slowfast["alpha"],
            norm_module=self.norm_module,
            train=self.training,
        )

        for pathway in range(self.num_pathways):
            pool = self.lib.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = ResStage(
            dim_in=[
                resnet["width_per_group"] * 4 + resnet["width_per_group"] * 4 // out_dim_ratio,
                resnet["width_per_group"] * 4 // slowfast["beta_inv"],
            ],
            dim_out=[
                resnet["width_per_group"] * 8,
                resnet["width_per_group"] * 8 // slowfast["beta_inv"],
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // slowfast["beta_inv"]],
            temp_kernel_sizes=temp_kernel[2],
            stride=resnet["spatial_strides"][1],
            num_blocks=[d3] * 2,
            num_groups=[resnet["num_groups"]] * 2,
            num_block_temp_kernel=resnet["n_block_temp_kernel"][1],
            nonlocal_inds=non_local["loc"][1],
            nonlocal_group=non_local["group"][1],
            nonlocal_pool=non_local["pool"][1],
            instantiation=non_local["instantiation"],
            trans_func_name=resnet["trans_func"],
            dilation=resnet["spatial_dilations"][1],
            norm_module=self.norm_module,
            train=self.training,
        )
        self.s3_fuse = FuseFastToSlow(
            resnet["width_per_group"] * 8 // slowfast["beta_inv"],
            slowfast["fusion_conv_ch_ratio"],
            slowfast["fusion_kernel_sz"],
            slowfast["alpha"],
            norm_module=self.norm_module,
            train=self.training,
        )

        self.s4 = ResStage(
            dim_in=[
                resnet["width_per_group"] * 8 + resnet["width_per_group"] * 8 // out_dim_ratio,
                resnet["width_per_group"] * 8 // slowfast["beta_inv"],
            ],
            dim_out=[
                resnet["width_per_group"] * 16,
                resnet["width_per_group"] * 16 // slowfast["beta_inv"],
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // slowfast["beta_inv"]],
            temp_kernel_sizes=temp_kernel[3],
            stride=resnet["spatial_strides"][2],
            num_blocks=[d4] * 2,
            num_groups=[resnet["num_groups"]] * 2,
            num_block_temp_kernel=resnet["n_block_temp_kernel"][2],
            nonlocal_inds=non_local["loc"][2],
            nonlocal_group=non_local["group"][2],
            nonlocal_pool=non_local["pool"][2],
            instantiation=non_local["instantiation"],
            trans_func_name=resnet["trans_func"],
            dilation=resnet["spatial_dilations"][2],
            norm_module=self.norm_module,
            train=self.training,
        )
        self.s4_fuse = FuseFastToSlow(
            resnet["width_per_group"] * 16 // slowfast["beta_inv"],
            slowfast["fusion_conv_ch_ratio"],
            slowfast["fusion_kernel_sz"],
            slowfast["alpha"],
            norm_module=self.norm_module,
            train=self.training,
        )

        self.s5 = ResStage(
            dim_in=[
                resnet["width_per_group"] * 16 + resnet["width_per_group"] * 16 // out_dim_ratio,
                resnet["width_per_group"] * 16 // slowfast["beta_inv"],
            ],
            dim_out=[
                resnet["width_per_group"] * 32,
                resnet["width_per_group"] * 32 // slowfast["beta_inv"],
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // slowfast["beta_inv"]],
            temp_kernel_sizes=temp_kernel[4],
            stride=resnet["spatial_strides"][3],
            num_blocks=[d5] * 2,
            num_groups=[resnet["num_groups"]] * 2,
            num_block_temp_kernel=resnet["n_block_temp_kernel"][3],
            nonlocal_inds=non_local["loc"][3],
            nonlocal_group=non_local["group"][3],
            nonlocal_pool=non_local["pool"][3],
            instantiation=non_local["instantiation"],
            trans_func_name=resnet["trans_func"],
            dilation=resnet["spatial_dilations"][3],
            norm_module=self.norm_module,
            train=self.training,
        )

        self.head = ResNetBasicHead(
            dim_in=[
                resnet["width_per_group"] * 32,
                resnet["width_per_group"] * 32 // slowfast["beta_inv"],
            ],
            num_classes=mdl["num_classes"],
            pool_size=[
                [
                    data["num_frames"]
                    // slowfast["alpha"]
                    // pool_size[0][0],
                    data["crop_sz"] // 32 // pool_size[0][1],
                    data["crop_sz"] // 32 // pool_size[0][2],
                ],
                [
                    data["num_frames"] // pool_size[1][0],
                    data["crop_sz"] // 32 // pool_size[1][1],
                    data["crop_sz"] // 32 // pool_size[1][2],
                ],
            ],
            dropout_rate=mdl["dropout"],
            act_func=mdl["head_act"],
            train=self.training,
        )

    def forward(self, inp, bboxes=None):
        x = self.s1(inp)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x

def slowfast_4x16(num_classes, train, range=(0,255), depth=50, num_frames=32, **kwargs):
    resnet = {
        "depth": depth,
        "width_per_group": 64,
        "num_groups": 1,
        "trans_func": "bottleneck_transform",
        "n_block_temp_kernel": [[3,3], [4,4], [6,6], [3,3]],
        "spatial_strides": [[1,1],[2,2],[2,2],[2,2]],
        "spatial_dilations":[[1,1],[1,1],[1,1],[1,1]]
    }
    slowfast = {
        "alpha": 8,
        "beta_inv": 8,
        "fusion_conv_ch_ratio": 2,
        "fusion_kernel_sz": 5
    }
    non_local = {
        "loc": [[[], []], [[], []], [[], []], [[], []]],
        "group": [[1, 1], [1, 1], [1, 1], [1, 1]],
        "instantiation": "dot_product",
        "pool": [
            # Res2
            [[1, 2, 2], [1, 2, 2]],
            # Res3
            [[1, 2, 2], [1, 2, 2]],
            # Res4
            [[1, 2, 2], [1, 2, 2]],
            # Res5
            [[1, 2, 2], [1, 2, 2]],
        ]
    }
    data = {
        "in_channels": [3,3],
        "num_frames": num_frames,
        "crop_sz": 256,
        "range": range
    }
    mdl = {
        "num_classes": num_classes,
        "head_act": None,
        "dropout": 0.5
    }
    return SlowFast(
        resnet=resnet,
        slowfast=slowfast,
        non_local=non_local,
        data=data,
        mdl=mdl
    )
