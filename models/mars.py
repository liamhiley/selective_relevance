import torch 

alpha = 50

"""
The MARS/MERS models are simply implementation of the ResNeXt3d models, with altered training strategies, therefore for the model architecture see ./resnet3d.py
"""

def teacher_loss(student, teacher):
    """
    The MERS model is a ResNeXt3d that takes as input RGB frames, but with the goal of reproducing the output feature vector
    of a 'Teacher' Optical Flow model that takes as input the corresponding flow frames for that input.
    Thus, we define the custom MERS loss function as the MSE between the two feature vectors.
    """
    return torch.nn.MSELoss()(student,teacher)

def joint_loss(out, labels, student, teacher):
    """
    The MARS model results from the joint effort of the model to minimise classification loss, as well as MSE with the 'Teacher'
    Flow model.
    """
    return torch.nn.CrossEntropyLoss()(out, labels) + alpha * teacher_loss(student,teacher)

def register_feature_hook(layer, feature_list):
    def feature_hook(self, input, output):
        feature_list[0] = output
    hook = layer.register_forward_hook(feature_hook)
    return hook
