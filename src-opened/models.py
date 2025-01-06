import timm
import torch.nn as nn

def mobilenet_v2(output=1, ckpt=None):
    model = timm.create_model("mobilenetv2_100", pretrained=True, checkpoint_path=ckpt)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, output)
    return model

def mobilenet_v3(output=1, ckpt=None):
    model = timm.create_model("mobilenetv3_large_100", pretrained=True, checkpoint_path=ckpt)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, output)
    return model