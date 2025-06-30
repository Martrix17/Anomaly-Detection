import torchvision.models as models
from backbones.backbone_wrapper import BackboneWrapper

_BACKBONES = {
    "resnet18": models.resnet18(weights="DEFAULT"),
    "resnet34": models.resnet34(weights="DEFAULT"),
    "resnet50": models.resnet50(weights="DEFAULT"),
    "resnet101": models.resnet101(weights="DEFAULT"),
    "resnet152": models.resnet152(weights="DEFAULT"),
    "mobilenet_v2": models.mobilenet_v2(weights="DEFAULT"),
    "mobilenet_v3_large": models.mobilenet_v3_large(weights="DEFAULT"),
    "mobilenet_v3_small": models.mobilenet_v3_small(weights="DEFAULT"),
    "efficientnet_b0": models.efficientnet_b0(weights="DEFAULT"),
    "efficientnet_b1": models.efficientnet_b1(weights="DEFAULT"),
    "efficientnet_b2": models.efficientnet_b2(weights="DEFAULT"),
    "efficientnet_b3": models.efficientnet_b3(weights="DEFAULT"),
}


def load_backbone(name, latent_dim, image_size):
    backbone = _BACKBONES[name]
    return BackboneWrapper(backbone, latent_dim, image_size)
