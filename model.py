import os

import torch
from torch import nn
import vision_transformer as vits


class AtmaModel(nn.Module):
    def __init__(
        self,
        backbone,
        in_features: int=384,
        hidden_dim: int=384,
        out_dim: int=1
    ):
        super(AtmaModel, self).__init__()
        self.backbone = backbone
        self.l1 = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
            )
        self.l2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
        self.l3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


def create_model(args):
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    model.to(args.device)

    if os.path.isfile(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=args.device)
        state_dict = state_dict['teacher']

    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(args.checkpoint, msg))

    a_model = AtmaModel(model).to(args.device)
    return a_model


def freeze_backbone_params(model):
    for param in model.parameters():
        param.requires_grad = True
    for param in model.backbone.parameters():
        param.requires_grad = False