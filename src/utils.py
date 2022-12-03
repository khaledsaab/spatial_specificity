from omegaconf import DictConfig
import torch
from typing import List, Mapping, Sequence, Union
from torchmetrics import Metric
import numpy as np
import os
import meerkat as mk

ROOT_DIR = "/home/ksaab/Documents/spatial_specificity"


class PredLogger(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state(
            "sample_ids",
            default=[],
            dist_reduce_fx=None,
        )

    def update(
        self, pred: torch.Tensor, target: torch.Tensor, sample_id: Union[str, int]
    ):
        self.preds.append(pred.detach())
        self.targets.append(target.detach())
        self.sample_ids.extend(sample_id)

    def compute(self):
        """TODO: this sometimes returns duplicates."""
        if torch.is_tensor(self.sample_ids[0]):
            sample_ids = torch.tensor(self.sample_ids).cpu()
        else:
            # support for string ids
            sample_ids = self.sample_ids
        preds = torch.cat(self.preds).cpu()
        targets = torch.cat(self.targets).cpu()
        self.preds, self.targets, self.sample_ids = [], [], []

        return mk.DataPanel({"preds": preds, "targets": targets, "ids": sample_ids})

    def _apply(self, fn):
        """
        https://github.com/PyTorchLightning/metrics/blob/fb0ee3ff0509fdb13bd07b6aac3e20c642bb5683/torchmetrics/metric.py#L280
        """
        this = super(Metric, self)._apply(fn)
        # Also apply fn to metric states
        for key in this._defaults.keys():
            current_val = getattr(this, key)
            if isinstance(current_val, torch.Tensor):
                setattr(this, key, fn(current_val))
            elif isinstance(current_val, Sequence):
                if (
                    len(current_val) > 0 and isinstance(current_val[0], tuple)
                ) or key == "sample_ids":
                    # avoid calling `.to`, `.cpu`, `.cuda` on string metric states
                    continue

                setattr(this, key, [fn(cur_v) for cur_v in current_val])
            else:
                raise TypeError(
                    "Expected metric state to be either a Tensor"
                    f"or a list of Tensor, but encountered {current_val}"
                )
        return this

def compute_dice(mask1, mask2):
    overlap = torch.sum(torch.logical_and(mask1, mask2))
    total = torch.sum(mask1) + torch.sum(mask2)
    return 2 * overlap / total


def compute_dice_score(outs, targets):
    preds = outs.argmax(1)
    dice_scores = []
    for (pred, target) in zip(preds, targets):
        dice_scores.append(compute_dice(pred, target).item())

    return np.nanmean(dice_scores)


def get_save_dir(config):
    method = config["train"]["method"]

    lr = config["train"]["lr"]
    wd = config["train"]["wd"]
    dropout = config["model"]["dropout"]
    save_dir = f"{ROOT_DIR}/results/method_{method}/lr_{lr}/wd_{wd}/dropout_{dropout}"


    seed = config["train"]["seed"]
    save_dir += f"/seed_{seed}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def dictconfig_to_dict(d):
    """Convert object of type OmegaConf to dict so Wandb can log properly
    Support nested dictionary.
    """
    return {
        k: dictconfig_to_dict(v) if isinstance(v, DictConfig) else v
        for k, v in d.items()
    }


from torchvision.models import resnet50

from src.task import score
import segmentation_models_pytorch as smp
from meerkat import DataPanel
import torch.nn as nn


def get_activations(
    dp: DataPanel,
    model_pth: str,
    segmentation: bool = False,
    isic = False,
    return_segmentations: bool = False,
    batch_size: int = 32,
    device: int = 0,
):

    if segmentation:
        model = smp.Unet(
                "se_resnext50_32x4d",
                encoder_weights="imagenet",
                activation=None,
                # segmentation_head=True,
                in_channels=3,
                classes=2 if not isic else 3,
            )
        model.fc = model.segmentation_head
        model.segmentation_head = nn.Identity()
        # model = fcn_resnet50(
        #         pretrained=False,
        #         num_classes=2,
        #     )
        # model.fc = model.classifier
        # model.classifier = nn.Identity()
        model_state_dict = torch.load(model_pth)["state_dict"]
        model_state_dict = {
            k.split("model.")[-1]: v
            for k, v in model_state_dict.items()
        }
        model.load_state_dict(model_state_dict)
    else:
        model = resnet50()
        d = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0),nn.Linear(d, 2))
        model_state_dict = torch.load(model_pth)["state_dict"]
        model_state_dict = {
            k.split("model.")[-1]: v
            for k, v in model_state_dict.items()
            if k in model_state_dict
        }
        model.load_state_dict(model_state_dict)
        
    act_dp = score(
        model=model,
        dp=dp,
        device=device,
        segmentation=segmentation,
        return_segmentations=return_segmentations,
        batch_size=batch_size,
    )
    return act_dp