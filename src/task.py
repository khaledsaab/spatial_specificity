import os
from typing import List, Mapping, Sequence, Union

import meerkat as mk
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb
from meerkat.nn import ClassificationOutputColumn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics.functional import dice_score
from torchvision import transforms as transforms
from torchvision.models.segmentation import fcn_resnet50

from src.modeling import ResNet
from src.utils import PredLogger, compute_dice_score, get_save_dir, dictconfig_to_dict, DOMINO_DIR


ROOT_DIR = "/home/ksaab/Documents/domino/domino/data/cxr"

CXR_SIZE = 512


class Classifier(pl.LightningModule):
    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config
        self.segmentation = (
            "seg" in config["train"]["method"]
            or config["train"]["method"] == "detection"
        )
        self.seg_resolution = config["train"]["seg_resolution"]
        self.model_type = config["train"]["model_type"]
        print(f"Model type: {self.model_type}")

        self._set_model()

        metrics = self.config.get("metrics", ["auroc", "accuracy"])

        self.targets = self.config["dataset"]["target_cols"]
        self.loss = nn.CrossEntropyLoss()
        self.metrics = self._get_metrics(
            metrics, num_classes=config["dataset"]["num_classes"]
        )

        self.valid_preds = PredLogger()

    def _get_metrics(self, metrics: List[str], num_classes: int = None):
        num_classes = (
            self.config["dataset"]["num_classes"]
            if num_classes is None
            else num_classes
        )
        _metrics = {
            "accuracy": torchmetrics.Accuracy(compute_on_step=False),
            "auroc": torchmetrics.AUROC(compute_on_step=False),
            "macro_f1": torchmetrics.F1Score(num_classes=num_classes, average="macro"),
            "macro_recall": torchmetrics.Recall(
                num_classes=num_classes, average="macro"
            ),
        }
        return nn.ModuleDict(
            {name: metric for name, metric in _metrics.items() if name in metrics}
        )  # metrics need to be child module of the model, https://pytorch-lightning.readthedocs.io/en/stable/metrics.html#metrics-and-devices

    def _set_model(self):
        model_cfg = self.config["model"]
        num_classes = self.config["dataset"]["num_classes"]

        if self.segmentation:
            if self.model_type == "resnet50":
                self.model = fcn_resnet50(
                    pretrained=False,
                    num_classes=num_classes,
                )

            elif self.model_type == "resunet":

                import segmentation_models_pytorch as smp

                self.model = smp.Unet(
                    "se_resnext50_32x4d",
                    encoder_weights="imagenet",
                    activation=None,
                    # segmentation_head=True,
                    in_channels=3,
                    classes=num_classes,
                )
                # breakpoint()
                self.fc = self.model.segmentation_head
                self.model.segmentation_head = nn.Identity()
            else:
                raise ValueError(f"{self.model_type} not implemented.")

        else:
            self.model = ResNet(
                num_classes=num_classes,
                arch=model_cfg["arch"],
                dropout=model_cfg["dropout"],
                pretrained=model_cfg["pretrained"],
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        if self.segmentation:
            inputs, targets = batch["input"]
            if self.config["dataset"]["source"] != "isic":
                targets = (targets / 255).long()
            else:
                targets = batch["segmentation_target"].long()
        else:
            if self.config["dataset"]["source"] != "isic":
                inputs = batch["input"]
            else:
                inputs, _ = batch["input"]
            targets = batch["target"].long()

        outs = self.forward(inputs)
        if self.segmentation:
            if self.model_type == "resnet50":
                outs = outs["out"]

            embs = outs  # ["out"]
            outs = self.fc(embs)
            dice = dice_score(outs, targets)
            # dice = compute_dice_score(outs, targets)
            self.log("train_dice", dice)

            

        loss = self.loss(outs, targets)

        self.log("train_loss", loss) 

        return loss

    def validation_step(self, batch, batch_idx):
        sample_id = batch["id"]

        if self.segmentation:
            inputs, targets = batch["input"]
            if self.config["dataset"]["source"] != "isic":
                targets = (targets / 255).long()
            else:
                targets = batch["segmentation_target"].long()
            binary_targets = batch["target"]
        else:
            targets = batch["target"].long()
            if self.config["dataset"]["source"] != "isic":
                inputs = batch["input"]
            else:
                inputs, _ = batch["input"]

        outs = self.forward(inputs)

        if self.segmentation:
            if self.model_type == "resnet50":
                outs = outs["out"]
            outs = self.fc(outs)
            dice = compute_dice_score(outs, targets)

            self.log("valid_dice", dice)

        loss = self.loss(outs, targets)
        self.log("valid_loss", loss)  # , sync_dist=True)

        if self.segmentation:  # do rest of metrics as if binary classification
            # flatten last 2 dim so outs = (batch size, num classes, HxW)
            outs = outs.view(outs.shape[0], outs.shape[1], -1).mean(-1)
            targets = binary_targets

        for metric in self.metrics.values():
            metric(torch.softmax(outs, dim=-1)[:, 1], targets)

        self.valid_preds.update(torch.softmax(outs, dim=-1), targets, sample_id)

    def validation_epoch_end(self, outputs) -> None:
        for metric_name, metric in self.metrics.items():
            self.log(f"valid_{metric_name}", metric.compute())  # , sync_dist=True)
            metric.reset()

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        train_cfg = self.config["train"]
        optimizer = torch.optim.Adam(
            self.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["wd"]
        )
        return optimizer


def train_model(
    dp: mk.DataPanel,
    input_column: str,
    target_column: Union[Sequence[str], str],
    id_column: str,
    model: Classifier = None,
    config: dict = None,
    wandb_config: dict = None,
    num_classes: int = 2,
    max_epochs: int = 50,
    samples_per_epoch: int = None,
    gpus: int = 1,  # Union[int, Iterable] = [0],
    num_workers: int = 10,
    batch_size: int = 16,
    train_split: str = "train",
    valid_split: str = "valid",
    segmentation_key: str = "segmentation_target",
    **kwargs,
):
    # Note from https://pytorch-lightning.readthedocs.io/en/0.8.3/multi_gpu.html: Make sure to set the random seed so that each model initializes with the same weights.
    pl.utilities.seed.seed_everything(config["train"]["seed"])

    config["dataset"]["target_cols"] = target_column

    segmentation = (
        "seg" in config["train"]["method"] or config["train"]["method"] == "detection"
    )

    train_mask = np.array(dp["split"].data == train_split)
    val_mask = np.array(dp["split"].data == valid_split)

    dp = mk.DataPanel.from_batch(
        {
            "input": dp[input_column],
            "target": dp[target_column],
            "id": dp[id_column],
            "segmentation_target": dp[segmentation_key],
        }
        if segmentation
        else {
            "input": dp[input_column],
            "target": dp[target_column],
            "id": dp[id_column],
        }
    )
    ckpt_metric = "valid_auroc"
    if config["train"]["method"] == "seg":
        ckpt_metric = "valid_dice"

    train_dp = dp.lz[train_mask]
    val_dp = dp.lz[val_mask]

    if config["dataset"]["source"] != "isic":
        # undersample train set to balance classes
        undersample_mask = (train_dp["target"] == 0) & (
            np.random.rand(len(train_dp)) < 0.33
        )
        undersample_mask = (undersample_mask) | (train_dp["target"] == 1)
        train_dp = train_dp.lz[undersample_mask]

    if config["dataset"]["sample_ratio"] < 1:
        train_dp = train_dp.lz[
            np.random.rand(len(train_dp)) < config["dataset"]["sample_ratio"]
        ]

    if (model is not None) and (config is not None):
        raise ValueError("Cannot pass both `model` and `config`.")

    if model is None:
        config = {} if config is None else config
        config["dataset"]["num_classes"] = num_classes
        if config["model"]["resume_ckpt"]:
            model = Classifier.load_from_checkpoint(
                checkpoint_path=config["model"]["resume_ckpt"], config=config
            )
        else:
            model = Classifier(config)

    save_dir = get_save_dir(config)
    config = dict(config)
    config.update(wandb_config)
    logger = WandbLogger(
        config=dictconfig_to_dict(config),
        config_exclude_keys="wandb",
        save_dir=save_dir,
        **config["wandb"],
    )

    model.train()

    mode = "max"
    checkpoint_callbacks = [ModelCheckpoint(
        monitor=ckpt_metric, mode=mode, every_n_train_steps=5
    )]
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        logger=logger,
        callbacks=checkpoint_callbacks,
        default_root_dir=save_dir,
        **kwargs,
    )

    sampler = None

    if samples_per_epoch is not None:
        sampler = RandomSampler(train_dp, num_samples=samples_per_epoch)

    train_dl = DataLoader(
        train_dp,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
    )
    valid_dl = DataLoader(
        val_dp,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    trainer.fit(model, train_dl, valid_dl)
    wandb.finish()
    return model


def score(
    model: nn.Module,
    dp: mk.DataPanel,
    mimic_model: nn.Module = None,
    layers: Mapping[str, nn.Module] = None,
    reduction_fns: Sequence[callable] = None,
    input_column: str = "input",
    pbar: bool = True,
    device: int = 0,
    segmentation: bool = False,
    return_segmentations: bool = False,
    **kwargs,
):
    model.to(device).eval()
    if mimic_model:
        mimic_model.to(device).eval()

    class ActivationExtractor:
        """Class for extracting activations a targetted intermediate layer"""

        def __init__(self, reduction_fn: callable = None):
            self.activation = None
            self.reduction_fn = reduction_fn

        def add_hook(self, module, input, output):
            if self.reduction_fn is not None:
                output = self.reduction_fn(output)
            self.activation = output

    layer_to_extractor = {}

    if layers is not None:
        for name, layer in layers.items():
            if reduction_fns is not None:
                for reduction_fn in reduction_fns:
                    extractor = ActivationExtractor(reduction_fn=reduction_fn)
                    layer.register_forward_hook(extractor.add_hook)
                    layer_to_extractor[name] = extractor
                    # layer_to_extractor[f"{name}_{reduction_fn.__name__}"] = extractor
            else:
                extractor = ActivationExtractor()
                layer.register_forward_hook(extractor.add_hook)
                layer_to_extractor[name] = extractor

    @torch.no_grad()
    def _score(batch: mk.DataPanel):
        # x = batch[input_column].data.to(device)
        x = torch.stack([entry[0].data.to(device) for entry in batch["input"]])
        out = model(x)  # Run forward pass
        if segmentation:
            seg_out = model.fc(out)
            seg_out_ = seg_out.softmax(1)
            # flatten last 2 dim so outs = (batch size, num classes, HxW)
            # out, _ = seg_out_.view(seg_out.shape[0], seg_out.shape[1], -1).max(-1)
            out = seg_out_.view(seg_out.shape[0], seg_out.shape[1], -1).mean(-1)

        if return_segmentations:
            return {
                "output": ClassificationOutputColumn(
                    logits=out.cpu(), multi_label=False
                ),
                "seg_output": ClassificationOutputColumn(
                    logits=seg_out.cpu(), multi_label=False
                ),
                **{
                    name: extractor.activation.cpu()
                    for name, extractor in layer_to_extractor.items()
                },
            }

        return {
            "output": ClassificationOutputColumn(logits=out.cpu(), multi_label=False),
            **{
                name: extractor.activation.cpu()
                for name, extractor in layer_to_extractor.items()
            },
        }

    dp = dp.update(
        function=_score,
        is_batched_fn=True,
        pbar=pbar,
        input_columns=[input_column],
        **kwargs,
    )
    return dp
