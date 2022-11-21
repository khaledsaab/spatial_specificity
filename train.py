import hydra
import meerkat as mk

# import terra
from omegaconf import DictConfig, OmegaConf

from src.data.cxr import build_cxr_dp
from src.data.isic import build_isic_dp
from src.task import train_model


def train(
    cfg: DictConfig,
):

    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    segmentation = "seg" in cfg["train"]["method"]
    detection = cfg["train"]["method"] == "detection"
    segmentation_key = "segmentation_target"
   
    if dataset_cfg["source"] == "cxr_p":
        dp = build_cxr_dp(
            root_dir="/media/nvme_data/siim",
            segmentation=segmentation,
            augmentation=dataset_cfg["augmentation"],
            detection=detection,
        )
        num_classes = 2
        target_column = "target"
       

    elif dataset_cfg["source"] == "isic":
        dp = build_isic_dp(seed=train_cfg["seed"])
        num_classes = 3 if segmentation else 2
        target_column = "target"

    train_model(
        dp=dp,
        input_column=dataset_cfg["input_column"],
        id_column=dataset_cfg["id_column"],
        target_column=target_column,
        batch_size=train_cfg["batch_size"],
        num_workers=dataset_cfg["num_workers"],
        valid_split=train_cfg["valid_split"],
        max_epochs=train_cfg["epochs"],
        num_classes=num_classes,
        segmentation_key=segmentation_key,
        wandb_config={},
        config=cfg,
    )


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(cfg: OmegaConf):
    # We want to add fields to cfg so need to call OmegaConf.set_struct
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    train(cfg)


if __name__ == "__main__":
    main()
