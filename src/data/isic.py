import os

import imageio
import meerkat as mk
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image


def build_isic_dp(data_dir: str = "/media/nvme_data/isic", seed: int = 1):

    # load the train set
    labels_pth = os.path.join(data_dir, f"trap-sets/isic_annotated_train{seed}.csv")
    labels_df = pd.read_csv(labels_pth)
    labels_df = labels_df.rename(columns={"label": "target"})
    dp = mk.DataPanel.from_pandas(labels_df)
    dp["split"] = ["train"] * len(dp)

    # load the val set
    labels_pth = os.path.join(data_dir, f"trap-sets/isic_annotated_val{seed}.csv")
    labels_df = pd.read_csv(labels_pth)
    labels_df = labels_df.rename(columns={"label": "target"})
    dp_val = mk.DataPanel.from_pandas(labels_df)
    dp_val["split"] = ["val"] * len(dp_val)

    dp = dp.append(dp_val) 

    # load the test set
    labels_pth = os.path.join(data_dir, f"trap-sets/isic_annotated_test{seed}.csv")
    labels_df = pd.read_csv(labels_pth)
    labels_df = labels_df.rename(columns={"label": "target"})
    dp_test = mk.DataPanel.from_pandas(labels_df)
    dp_test["split"] = ["test"] * len(dp_test)

    dp = dp.append(dp_test)

    dp["id"] = dp["image"].to_lambda(fn=lambda x: x.split(".")[0])

    # set up images and seg masks
    dp["input"] = dp["id"].to_lambda(
        fn=lambda x: (
            img_loader(
                os.path.join(data_dir, f"ISIC2018_Task1-2_Training_Input/{x}.jpg")
            ),
            -1,
        )
    )

    dp["segmentation_target"] = dp[["id", "target"]].to_lambda(
        fn=lambda x: seg_loader(x, data_dir)
    )
    
    return dp


def img_loader(filepath, seg=False):
    if seg:
        mean = 0
        std = 1
    else:
        mean = [0.6045568, 0.60814506, 0.6078789]
        std = [0.18579379, 0.1863619, 0.19625686]

    img = imageio.imread(filepath)
    img = Image.fromarray(np.uint8(img))
    img = transforms.Compose(
        [
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )(img)

    return img.squeeze()


def seg_loader(x, data_dir):
    x_id = x["id"]
    target = x["target"]

    filepath = os.path.join(
        data_dir, f"ISIC2018_Task1_Training_GroundTruth/{x_id}_segmentation.png"
    )

    seg_mask = (img_loader(filepath, seg=True) > 0.5).long()
    seg_mask = (target + 1) * seg_mask
    return seg_mask
