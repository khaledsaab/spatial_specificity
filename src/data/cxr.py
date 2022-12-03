import os
import pickle
from functools import partial
from glob import glob

import albumentations as albu
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from albumentations.pytorch.transforms import ToTensorV2
from dosma import DicomReader
from meerkat import DataPanel
from PIL import Image


ROOT_DIR = "/home/ksaab/Documents/spatial_specificity/src/data"
CXR_MEAN = 0.48865
CXR_STD = 0.24621
CXR_SIZE = 512  
CROP_SIZE = 512  

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes



def cxr_pil_loader(input_dict, numpy=False):
    filepath = input_dict["filepath"]
    loader = DicomReader(group_by=None, default_ornt=("SI", "AP"))
    volume = loader(filepath)[0]
    array = volume._volume.squeeze()
    if numpy:
        return np.uint8(array)
    return Image.fromarray(np.uint8(array))


def cxr_loader(input_dict, augmentation):

    train = input_dict["split"] == "train"
    # loader = DicomReader(group_by=None, default_ornt=("SI", "AP"))
    # volume = loader(filepath)

    if not augmentation:
        img = cxr_pil_loader(input_dict)
        img = transforms.Compose(
            [
                transforms.Resize([CROP_SIZE, CROP_SIZE]),
                transforms.ToTensor(),
                transforms.Normalize(CXR_MEAN, CXR_STD),
            ]
        )(img)
        return img.repeat([3, 1, 1])
    else:
        img = cxr_pil_loader(input_dict, numpy=True)

        if train:
            transform_pth = os.path.join(ROOT_DIR, "train_transform.json")
        else:
            transform_pth = os.path.join(ROOT_DIR, "val_transform.json")

        transform = albu.load(transform_pth)
        sample = transform(image=img)
        sample = ToTensorV2()(image=sample["image"])
        img = sample["image"]

        return img.float().squeeze().repeat([3, 1, 1])


def seg_loader(input_dict, augmentation):

    train = input_dict["split"] == "train"
    seg = input_dict["segmentation_target"]

    if augmentation:
        img = cxr_pil_loader(input_dict, numpy=True)
        sample = {"image": img, "mask": seg}

        if train:
            transform_pth = os.path.join(ROOT_DIR, "train_transform.json")
        else:
            transform_pth = os.path.join(ROOT_DIR, "val_transform.json")

        transform = albu.load(transform_pth)
        sample = transform(**sample)
        sample = ToTensorV2()(**sample)
        img, seg = sample["image"], sample["mask"]

        return (img.float().squeeze().repeat([3, 1, 1]), seg)

    img = cxr_pil_loader(input_dict)
    transform = transforms.Compose(
        [
            transforms.Resize([CXR_SIZE, CXR_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(CXR_MEAN, CXR_STD),
        ]
    )
    img = transform(img).repeat([3, 1, 1]).float()

    return (img, torch.Tensor(seg))


def segmask_loader(rle):

    seg = rle2mask(rle, 1024, 1024).T if rle != "-1" else np.zeros((1024, 1024))

    seg_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            torch.nn.MaxPool2d(int(1024 / CXR_SIZE)),
            transforms.ToPILImage(),
            transforms.Resize(CXR_SIZE),
            transforms.ToTensor(),
        ]
    )
    seg = 255 * (seg_transform(seg).squeeze() > 0).float()

    return seg.numpy()


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def detection_segmask_loader(rle):
    seg_mask = segmask_loader(rle)
    if seg_mask.sum() > 0:
        bbox = masks_to_boxes(torch.Tensor(seg_mask).unsqueeze(0))
        assert len(bbox) == 1
        x1, y1, x2, y2 = bbox[0]
        seg_mask = np.zeros(seg_mask.shape)
        seg_mask[int(y1) : int(y2), int(x1) : int(x2)] = 255

    return seg_mask


def get_dp(
    df: pd.DataFrame,
    segmentation: bool,
    augmentation: bool = True,
    detection: bool = False,
):
    dp = DataPanel.from_pandas(df)

    if segmentation:
        dp["segmentation_target"] = dp["encoded_pixels"].to_lambda(fn=segmask_loader)

        dp["input"] = dp[["filepath", "split", "segmentation_target"]].to_lambda(
            fn=partial(seg_loader, augmentation=augmentation)
        )
    elif detection:
        dp["segmentation_target"] = dp["encoded_pixels"].to_lambda(
            fn=detection_segmask_loader
        )
        dp["input"] = dp[["filepath", "split", "segmentation_target"]].to_lambda(
            fn=partial(seg_loader, augmentation=augmentation)
        )

    else:
        dp["input"] = dp[["filepath", "split"]].to_lambda(
            fn=partial(cxr_loader, augmentation=augmentation)
        )

    dp["img"] = dp[["filepath"]].to_lambda(fn=cxr_pil_loader)

    return dp


def build_cxr_dp(
    root_dir: str = ROOT_DIR,
    segmentation: bool = False,
    augmentation: bool = True,
    detection: bool = False,
):
    # get segment annotations
    segment_df = pd.read_csv(os.path.join(root_dir, "train-rle.csv"))
    segment_df = segment_df.rename(
        columns={"ImageId": "id", " EncodedPixels": "encoded_pixels"}
    )
    # there are some image ids with multiple label rows, we'll just take the first
    segment_df = segment_df[~segment_df.id.duplicated(keep="first")]

    # get binary labels for pneumothorax, any row with a "-1" for encoded pixels is
    # considered a negative
    segment_df["target"] = (segment_df.encoded_pixels != "-1").astype(int)

    # start building up a main dataframe with a few `merge` operations (i.e. join)
    df = segment_df

    # get filepaths for all images in the "dicom-images-train" directory
    filepaths = sorted(glob(os.path.join(root_dir, "dicom-images-train/*/*/*.dcm")))
    filepath_df = pd.DataFrame(
        [
            {
                "filepath": filepath,
                "id": os.path.splitext(os.path.basename(filepath))[0],
            }
            for filepath in filepaths
        ]
    )

    # important to perform a left join here, because there are some images in the
    # directory without labels in `segment_df`
    df = df.merge(filepath_df, how="left", on="id")

    # add in chest tube annotations
    rows = []
    for split in ["train", "test"]:
        tube_dict = pickle.load(
            open(
                os.path.join(root_dir, f"cxr_tube_labels/cxr_tube_dict_{split}.pkl"),
                "rb",
            )
        )
        rows.extend(
            [
                {"id": k, "chest_tube": int(v), "split": split}
                for k, v in tube_dict.items()
            ]
        )
    tube_df = pd.DataFrame(rows)

    df = df.merge(tube_df, how="left", on="id")
    df.split = df.split.fillna("train")

    dp = get_dp(
        df,
        segmentation=segmentation,
        augmentation=augmentation,
        detection=detection,
    )

    tube_mask = np.array(dp["chest_tube"].data.astype(str) != "nan")
    val_mask = (tube_mask) & ~(dp["split"] == "test")
    dp["split"][val_mask] = "val"

    return dp
