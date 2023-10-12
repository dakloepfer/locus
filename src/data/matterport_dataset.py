import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from einops import rearrange

from src.utils.misc import invert_se3


class MatterportDataset(torch.utils.data.Dataset):
    """Dataset for a single Matterport3D scene."""

    def __init__(
        self,
        data_root,
        scene_name,
        intrinsics_path,
        mode="train",
        pose_dir=None,
        augment_fn=None,
        **kwargs,
    ):
        super().__init__()

        self.data_root = data_root
        self.scene_name = scene_name
        self.intrinsics_path = intrinsics_path
        self.augment_fn = augment_fn if mode == "train" else None

        self.horizontal_only = kwargs.get("horizontal_only", False)

        if kwargs.get("normalize", "imagenet") == "imagenet":
            self.normalize = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )  # for backbones pretrained on ImageNet
        else:
            raise NotImplementedError(
                f"Image normalization {kwargs['normalize']} not implemented."
            )
        self.img_height = kwargs.get("img_height", 1024)
        self.img_width = kwargs.get("img_width", 1024)
        self.downsampling_factor = self.img_height / 1024

        self.img_dir = None
        self.depth_dir = None

        self.data_dict = self._create_data_dict()

    def filter_img_files(self, img_dir):
        """Filter out images that aren't used for various reasons."""
        all_img_files = os.listdir(img_dir)
        used_img_files = []
        used_file_mask = torch.ones(len(all_img_files), dtype=torch.bool)

        for i, file_name in enumerate(all_img_files):
            if self.horizontal_only:
                # remove all the files that aren't looking horizontally
                if not file_name[-7] == "1":
                    used_file_mask[i] = False
                    continue

            used_img_files.append(file_name)

        return used_img_files, used_file_mask

    def _create_data_dict(self):
        data_dict = {}
        scene_dir = os.path.join(self.data_root, self.scene_name)
        camera_parameter_file_path = os.path.join(
            scene_dir,
            "undistorted_camera_parameters",
            "{}.conf".format(self.scene_name),
        )

        with open(camera_parameter_file_path, "r") as f:
            params_file_lines = f.readlines()

        current_intrinsics = None
        if self.img_height >= 448:
            self.depth_dir = os.path.join(scene_dir, "depth_highres")
            self.img_dir = os.path.join(scene_dir, "rgb_highres")
        else:
            self.depth_dir = os.path.join(scene_dir, "depth")
            self.img_dir = os.path.join(scene_dir, "rgb")

        used_img_files, _ = self.filter_img_files(self.img_dir)

        sample_idx = 0
        for line in params_file_lines:
            if line.startswith("intrinsics"):
                intrinsics_line = [
                    i
                    for i in line.strip().split(" ")
                    if not (i.isspace() or len(i) == 0)
                ]

                current_intrinsics = torch.eye(4)
                current_intrinsics[0, 0] = (
                    float(intrinsics_line[1]) * self.downsampling_factor
                )
                current_intrinsics[1, 1] = (
                    float(intrinsics_line[5]) * self.downsampling_factor
                )
                current_intrinsics[0, 2] = (
                    float(intrinsics_line[3]) * self.downsampling_factor
                )
                current_intrinsics[1, 2] = (
                    float(intrinsics_line[6]) * self.downsampling_factor
                )

            elif line.startswith("scan"):
                scan_line = line.split(" ")[1:]
                depth_file_name = scan_line[0].replace(".png", ".npy")
                img_file_name = scan_line[1].replace(".jpg", ".npy")

                # I filter out some images that had too few good depth values to inpaint the depth map
                if img_file_name not in used_img_files:
                    continue

                cam_to_world_pose = torch.tensor(
                    [float(t) for t in scan_line[2:]], dtype=torch.float
                )
                cam_to_world_pose = rearrange(
                    cam_to_world_pose, "(h w) -> h w", h=4, w=4
                )
                world_to_cam_pose = invert_se3(cam_to_world_pose)

                data_dict[sample_idx] = {
                    "img_path": os.path.join(self.img_dir, img_file_name),
                    "depth_path": os.path.join(self.depth_dir, depth_file_name),
                    "world_to_cam_pose": world_to_cam_pose,
                    "intrinsics": current_intrinsics,
                }
                sample_idx += 1

            else:
                continue

        return data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        img_path = self.data_dict[index]["img_path"]
        depth_path = self.data_dict[index]["depth_path"]
        world_to_cam_pose = self.data_dict[index]["world_to_cam_pose"]
        intrinsics = self.data_dict[index]["intrinsics"]

        img = np.load(img_path)
        img = F.interpolate(
            to_tensor(img).unsqueeze(0),
            size=(self.img_height, self.img_width),
        ).squeeze(0)
        img = self.normalize(img)

        if self.augment_fn is not None:
            img = self.augment_fn(img)

        depth = torch.from_numpy(np.load(depth_path)).to(dtype=torch.float)
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0), size=img.shape[-2:]
        ).squeeze()

        sample = {
            "img": img,
            "gt_depth": depth,
            "intrinsics": intrinsics,
            "world_to_cam_pose": world_to_cam_pose,
        }

        return sample
