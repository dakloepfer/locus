from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from math import floor

import torch
from torch.nn import functional as F
import lightning.pytorch as pl
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.linalg import transform_points
from kornia.geometry.conversions import convert_points_from_homogeneous
from kornia.utils import create_meshgrid
from einops import rearrange, repeat, reduce

from src.models.dino_feature_extractor import DINO_FeatureExtractor
from src.losses.vec_smooth_ap import VecSmoothAP
from src.utils.misc import batched_2d_index_select
from src.utils.profiler import PassThroughProfiler


class PLModule(pl.LightningModule):
    def __init__(self, config, ckpt_path=None, devices=[], profiler=None):
        super().__init__()
        self.config = config
        self.profiler = profiler or PassThroughProfiler()

        if self.config.MODEL_TYPE == "dino":
            self.model = DINO_FeatureExtractor(self.config.MODEL)
        else:
            raise NotImplementedError(f"Unknown matcher: {self.config.MODEL_TYPE}")

        self.feature_subsample = self.model.output_subsample

        self.loss = VecSmoothAP(self.config.LOSS)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(state_dict["model"])
            logger.info(f"Loaded pretrained weights from {ckpt_path}")

        if len(devices) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=devices)

        self.n_landmarks = config.TRAIN.N_LANDMARKS  # number of landmarks to sample
        self.frac_patch_subsample = config.TRAIN.FRAC_PATCH_SUBSAMPLE
        self.keep_all_positive_patches = config.TRAIN.KEEP_ALL_POSITIVE_PATCHES
        self.pos_radius = config.MODEL.POS_LANDMARK_RADIUS
        self.neg_radius = config.MODEL.NEG_LANDMARK_RADIUS
        self.landmark_embedding_method = config.MODEL.LANDMARK_EMBEDDING_METHOD

    def configure_optimizers(self):
        optim_conf = self.config.OPTIMIZER
        if optim_conf.TYPE == "Adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=optim_conf.LR,
                weight_decay=optim_conf.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(f"Unknown optimizer: {optim_conf.TYPE}")

    def training_step(self, batch, batch_idx):
        return self._trainval_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._trainval_step(batch, batch_idx)

    def _trainval_step(self, batch, batch_idx):
        imgs = batch["img"]
        gt_depths = batch["gt_depth"].unsqueeze(1)
        cam_matrices = batch["intrinsics"]
        world_to_cam_poses = batch["world_to_cam_pose"]
        depths = F.interpolate(
            gt_depths, scale_factor=1 / self.feature_subsample, mode="nearest-exact"
        )

        if imgs.shape[0] <= 1:
            raise ValueError("Batch size must be greater than 1")
        img_height = imgs.shape[2]
        img_width = imgs.shape[3]
        cameras = PinholeCamera(
            cam_matrices,
            world_to_cam_poses,
            torch.ones_like(cam_matrices[:, 0, 0]) * img_height,
            torch.ones_like(cam_matrices[:, 0, 0]) * img_width,
        )

        # cannot use PyTorch Lightning's DDP because the loss function requires us to accumulate all the features from all the GPUs
        with self.profiler.profile("Feature Extraction"):
            features = self.model(imgs)
            features = F.normalize(features, p=2, dim=1)

        with self.profiler.profile("Computing landmarks"):
            landmarks = self._sample_landmarks(cameras, depths, features)
            self._project_landmarks(landmarks, cameras, gt_depths)
            self._filter_landmarks(landmarks)
            self._compute_landmark_embeddings(landmarks, features)

        with self.profiler.profile("Computing positive / negative patches"):
            pos_patches, pos_neg_patches = self._calculate_contrastive_masks(
                landmarks, cameras, depths
            )
            (
                selected_patch_features,
                pos_patches,
                pos_neg_patches,
            ) = self._subsample_patches(pos_patches, pos_neg_patches, features)

        with self.profiler.profile("Computing loss"):
            loss, log_dict = self.loss(
                landmarks, pos_patches, pos_neg_patches, selected_patch_features
            )

        self.log_dict(log_dict)
        return loss

    def _sample_landmarks(self, cameras, depths, features):
        """Sample landmarks from the scene using the network outputs and ground truths.

        Parameters
        ----------
        cameras (PinholeCamera):
            A batch of Kornia PinholeCameras with the intrinsics and exrtrinsics for the batch.

        depths (batch_size x 1 x feature_height x feature_width tensor):
            ground truth depth in meters for each pixel patch

        features (batch_size x n_channels x feature_height x feature_width tensor):
            the features extracted.

        Returns
        -------
        landmarks (dict):
            keys:
                landmark_pos (n_landmarks x 3 tensor):
                    the positions of the selected (sampled) landmarks

                landmark_embeddings (n_landmarks x n_channels tensor):
                    the features of the selected patches
        """
        batch_size = depths.shape[0]
        height = depths.shape[2]
        width = depths.shape[3]

        patch_idxs = torch.randperm(height * width, device=self.device)[
            : self.n_landmarks
        ]
        batch_idxs = repeat(
            torch.arange(batch_size, device=self.device),
            "b -> (b h w)",
            h=height,
            w=width,
        )[patch_idxs]

        landmark_cameras = PinholeCamera(
            cameras.intrinsics[batch_idxs],
            cameras.extrinsics[batch_idxs],
            cameras.height[batch_idxs],
            cameras.width[batch_idxs],
        )

        landmark_pixel_coords = (
            create_meshgrid(
                height, width, normalized_coordinates=False, device=self.device
            )
            + 0.5
        ) * self.feature_subsample
        landmark_pixel_coords = repeat(
            landmark_pixel_coords,
            "() h w c -> (b h w) c",
            b=batch_size,
            h=height,
            w=width,
            c=2,
        )[patch_idxs]

        landmark_depths = rearrange(
            depths, "b () h w -> (b h w) ()", b=batch_size, h=height, w=width
        )[patch_idxs]
        landmark_pos = landmark_cameras.unproject(
            landmark_pixel_coords.unsqueeze(1), landmark_depths.unsqueeze(1)
        ).squeeze(1)

        landmark_embeddings = rearrange(
            features, "b c h w -> (b h w) c", b=batch_size, h=height, w=width
        )[patch_idxs]

        return {
            "landmark_pos": landmark_pos,
            "landmark_embeddings": landmark_embeddings,
        }

    def _project_landmarks(self, landmarks, cameras, gt_depths):
        """Modifies the landmarks dictionary in-place, updating it with the pixel locations, the visibility mask, and the depths for each landmark-image pair.

        Parameters
        ----------
        landmarks (dictionary):):
            minimum keys:
                landmark_pos (n_landmarks x 3 tensor):
                    the positions of the selected (sampled) landmarks in 3D space.

        cameras (PinholeCamera):
            A Kornia PinholeCamera object with the intrinsics and extrinsics for the batch.

        gt_depths (batch_size x 1 x img_height x img_width tensor):
            ground truth depth in meters for each pixel

        Updates
        ------
        landmarks (dictionary):
            new keys:
                landmark_pixel_coords (batch_size x n_landmarks x 2 tensor):
                    the pixel coordinates of the selected (sampled) landmarks in each image

                landmark_visibility_mask (batch_size x n_landmarks x 1 bool tensor):
                    a mask indicating which landmarks are visible whether the landmark is visible in a given image (occluded or out of frame)

                landmark_depths (batch_size x n_landmarks x 1 tensor):
                    the depths of the selected (sampled) landmarks in each image
        """
        batch_size = cameras.batch_size
        landmark_pos = repeat(
            landmarks["landmark_pos"],
            "n_landmarks three -> batch_size n_landmarks three",
            batch_size=batch_size,
        )
        n_landmarks = landmark_pos.shape[1]
        img_height = gt_depths.shape[2]
        img_width = gt_depths.shape[3]

        # need to also get the landmark depths, so can't use the PinholeCamera.project method
        landmark_cam_coords = transform_points(cameras.extrinsics, landmark_pos)
        landmark_depths = landmark_cam_coords[..., 2:]
        landmark_pixel_coords = convert_points_from_homogeneous(
            transform_points(cameras.intrinsics, landmark_cam_coords)
        )

        # occlusion check
        landmark_pixel_idxs = torch.round(landmark_pixel_coords).long()
        landmark_pixel_idxs = torch.clamp(
            landmark_pixel_idxs,
            torch.tensor([0, 0], device=self.device),
            torch.tensor([img_width - 1, img_height - 1], device=self.device),
        )
        gt_landmark_depths = batched_2d_index_select(
            gt_depths, torch.flip(landmark_pixel_idxs, dims=(-1,))
        )

        # add some small tolerance
        landmark_visibility_mask = (
            (landmark_depths < gt_landmark_depths + 0.05)
            & (landmark_pixel_coords[..., 0:1] > 0 - 1e-3)
            & (landmark_pixel_coords[..., 1:2] > 0 - 1e-3)
            & (landmark_pixel_coords[..., 0:1] < img_width + 1e-3)
            & (landmark_pixel_coords[..., 1:2] < img_height + 1e-3)
        )

        landmarks.update(
            {
                "landmark_pixel_coords": landmark_pixel_coords,
                "landmark_visibility_mask": landmark_visibility_mask,
                "landmark_depths": landmark_depths,
            }
        )

    def _filter_landmarks(self, landmarks):
        """Filter out unsuitable landmarks.

        Parameters
        ----------
        landmarks (dict):
            Information about the landmarks.

        Updates
        -------
        landmarks (dict):
            filters out some unsuitable landmarks
        """
        # filter out landmarks that are only visible in one image
        mask = landmarks["landmark_visibility_mask"].sum(dim=0) > 1
        mask = mask.squeeze(-1)

        for key in landmarks.keys():
            if len(landmarks[key].shape) == 2:
                landmarks[key] = landmarks[key][mask]
            elif len(landmarks[key].shape) == 3:
                landmarks[key] = landmarks[key][:, mask]
            else:
                raise ValueError(f"Unexpected shape for {key}: {landmarks[key].shape}")

    def _compute_landmark_embeddings(self, landmarks, features):
        """Modifies the landmarks dictionary in-place, adding the computed landmark embeddings.

        Parameters
        ----------
        landmarks (dict):
            the dictionary containing information about the landmarks.

        features (batch_size x feature_dim x feature_height x feature_width tensor):
            the extracted features

        Updates
        -------
        landmarks (dict):
            landmark_embeddings (n_landmarks x feature_dim tensor):
                the embeddings for the landmarks; should be normalised.
        """
        if self.landmark_embedding_method == "sampled_patch":
            return
        else:
            raise NotImplementedError

    def _calculate_contrastive_masks(self, landmarks, cameras, depths):
        """Calculate the masks for which patches are positive pairs, and which ones are used in as the entire set of patches (all pairs).

        Parameters:
        -----------
        landmarks (dict):
            the dictionary containing information about the landmarks.

        cameras (PinholeCamera):
            A Kornia PinholeCamera object with the intrinsics and extrinsics for the batch.

        depths (batch_size x 1 x feature_height x feature_width tensor):
            ground truth depth in meters for each pixel patch

        Returns:
        --------
        pos_patches: n_landmarks x (batch_size * height * width) bool tensor
            A boolean tensor indicating which patches are positive patches for a given landmark

        pos_neg_patches: n_landmarks x (batch_size * height * width) bool tensor
            A boolean tensor indicating which patches are positive or negative patches for a given landmark
        """
        # first, for every image for every landmark, calculate the rectangular patch within the negative patch radius (the larger of the two)
        # then index all these patches and calculate the 3D locations of only these patches -- this means a lot fewer points to calculate
        # then filter based on distance to the landmarks

        batch_size = cameras.batch_size
        n_landmarks = landmarks["landmark_pos"].shape[0]
        height, width = depths.shape[2:4]

        # calculate rectangular patches that should contain all the patches we might use
        # if neg_radius < pos_radius, use all patches as negative patches --> don't need to calculate 3D locations for them
        if self.neg_radius >= self.pos_radius:
            coarse_patch_window_size = (
                self.neg_radius
                * torch.stack((cameras.fx, cameras.fy), dim=-1).view(batch_size, 1, 2)
                / (landmarks["landmark_depths"] + 1e-8)
            )  # shape batch_size x n_landmarks x 2
        else:
            coarse_patch_window_size = (
                self.pos_radius
                * torch.stack((cameras.fx, cameras.fy), dim=-1).view(batch_size, 1, 2)
                / (landmarks["landmark_depths"] + 1e-8)
            )  # shape batch_size x n_landmarks x 2

        patch_pixel_coords = (
            create_meshgrid(
                height, width, normalized_coordinates=False, device=self.device
            )
            + 0.5
        ) * self.feature_subsample
        patch_pixel_coords = repeat(
            patch_pixel_coords,
            "() h w c -> b n (h w) c",
            b=batch_size,
            n=n_landmarks,
            h=height,
            w=width,
            c=2,
        )

        landmark_pixel_coords = landmarks["landmark_pixel_coords"]

        coarse_patch_mask = (
            (
                patch_pixel_coords[..., 0]
                <= landmark_pixel_coords[..., None, 0]
                + coarse_patch_window_size[..., None, 0]
            )
            & (
                patch_pixel_coords[..., 0]
                >= landmark_pixel_coords[..., None, 0]
                - coarse_patch_window_size[..., None, 0]
            )
            & (
                patch_pixel_coords[..., 1]
                <= landmark_pixel_coords[..., None, 1]
                + coarse_patch_window_size[..., None, 1]
            )
            & (
                patch_pixel_coords[..., 1]
                >= landmark_pixel_coords[..., None, 1]
                - coarse_patch_window_size[..., None, 1]
            )
        )
        coarse_patch_mask = reduce(coarse_patch_mask, "b n hw -> b hw", "sum")

        patch_idxs = torch.nonzero(coarse_patch_mask, as_tuple=True)

        patch_cameras = PinholeCamera(
            cameras.intrinsics[patch_idxs[0]],
            cameras.extrinsics[patch_idxs[0]],
            cameras.height[patch_idxs[0]],
            cameras.width[patch_idxs[0]],
        )
        patch_pixel_coords = patch_pixel_coords[:, 0][patch_idxs].unsqueeze(1)
        patch_depths = rearrange(depths, "b () h w -> b (h w) () ()")[patch_idxs]

        patch_world_pos = patch_cameras.unproject(
            patch_pixel_coords, patch_depths
        ).squeeze(1)

        patch_landmark_dists = torch.linalg.norm(
            patch_world_pos[None] - landmarks["landmark_pos"][:, None], ord=2, dim=-1
        )  # shape n_landmarks x n_patches

        pos_patch_mask = patch_landmark_dists < self.pos_radius
        if self.landmark_embedding_method == "sampled_patch":
            # remove self-pairs
            pos_patch_mask[:, patch_landmark_dists.argmin(dim=1)] = False

        pos_patches = torch.zeros(
            (n_landmarks, batch_size, height * width),
            dtype=torch.bool,
            device=self.device,
        )
        pos_patches[:, patch_idxs[0], patch_idxs[1]] = pos_patch_mask
        pos_patches = rearrange(pos_patches, "n b hw -> n (b hw)")

        if self.neg_radius >= self.pos_radius:
            pos_neg_patch_mask = patch_landmark_dists < self.neg_radius
            if self.landmark_embedding_method == "sampled_patch":
                # remove self-pairs
                pos_neg_patch_mask[:, patch_landmark_dists.argmin(dim=1)] = False

            pos_neg_patches = torch.zeros(
                (n_landmarks, batch_size, height * width),
                dtype=torch.bool,
                device=self.device,
            )
            pos_neg_patches[:, patch_idxs[0], patch_idxs[1]] = pos_neg_patch_mask
            pos_neg_patches = rearrange(pos_neg_patches, "n b hw -> n (b hw)")
        else:
            pos_neg_patches = torch.ones(
                (n_landmarks, batch_size * height * width),
                dtype=torch.bool,
                device=self.device,
            )

        return pos_patches, pos_neg_patches

    def _subsample_patches(self, pos_patches, pos_neg_patches, features):
        """Subsample patches used for computational efficiency. Also collects the features for the patches.

        Parameters
        ----------
        pos_patches: n_landmarks x (batch_size * height * width) bool tensor
            A boolean tensor indicating which patches are positive patches for a given landmark

        pos_neg_patches: n_landmarks x (batch_size * height * width) bool tensor
            A boolean tensor indicating which patches are positive or negative patches for a given landmark

        features (batch_size x n_channels x feature_height x feature_width tensor):
            the features extracted.

        Returns
        -------
        selected_patch_features (n_patches x n_channels tensor):
            the features for the patches that are used

        pos_patches (n_landmarks x n_patches bool tensor):
            A mask for the patches that are used as positive patches

        pos_neg_patches (n_landmarks x n_patches bool tensor):
            A mask for the patches that are used as either positive or negative patches.
            There should be at least one True entry in each row.
        """
        if self.keep_all_positive_patches:
            # make sure all positive patches are always used
            pos_patch_mask = torch.any(pos_patches, dim=0)
        else:
            pos_patch_mask = torch.zeros_like(pos_patches[0])

        n_total_patches = pos_patches.shape[1]
        n_selected_patches = floor(self.frac_patch_subsample * n_total_patches)

        patch_mask = torch.zeros_like(pos_patches[0])
        patch_mask[
            torch.randperm(n_total_patches, device=self.device)[:n_selected_patches]
        ] = True
        patch_mask = patch_mask | pos_patch_mask

        selected_patch_features = rearrange(features, "b c h w -> (b h w) c")[
            patch_mask
        ]
        pos_patches = pos_patches[:, patch_mask]
        pos_neg_patches = pos_neg_patches[:, patch_mask]

        return selected_patch_features, pos_patches, pos_neg_patches
