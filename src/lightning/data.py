from tqdm import tqdm

import torch
import lightning.pytorch as pl
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader

from src.utils.augment import build_augmentor
from src.data.matterport_dataset import MatterportDataset
from src.data.single_scene_batchsampler import SingleSceneBatchSampler


class MultiSceneDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()

        # Collect all necessary config parameters
        self.train_data_source = config.DATASET.TRAIN_DATA_SOURCE
        self.val_data_source = config.DATASET.VAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE

        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_pose_root = config.DATASET.TRAIN_POSE_ROOT
        self.train_intrinsics_path = config.DATASET.TRAIN_INTRINSICS_PATH
        self.train_scene_list_path = config.DATASET.TRAIN_SCENE_LIST

        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_pose_root = config.DATASET.VAL_POSE_ROOT
        self.val_intrinsics_path = config.DATASET.VAL_INTRINSICS_PATH
        self.val_scene_list_path = config.DATASET.VAL_SCENE_LIST

        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_pose_root = config.DATASET.TEST_POSE_ROOT
        self.test_intrinsics_path = config.DATASET.TEST_INTRINSICS_PATH
        self.test_scene_list_path = config.DATASET.TEST_SCENE_LIST

        self.augment_type = config.DATASET.AUGMENTATION_TYPE
        self.shuffle = config.DATASET.TRAIN_SHUFFLE

        self.matterport_config = {
            "horizontal_only": config.DATASET.MATTERPORT_HORIZONTAL_IMGS_ONLY,
            "normalize": config.DATASET.MATTERPORT_NORMALIZE,
            "img_height": config.DATASET.IMG_HEIGHT,
            "img_width": config.DATASET.IMG_WIDTH,
        }
        self.batch_sampler_config = {"use_overlaps": config.DATASET.USE_OVERLAPS}

        # Collect all necessary args parameters
        self.batch_size = args.batch_size
        self.train_loader_params = {
            "num_workers": args.num_workers,
            "pin_memory": getattr(args, "pin_memory", True),
        }
        self.val_loader_params = {
            "num_workers": args.num_workers,
            "pin_memory": getattr(args, "pin_memory", True),
        }
        self.test_loader_params = {
            "num_workers": args.num_workers,
            "pin_memory": True,
        }
        self.disable_tqdm = args.tqdm_refresh_rate != 1

    def setup(self, stage: str = None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ["fit", "test"], "stage must be either fit or test"

        if stage == "fit":
            self.augment_fn = build_augmentor(
                self.augment_type, data_source=self.train_data_source
            )

            self.train_dataset = self._setup_dataset(
                self.train_data_source,
                self.train_data_root,
                self.train_scene_list_path,
                self.train_intrinsics_path,
                mode="train",
                pose_dir=self.train_pose_root,
                augment_fn=self.augment_fn,
            )

            self.val_dataset = self._setup_dataset(
                self.val_data_source,
                self.val_data_root,
                self.val_scene_list_path,
                self.val_intrinsics_path,
                mode="val",
                pose_dir=self.val_pose_root,
                augment_fn=None,
            )
            logger.info("Train & Val Dataset loaded!")

        elif stage == "test":
            self.test_dataset = self._setup_dataset(
                self.test_data_source,
                self.test_data_root,
                self.test_scene_list_path,
                self.test_intrinsics_path,
                mode="test",
                pose_dir=self.test_pose_root,
                augment_fn=None,
            )
            logger.info("Test Dataset loaded!")

        else:
            raise NotImplementedError(f"Stage must be either fit or test, was {stage}")

    def _setup_dataset(
        self,
        data_source,
        data_root,
        scene_list_path,
        intrinsics_path,
        mode="train",
        pose_dir=None,
        augment_fn=None,
    ):
        """To make it a bit easier to set up different datasets"""

        with open(scene_list_path, "r") as f:
            scene_names = [name.split()[0] for name in f.readlines()]

        logger.info(f"{len(scene_names)} scene(s) assigned.")

        if self.disable_tqdm and int(self.rank) == 0:
            print(f"Loading {mode} datasets")

        datasets = []
        for scene_name in tqdm(
            scene_names,
            desc=f"Loading {mode} datasets",
            disable=self.disable_tqdm,
        ):
            if data_source == "Matterport":
                datasets.append(
                    MatterportDataset(
                        data_root,
                        scene_name,
                        intrinsics_path,
                        mode=mode,
                        pose_dir=pose_dir,
                        augment_fn=augment_fn,
                        **self.matterport_config,
                    )
                )
            else:
                raise NotImplementedError(f"Data source {data_source} not implemented")

        if self.disable_tqdm and int(self.rank) == 0:
            print(f"Finished loading {mode} datasets.")

        return ConcatDataset(datasets)

    def train_dataloader(self):
        batch_sampler = SingleSceneBatchSampler(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **self.batch_sampler_config,
        )
        dataloader = DataLoader(
            self.train_dataset, batch_sampler=batch_sampler, **self.train_loader_params
        )

        return dataloader

    def val_dataloader(self):
        batch_sampler = SingleSceneBatchSampler(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self.batch_sampler_config,
        )
        dataloader = DataLoader(
            self.val_dataset, batch_sampler=batch_sampler, **self.val_loader_params
        )

        return dataloader

    def test_dataloader(self):
        batch_sampler = SingleSceneBatchSampler(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self.batch_sampler_config,
        )
        dataloader = DataLoader(
            self.test_dataset, batch_sampler=batch_sampler, **self.test_loader_params
        )
        return dataloader
