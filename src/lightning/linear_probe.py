from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from lightning import pytorch as pl
import torchmetrics
from src.data.matterport_segmentation_dataset import (
    N_OBJECTS_PER_SCENE,
    N_STUFF_CLASS_OBJECTS,
)


class LinearProbeModule(pl.LightningModule):
    """LightningModule for linear probe training."""

    def __init__(self, config, scene):
        super().__init__()
        self.config = config
        self.scene = scene

        self.feature_dim = config.MODEL.FEATURE_DIM
        # +1 for invalid object
        self.n_objects = N_OBJECTS_PER_SCENE[scene] + 1
        # have outputs classes for individual "stuff class objects", but don't require the linear probe to re-ID them (see forward() method)
        # (the object segmentations for eg different parts of wall often don't make sense)

        self.n_stuff_objects = N_STUFF_CLASS_OBJECTS[scene]

        # one output per object, not grouping stuff class 'objects'
        self.n_outputs = (
            self.n_objects + sum(self.n_stuff_objects) - len(self.n_stuff_objects)
        )

        self.linear_layer = nn.Conv2d(
            self.feature_dim,
            self.n_outputs,
            kernel_size=1,
            bias=True,
        )

        self.test_results = {}

    def configure_optimizers(self):
        return torch.optim.Adam(self.linear_layer.parameters(), lr=0.01)

    def forward(self, x):
        raw_logits = self.linear_layer(x)

        # the logits for the "stuff class objects" are the maximum of the logits for the individual stuff class 'objects'

        logits = torch.zeros(
            raw_logits.shape[0],
            self.n_objects,
            *raw_logits.shape[2:],
            device=self.device,
        )
        for i in range(len(self.n_stuff_objects)):
            start_idx = sum(self.n_stuff_objects[:i])
            end_idx = sum(self.n_stuff_objects[: i + 1])
            if end_idx == start_idx:
                continue
            stuff_logits, used_idxs = torch.max(raw_logits[:, start_idx:end_idx], dim=1)

            logits[:, i] = stuff_logits

        logits[:, len(self.n_stuff_objects) :] = raw_logits[
            :, sum(self.n_stuff_objects) :
        ]

        return logits

    def training_step(self, batch, batch_idx):
        features, gt_segmentation = batch
        gt_segmentation = (
            F.interpolate(
                gt_segmentation.float().unsqueeze(dim=1),
                size=features.shape[-2:],
                mode="nearest-exact",
            )
            .long()
            .squeeze(dim=1)
        )

        logits = self(features)

        loss = F.cross_entropy(logits, gt_segmentation)

        self.log("linear_probe_train_loss", loss)
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_results.update(
            {
                "all_logits": [],
                "all_gt_labels": [],
                "true_positives": torch.zeros(self.n_objects, dtype=torch.long),
                "false_positives": torch.zeros(self.n_objects, dtype=torch.long),
                "false_negatives": torch.zeros(self.n_objects, dtype=torch.long),
                "true_negatives": torch.zeros(self.n_objects, dtype=torch.long),
            }
        )

    def test_step(self, batch, batch_idx):
        features, gt_segmentation_class = batch

        logits = self(features)

        # upsample logits to match gt_segmentation
        upsampled_logits = F.interpolate(
            logits, size=gt_segmentation_class.shape[-2:], mode="nearest-exact"
        )
        predicted_segmentation = torch.argmax(upsampled_logits, dim=1)
        upsampled_logits = None  # free memory
        predicted_segmentation = predicted_segmentation.unsqueeze(-1) == torch.arange(
            self.n_objects, device=predicted_segmentation.device
        ).view(1, 1, 1, self.n_objects)

        gt_segmentation = gt_segmentation_class.to(
            predicted_segmentation.device
        ).unsqueeze(dim=-1) == torch.arange(
            self.n_objects, device=predicted_segmentation.device
        ).view(
            1, 1, 1, self.n_objects
        )

        for i in range(self.n_objects):
            self.test_results["true_positives"][i] += torch.sum(
                predicted_segmentation[:, :, :, i] & gt_segmentation[:, :, :, i]
            ).cpu()
            self.test_results["false_positives"][i] += torch.sum(
                predicted_segmentation[:, :, :, i] & ~gt_segmentation[:, :, :, i]
            ).cpu()
            self.test_results["false_negatives"][i] += torch.sum(
                ~predicted_segmentation[:, :, :, i] & gt_segmentation[:, :, :, i]
            ).cpu()
            self.test_results["true_negatives"][i] += torch.sum(
                ~predicted_segmentation[:, :, :, i] & ~gt_segmentation[:, :, :, i]
            ).cpu()
        predicted_segmentation = None
        gt_segmentation = None

        # to compute AP over the entire epoch, I need to store the actual logits and grond truth labels -- do that in the downsampled version to save memory
        downsampled_gt_segmentation = (
            F.interpolate(
                gt_segmentation_class.unsqueeze(dim=1).float(),
                size=logits.shape[-2:],
                mode="nearest-exact"
                if not torch.__version__ in ["1.10.2", "1.9.0"]
                else "nearest",
            )
            .squeeze(dim=1)
            .long()
        ).cpu()

        # scale to between 0 and 1 so the AP calculation is reasonably accurate with 100 bins
        # only thing that matters for average precision is that the ordering is preserved for each class individually
        logits = logits - torch.min(logits, dim=1, keepdim=True)[0]
        logits = logits / torch.max(logits, dim=1, keepdim=True)[0]
        logits = logits.cpu()

        self.test_results["all_logits"].append(logits)
        self.test_results["all_gt_labels"].append(downsampled_gt_segmentation)

    def on_test_epoch_end(self) -> None:
        """compute metrics across entire epoch"""
        self.test_results["mIoU"] = torch.mean(
            self.test_results["true_positives"]
            / (
                self.test_results["true_positives"]
                + self.test_results["false_positives"]
                + self.test_results["false_negatives"]
                + 1e-8
            )
        ).item()
        self.test_results["Jaccard"] = (
            torch.sum(self.test_results["true_positives"])
            / (
                torch.sum(self.test_results["true_positives"])
                + torch.sum(self.test_results["false_positives"])
                + torch.sum(self.test_results["false_negatives"])
            ).item()
        )
        self.test_results["Accuracy"] = (
            torch.sum(self.test_results["true_positives"])
            + torch.sum(self.test_results["true_negatives"])
        ) / (
            torch.sum(self.test_results["true_positives"])
            + torch.sum(self.test_results["false_positives"])
            + torch.sum(self.test_results["false_negatives"])
            + torch.sum(self.test_results["true_negatives"])
        ).item()

        self.test_results["object_mIoU"] = torch.mean(
            self.test_results["true_positives"][len(self.n_stuff_objects) :]
            / (
                self.test_results["true_positives"][len(self.n_stuff_objects) :]
                + self.test_results["false_positives"][len(self.n_stuff_objects) :]
                + self.test_results["false_negatives"][len(self.n_stuff_objects) :]
                + 1e-8
            )
        ).item()
        self.test_results["object_Jaccard"] = (
            self.test_results["true_positives"][len(self.n_stuff_objects) :].sum()
            / (
                self.test_results["true_positives"][len(self.n_stuff_objects) :].sum()
                + self.test_results["false_positives"][
                    len(self.n_stuff_objects) :
                ].sum()
                + self.test_results["false_negatives"][
                    len(self.n_stuff_objects) :
                ].sum()
            ).item()
        )
        self.test_results["object_Accuracy"] = (
            self.test_results["true_positives"][len(self.n_stuff_objects) :].sum()
            + self.test_results["true_negatives"][len(self.n_stuff_objects) :].sum()
        ) / (
            self.test_results["true_positives"][len(self.n_stuff_objects) :].sum()
            + self.test_results["false_positives"][len(self.n_stuff_objects) :].sum()
            + self.test_results["false_negatives"][len(self.n_stuff_objects) :].sum()
            + self.test_results["true_negatives"][len(self.n_stuff_objects) :].sum()
        ).item()

        self.test_results["stuff_mIoU"] = torch.mean(
            self.test_results["true_positives"][: len(self.n_stuff_objects)]
            / (
                self.test_results["true_positives"][: len(self.n_stuff_objects)]
                + self.test_results["false_positives"][: len(self.n_stuff_objects)]
                + self.test_results["false_negatives"][: len(self.n_stuff_objects)]
                + 1e-8
            )
        ).item()
        self.test_results["stuff_Jaccard"] = (
            self.test_results["true_positives"][: len(self.n_stuff_objects)].sum()
            / (
                self.test_results["true_positives"][: len(self.n_stuff_objects)].sum()
                + self.test_results["false_positives"][
                    : len(self.n_stuff_objects)
                ].sum()
                + self.test_results["false_negatives"][
                    : len(self.n_stuff_objects)
                ].sum()
            ).item()
        )
        self.test_results["stuff_Accuracy"] = (
            self.test_results["true_positives"][: len(self.n_stuff_objects)].sum()
            + self.test_results["true_negatives"][: len(self.n_stuff_objects)].sum()
        ) / (
            self.test_results["true_positives"][: len(self.n_stuff_objects)].sum()
            + self.test_results["false_positives"][: len(self.n_stuff_objects)].sum()
            + self.test_results["false_negatives"][: len(self.n_stuff_objects)].sum()
            + self.test_results["true_negatives"][: len(self.n_stuff_objects)].sum()
        ).item()

        all_logits = torch.cat(self.test_results["all_logits"], dim=0)
        all_gt_labels = torch.cat(self.test_results["all_gt_labels"], dim=0)

        AP_metric = torchmetrics.classification.BinaryAveragePrecision()
        class_APs = torch.zeros(self.n_objects, device=all_logits.device)
        for i in tqdm(range(self.n_objects), "calculating mAP"):
            class_APs[i] = AP_metric(all_logits[:, i], (all_gt_labels == i))

        self.test_results["mAP"] = torch.mean(class_APs[~torch.isnan(class_APs)]).item()
        self.test_results["object_mAP"] = torch.mean(
            class_APs[len(self.n_stuff_objects) :][
                ~torch.isnan(class_APs[len(self.n_stuff_objects) :])
            ]
        ).item()
        self.test_results["stuff_mAP"] = torch.mean(
            class_APs[: len(self.n_stuff_objects)][
                ~torch.isnan(class_APs[: len(self.n_stuff_objects)])
            ]
        ).item()

    def get_results(self):
        return self.test_results
