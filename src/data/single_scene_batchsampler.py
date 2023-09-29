import os
import random
import numpy as np
import torch


class SingleSceneBatchSampler(torch.utils.data.Sampler):
    """Samples batches from a ConcatDataset so that each batch contains only samples from a single scene (sub-dataset)."""

    def __init__(self, datasource, batch_size, shuffle=True, use_overlaps=True):
        super().__init__(datasource)

        self.datasource = datasource
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_overlaps = use_overlaps

        self.n_scenes = len(datasource.datasets)
        self.scene_lengths = [len(dataset) for dataset in datasource.datasets]
        self.batches_per_scene = [length // batch_size for length in self.scene_lengths]

        if use_overlaps:
            self.overlap_matrices = []

            for scene in datasource.datasets:
                scene_name = scene.scene_name
                overlap_matrix = torch.from_numpy(
                    np.load(
                        os.path.join(
                            scene.data_root,
                            scene_name,
                            "img_overlaps.npy",
                        )
                    )
                )

                # filter the overlaps so they only contain images that are used
                _, used_img_mask = scene.filter_img_files(scene.img_dir)
                overlap_matrix = overlap_matrix[used_img_mask, :][:, used_img_mask]

                self.overlap_matrices.append(overlap_matrix)

    def _make_epoch_with_overlaps(self):
        """Create a list of batches, with each batch containing just images from a single scene.
        First randomly sample a scene, then randomly sample an seed image from that scene, then sample the rest of the batch weighted by the relative overlaps with the seed image (i.e. images that overlap more with the seed image are more likely to be sampled).
        """

        batches = []
        batches_sampled_per_scene = [0 for _ in range(self.n_scenes)]
        remaining_scene_idxs = list(range(self.n_scenes))
        all_unused_images = [list(range(length)) for length in self.scene_lengths]

        for _ in range(len(self)):
            # sample a scene for which there are batches left to sample
            if self.shuffle:
                scene_idx = random.choice(remaining_scene_idxs)
            else:
                scene_idx = remaining_scene_idxs[0]
            batches_sampled_per_scene[scene_idx] += 1
            if (
                batches_sampled_per_scene[scene_idx]
                >= self.batches_per_scene[scene_idx]
            ):
                remaining_scene_idxs.remove(scene_idx)

            scene_length = self.scene_lengths[scene_idx]
            overlap_matrix = self.overlap_matrices[scene_idx]
            unused_images = all_unused_images[scene_idx]

            scene_idx_offset = (
                self.datasource.cumulative_sizes[scene_idx] - scene_length
            )

            # sample a seed image
            if self.shuffle:
                seed_idx = random.choice(unused_images)
            else:
                seed_idx = unused_images[0]
            unused_images.remove(seed_idx)

            # sample the rest of the batch
            batch = [seed_idx + scene_idx_offset]
            if self.shuffle:
                # sample the rest of the batch from the overlap matrix
                indices = torch.multinomial(
                    overlap_matrix[seed_idx],
                    self.batch_size - 1,
                    replacement=False,
                )
                # NOTE I am actually not removing the images that are sampled, so some are used multiple times and some not at all (for a certain epoch). This is because otherwise the overlap tends to decrease throughout the epoch as the number of unused images decreases, which tends to artificially increase the AP.
            else:
                # take the images with the largest overlaps with the seed image
                indices = torch.topk(
                    overlap_matrix[seed_idx],
                    self.batch_size - 1,
                )[1]

            batch.extend([idx.item() for idx in indices + scene_idx_offset])
            batches.append(batch)

        return batches

    def _make_epoch_without_overlaps(self):
        """create list of batches sampled uniformly, with each batch coming from only a single scene"""
        batches = []

        for scene_idx in range(self.n_scenes):
            scene_length = self.scene_lengths[scene_idx]
            n_batches = self.batches_per_scene[scene_idx]

            indices = list(range(scene_length))
            indices = [
                idx + self.datasource.cumulative_sizes[scene_idx] - scene_length
                for idx in indices
            ]
            if self.shuffle:
                random.shuffle(indices)

            for batch_idx in range(n_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = (batch_idx + 1) * self.batch_size
                batches.append(indices[batch_start:batch_end])

        if self.shuffle:
            random.shuffle(batches)

        return batches

    def __len__(self):
        return sum(self.batches_per_scene)

    def __iter__(self):
        # create batches first, then yield from the tensor of indices
        if self.use_overlaps:
            batches = self._make_epoch_with_overlaps()
        else:
            batches = self._make_epoch_without_overlaps()

        for batch in batches:
            yield batch
