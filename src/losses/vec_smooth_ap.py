import torch
from einops import rearrange


class VecSmoothAP(torch.nn.Module):
    """Computes the Vectorised Smooth Average Precision loss."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.sigmoid_temperature = config.SIGMOID_TEMPERATURE

    def forward(self, landmarks, pos_patches, pos_neg_patches, patch_features):
        """Computes the Vectorised Smooth Average Precision loss given landmarks, masks for positive patches and all patches, and the features.

        Parameters
        ----------
        landmarks (dict):
            minimal keys:
                landmark_embeddings (n_landmarks x feature_dim tensor):
                    the embeddings for the landmarks; should be normalised

        pos_patches (n_landmarks x n_all_patches bool tensor):
            A mask for the patches that are used as positive patches

        pos_neg_patches (n_landmarks x n_all_patches bool tensor):
            A mask for the patches that are used as either positive or negative patches.

        patch_features (n_all_patches x feature_dim tensor):
            The features for all the patches; should be normalised

        Returns
        -------
        loss (scalar tensor):
            -Vectorised Smooth AP as the loss

        log_dict (dict):
            a dictionary with information to log
        """
        landmark_embeddings = landmarks["landmark_embeddings"]

        n_all_patches = patch_features.shape[0]
        n_landmarks = landmark_embeddings.shape[0]

        similarities = landmark_embeddings @ patch_features.t()

        similarities = rearrange(
            similarities,
            "n_landmarks n_all_patches -> (n_landmarks n_all_patches)",
            n_landmarks=n_landmarks,
            n_all_patches=n_all_patches,
        )
        pos_patches = rearrange(
            pos_patches,
            "n_landmarks n_all_patches -> (n_landmarks n_all_patches)",
            n_landmarks=n_landmarks,
            n_all_patches=n_all_patches,
        )
        pos_neg_patches = rearrange(
            pos_neg_patches,
            "n_landmarks n_all_patches -> (n_landmarks n_all_patches)",
            n_landmarks=n_landmarks,
            n_all_patches=n_all_patches,
        )

        # similarity_differences[i, j] = similarity of the j-th positive/negative pair - similarity of the i-th positive pair
        # shape = (n_positive_pairs, n_pos_neg_pairs)
        similarity_differences = similarities[pos_neg_patches].unsqueeze(
            dim=0
        ) - similarities[pos_patches].unsqueeze(dim=1)

        # pass through sigmoid
        similarity_differences = torch.sigmoid(
            similarity_differences / self.sigmoid_temperature
        )

        # Compute the rankings of the positive pairs with respect to the entire set
        ranking_pos_neg = 1 + torch.sum(similarity_differences, dim=1)

        # Compute the rankings of the positive pairs with respect to the positive pairs
        ranking_pos = 1 + torch.sum(
            similarity_differences * pos_patches[pos_neg_patches].unsqueeze(dim=0),
            dim=1,
        )

        vec_smooth_ap = torch.mean(ranking_pos / ranking_pos_neg)

        loss = -vec_smooth_ap

        log_dict = {
            "vec_smooth_ap": vec_smooth_ap.detach(),
            "loss": loss.detach(),
            "avg_pos_similarity": similarities[pos_patches].mean().detach(),
            "avg_neg_similarity": similarities[pos_neg_patches & (~pos_patches)]
            .mean()
            .detach(),
            "avg_dontcare_similarity": similarities[~pos_neg_patches].mean().detach(),
        }

        return loss, log_dict
