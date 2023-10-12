import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from einops import rearrange


class DINO_Network(nn.Module):
    def __init__(self, n_blocks=11, model="vitb8"):
        super().__init__()

        assert n_blocks <= 11
        assert model in ["vitb8", "vits8"]
        self.output_subsample = 8

        self.n_blocks = n_blocks

        self.dino_model = torch.hub.load(
            "facebookresearch/dino:main", "dino_{}".format(model)
        ).eval()

        self.out_dim = 384 if model == "vits8" else 768

        self.dino_model.blocks = self.dino_model.blocks[:n_blocks]
        for param in self.dino_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape

        with torch.no_grad():
            x = self.dino_model.prepare_tokens(x)

            # if I stop before the last block, the elements should have all the relevant information about the image patches
            # the attention masks that they use to segment images attend to these features
            for blk in self.dino_model.blocks[: self.n_blocks]:
                x = blk(x)

        return rearrange(
            x[:, 1:], "b (h w) c -> b c h w", h=int(H // 8), w=int(W // 8)
        )  # first token does not derive from an image patch


class DINO_Baseline(nn.Module):
    # Just a wrapper to keep the code consistent with the actual models
    def __init__(self, config):
        super().__init__()

        n_blocks = config.N_PRETRAINED_BLOCKS
        feature_dim = config.FEATURE_DIM
        model = config.VIT_BACKBONE

        assert n_blocks <= 11
        self.network = DINO_Network(n_blocks=n_blocks, model=model)
        self.out_dim = self.network.out_dim
        self.feature_dim = feature_dim
        self.training_features_for_pca = []
        self.pca = None

    def calc_and_store_features_for_pca(self, imgs):
        features = self.network(imgs)
        self.training_features_for_pca.append(
            rearrange(features.cpu(), "b c h w -> (b h w) c")
        )

    def calculate_pca(self):
        """Use the stored training features to calculate principal components"""
        if len(self.training_features_for_pca) == 0:
            raise ValueError("No training features stored")

        self.training_features_for_pca = torch.cat(
            self.training_features_for_pca, dim=0
        ).numpy()
        self.pca = PCA(self.feature_dim)
        self.pca.fit(self.training_features_for_pca)
        self.training_features_for_pca = None
        self.pca_mean = (
            torch.tensor(
                self.pca.mean_, dtype=torch.float, device=next(self.parameters()).device
            )
            if self.pca.mean_ is not None
            else torch.zeros(1, self.feature_dim, device=next(self.parameters()).device)
        )
        self.pca_components = torch.tensor(
            self.pca.components_,
            dtype=torch.float,
            device=next(self.parameters()).device,
        )

    def load_pca(self, pca):
        self.pca = pca
        self.training_features_for_pca = None
        self.pca_mean = (
            torch.tensor(
                self.pca.mean_, dtype=torch.float, device=next(self.parameters()).device
            )
            if self.pca.mean_ is not None
            else torch.zeros(1, self.feature_dim, device=next(self.parameters()).device)
        )
        self.pca_components = torch.tensor(
            self.pca.components_,
            dtype=torch.float,
            device=next(self.parameters()).device,
        )

    def forward(self, imgs):
        features = self.network(imgs)
        B, _, H, W = features.shape

        if self.feature_dim != self.out_dim:
            if self.pca is not None:
                features = rearrange(features, "b c h w -> (b h w) c")
                features = features - self.pca_mean
                features = torch.matmul(features, self.pca_components.T)
                features = rearrange(
                    features, "(b h w) c -> b c h w", b=B, h=H, w=W, c=self.feature_dim
                )
            else:
                raise ValueError("PCA not calculated yet")

        return features
