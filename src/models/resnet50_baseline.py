import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.decomposition import PCA
from einops import rearrange


class ResNet50Baseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        feature_dim = config.FEATURE_DIM
        assert feature_dim == 64

        self.weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=self.weights)
        resnet.eval()
        self.network = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.network.eval()

        self.output_subsample = 32

        self.out_dim = 512
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
        assert H == 32, "{} not 32".format(H)
        assert W == 40, "{} not 40".format(W)

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
