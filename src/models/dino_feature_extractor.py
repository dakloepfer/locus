"""Feature extractor that uses frozen, pre-trained DINO model and several linear layers on top."""

import torch
import torch.nn as nn
from einops import rearrange
from timm.models import vision_transformer


######################## Copied from facebookresearch_dino_main/vision_transformer.py ########################
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


############################################################################################################


class DINO_FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.N_PRETRAINED_BLOCKS <= 11

        self.output_subsample = 8
        self.n_pretrained_blocks = config.N_PRETRAINED_BLOCKS

        self.dino_model = torch.hub.load(
            "facebookresearch/dino:main", "dino_{}".format(config.VIT_BACKBONE)
        ).eval()

        backbone_dim = 384 if config.VIT_BACKBONE == "vits8" else 768

        self.trainable_head = nn.Sequential(
            *(
                [nn.Linear(backbone_dim, 128)]
                + [
                    Block(128, num_heads=4, qkv_bias=True, qk_scale=0.125)
                    for _ in range(2)
                ]
                + [nn.Linear(128, config.FEATURE_DIM)]
            )
        )

        self.dino_model.blocks = self.dino_model.blocks[: self.n_pretrained_blocks]

        for param in self.dino_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape

        with torch.no_grad():
            x = self.dino_model.prepare_tokens(x)
            for blk in self.dino_model.blocks[: self.n_pretrained_blocks]:
                x = blk(x)

        # turn into extracted features
        x = self.trainable_head(x)
        x = x[:, 1:]

        x = rearrange(x, "b (h w) c -> b c h w", h=int(H // 8), w=int(W // 8))

        return x

    def get_dino_features(self, x):
        B, C, H, W = x.shape

        with torch.no_grad():
            x = self.dino_model.prepare_tokens(x)
            for blk in self.dino_model.blocks[:-1]:
                x = blk(x)

        return rearrange(
            x[:, 1:], "b (h w) c -> b c h w", h=int(H // 8), w=int(W // 8)
        )  # first token does not derive from an image patch
