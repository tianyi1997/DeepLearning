import torch
from torch import nn

# ViT-B/16

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(3, 224, 224), patch_size=(16, 16), hidden_size=768, num_blocks=12, num_heads=12, mlp_size=3072, output_size=100) -> None:
        super().__init__()
        self.img_size = img_size
        self.scale = hidden_size**-0.5
        num_patch = (img_size[1]//patch_size[0]) * (img_size[2]//patch_size[1])
        self.patch_embedding = nn.Conv2d(in_channels=img_size[0], out_channels=hidden_size, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(self.scale * torch.rand((1, hidden_size)))
        self.position_embedding = nn.Parameter(self.scale * torch.rand((num_patch + 1, hidden_size)))
        self.layernorm_pre = nn.LayerNorm(hidden_size)
        self.transformer = nn.Sequential(*[EncoderBlock(hidden_size, num_heads) for _ in range(num_blocks)])
        self.layernorm_post = nn.LayerNorm(hidden_size)
        self.mlp_head = nn.Sequential(
                                    nn.LayerNorm(hidden_size),
                                    nn.Linear(in_features=hidden_size, out_features=mlp_size),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(in_features=mlp_size, out_features=output_size),
                                )

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], "Incorrect image size."
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        x = torch.concat([self.class_token, x], dim=1)
        x = torch.add(x, self.position_embedding, dtype=x.type, device=x.device)
        x = self.layernorm_pre(x)
        x = self.transformer(x)
        x = self.layernorm_post(x)
        x = x.select(dim=1, index=0)
        y = self.mlp_head(x)
        return y


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_encoders=2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.encoders = nn.Sequential(*[ResidualAttentionBlock(hidden_size, num_heads) for _ in range(num_encoders)])

    def forward(self, x):
        y = self.encoders(x)
        return y


class ResidualAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=0)
        
    def forward(self, x):
        residual = self.layernorm(x)
        residual = self.multi_head_attn(x, x, x)
        y = torch.add(x, residual)
        return y



