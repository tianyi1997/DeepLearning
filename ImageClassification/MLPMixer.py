import torch 
from torch import nn


def mlp(num_features, hidden_dim):
    return nn.Sequential(
        nn.Linear(num_features, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, num_features)
    )


class MLPMixer(nn.Module):
    def __init__(self,
        image_size,
        num_classes,
        patch_size,
        num_layers,
        num_channels,
        mlp_dim_channel, 
        mlp_dim_token
        ) -> None:
        super().__init__()
        self.image_size = image_size
        assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0
        num_tokenes = (image_size[0] // patch_size) * (image_size[1] // patch_size) 
        self.proj = nn.Conv2d(in_channels=3, out_channels=num_channels, kernel_size=patch_size, stride=patch_size)
        self.mixerlayers = nn.Sequential(*[MixerLayer(num_channels, num_tokenes, mlp_dim_channel, mlp_dim_token) for _ in range(num_layers)])
        self.layernorm = nn.LayerNorm(num_channels)
        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.image_size
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.mixerlayers(x).transpose(1, 2)
        x = self.globalpool(x).flatten(1)
        y = self.fc(x)
        return y        

    
class MixerLayer(nn.Module):
    def __init__(self, num_channels, num_tokenes, mlp_dim_channel, mlp_dim_token) -> None:
        super().__init__()
        self.channelmixing = ChannelMixingMLP(num_channels, mlp_dim_channel)
        self.tokenmixing = TokenMixingMLP(num_tokenes, num_channels, mlp_dim_token)

    def forward(self, x):
        x = self.tokenmixing(x)
        y = self.channelmixing(x)
        return y

    
class ChannelMixingMLP(nn.Module):
    def __init__(self, num_channels, hidden_dim) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(num_channels)
        self.mlp = mlp(num_channels, hidden_dim)
        
    def forward(self, x):
        res = self.layernorm(x)
        y = self.mlp(res) + x
        return y


class TokenMixingMLP(nn.Module):
    def __init__(self, num_tokenes, num_channels, hidden_dim) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(num_channels)
        self.mlp = mlp(num_tokenes, hidden_dim)

    def forward(self, x):
        res = self.layernorm(x)
        res = res.transpose(1, 2)
        res = self.mlp(res)
        y = res.transpose(1, 2) + x
        return y
    

if __name__ == '__main__':
    model = MLPMixer((224, 224), 100, 32, 8, 512, 2048, 256)
    model.eval()
    inputs = torch.rand((5, 3, 224, 224))
    print(model(inputs).shape)   