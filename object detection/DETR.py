import torch
from torch import nn
from torchvision.models import resnet50
from torch.nn.modules.container import ModuleList

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers) -> None:
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv1x1 = nn.Conv2d(2048, hidden_dim, kernel_size=(1, 1))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.transformer = DETRTransformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.ffn_cls = nn.Linear(hidden_dim, num_classes+1)
        self.ffn_bbox = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        x = self.backbone(x)
        B, C, H, W = x.shape
        x = self.conv1x1(x)
        pos_embed = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
       # x = self.transformer(pos + x.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
        query_pos = self.query_pos.unsqueeze(1)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer(x, pos_embed, query_pos)
        return self.ffn_cls(x), self.ffn_bbox(x).sigmoid()


class DETRTransformer(nn.Module):
    def __init__(self, hidden_dim, nheads, num_encoder_layers, num_decoder_layers) -> None:
        super().__init__()
        self.encoder = DETREncoder(nheads, hidden_dim, num_encoder_layers)
        self.decoder = DETRDecoder(nheads, hidden_dim, num_decoder_layers)
        

    def forward(self, x, pos_embed, query_pos):
        x = self.encoder(x, pos_embed)
        y = self.decoder(x, pos_embed, query_pos)
        return y


class DETREncoder(nn.Module):
    def __init__(self, nheads, hidden_dim, num_encoder_layers) -> None:
        super().__init__()
        self.layers = ModuleList([DETREncoderLayer(hidden_dim, nheads) for _ in range(num_encoder_layers)])

    def forward(self, x, pos_embed):
        for mod in self.layers:
            x = mod(x, pos_embed)
        return x


class DETREncoderLayer(nn.Module):
    def __init__(self, hidden_dim, nheads) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, nheads)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def forward(self, x, pos_embed):
        q, k, v = x+pos_embed, x+pos_embed, x
        print('encoder', q.shape, k.shape, v.shape)
        x = self.mha(q, k, v)[0]
        x = torch.add(x, self.layernorm1(self.dropout(x)))
        x = self.ffn(x)
        y = torch.add(x, self.layernorm2(self.dropout(x)))
        return y


class DETRDecoder(nn.Module):
    def __init__(self, nheads, hidden_dim, num_decoder_layers) -> None:
        super().__init__()
        self.q_mhsa = nn.MultiheadAttention(hidden_dim, nheads)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.layers = ModuleList([DETRDecoderLayer(hidden_dim, nheads) for _ in range(num_decoder_layers)])
        
    def forward(self, x, pos_embed, query_pos):
        B = x.shape[1]
        output = self.q_mhsa(query_pos+query_pos, query_pos+query_pos, query_pos)[0]
        output = torch.add(query_pos, self.layernorm(self.dropout(output))).repeat(1, B, 1)
        for mod in self.layers:
            output = mod(x, output, pos_embed, query_pos)
        return output


class DETRDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, nheads) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, nheads)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def forward(self, x, prev_output, pos_embed, query_pos):
        q, k, v = prev_output+query_pos, x+pos_embed, x
        print('decoder', q.shape, k.shape, v.shape)
        x = self.mha(q, k, v)[0]
        x = torch.add(x, self.layernorm1(self.dropout(x)))
        x = self.ffn(x)
        y = torch.add(x, self.layernorm2(self.dropout(x)) )
        return y


if __name__ == '__main__':
    input = torch.rand(2, 3, 800, 800).to('cuda')
    model = DETR(100, 256, 8, 6, 6).to('cuda')
    model.eval()
    output = model(input)
    print(output[0].shape, output[1].shape)
