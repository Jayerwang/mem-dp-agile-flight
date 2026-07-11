import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_ch,
            bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(x)


class Model(nn.Module):
    def __init__(self, dim_obs=10, dim_action=6) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(1, 32, 2, 2),
            DepthwiseSeparableConv(32, 64, 3, 1),
            DepthwiseSeparableConv(64, 96, 3, 1),
            nn.Flatten(),
            nn.LayerNorm(96 * 2 * 4),
            nn.Linear(96 * 2 * 4, 128, bias=False),
            nn.ReLU(inplace=True),
        )

        self.v_proj = nn.Linear(dim_obs, 128)
        self.fuse_act = nn.ReLU(inplace=True)

        self.gru = nn.GRUCell(128, 256)

        self.action_head = nn.Linear(256, dim_action, bias=False)
        self.sigma_head = nn.Sequential(
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid(),
        )
        self.memory_gate_head = nn.Sequential(
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid(),
        )

        self.aux_decoder = nn.Sequential(
            nn.Linear(256, 96 * 2 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (96, 2, 4)),
            nn.ConvTranspose2d(96, 64, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, bias=False),
        )

        self.action_head.weight.data.mul_(0.01)
        self.v_proj.weight.data.mul_(0.5)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v: torch.Tensor, hx=None):
        img_feat = self.stem(x)
        fused = self.fuse_act(img_feat + self.v_proj(v))
        hx = self.gru(fused, hx)

        hidden = self.fuse_act(hx)
        act = self.action_head(hidden)
        sigma = self.sigma_head(hidden)
        memory_gate = self.memory_gate_head(hidden)

        recon = None
        if self.training:
            recon_raw = self.aux_decoder(hx)
            gate_map = memory_gate.view(-1, 1, 1, 1)
            # Blend decoder output with current frame to suppress stale memory under hard perturbations.
            recon = gate_map * recon_raw + (1.0 - gate_map) * x

        aux = {
            'sigma': sigma,
            'recon': recon,
            'memory_gate': memory_gate,
        }
        return act, aux, hx


if __name__ == '__main__':
    m = Model()
    x = torch.randn(2, 1, 12, 16)
    v = torch.randn(2, 10)
    a, aux, h = m(x, v)
    print(a.shape, aux['sigma'].shape, aux['recon'].shape, h.shape)
