"""
model.py  ——  修改版
在原始 Model 基础上新增：
  1. recon_decoder   : 记忆重建解码器 (输入192，输出48像素)
  2. v_scale_head    : 轻量速度缩放头（通过 detach 隔离梯度）
  3. decode()        : 供训练循环调用的重建接口
  4. forward() 支持返回 v_scale（可选）

原有逻辑（stem / v_proj / gru / fc）完全不动，确保能完美加载 Checkpoint。
"""

import torch
from torch import nn
import torch.nn.functional as F

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

class Model(nn.Module):
    def __init__(
        self,
        dim_obs: int = 9,
        dim_action: int = 4,
        recon_members: tuple = (True, True, False),
        recon_steps: int = 12,
        enable_v_scale: bool = True,
    ) -> None:
        super().__init__()

        # ── 原有结构（一字不改，确保能加载旧权重）────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),   
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),     
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),    
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, 192, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, 192)
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192, 192)
        self.fc  = nn.Linear(192, dim_action, bias=False)  
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

        # ── 新增：记忆重建解码器 ───────────────────────────────
        self.recon_members = recon_members
        self.recon_steps   = recon_steps
        
        # 直接用 192 维特征重建 6x8=48 像素，去掉 Sigmoid，降低难度
        self.recon_decoder = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 48)  # 输出 48 维
        )
        
        # 小权重初始化
        for layer in self.recon_decoder:
            if hasattr(layer, 'weight'):
                layer.weight.data.mul_(0.1)

        # ── 新增：轻量速度缩放头 ──────────────────────────────
        self.enable_v_scale = enable_v_scale
        if enable_v_scale:
            self.v_scale_head = nn.Sequential(
                nn.Linear(192, 32),
                nn.LeakyReLU(0.05),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.v_scale_head[2].weight)
            nn.init.constant_(self.v_scale_head[2].bias, 1.5)

    # ── 重建接口 ────────────────────
    def decode(self, hx: torch.Tensor) -> torch.Tensor:
        """供外部调用，用 GRU 的 192 维隐状态去还原深度图"""
        return self.recon_decoder(hx)

    def reset(self):
        pass

    def forward(
        self,
        x: torch.Tensor,       
        v: torch.Tensor,       
        hx=None,               
    ):
        # 主线物理计算逻辑
        img_feat = self.stem(x)
        feat     = self.act(img_feat + self.v_proj(v))
        hx       = self.gru(feat, hx)
        act      = self.fc(self.act(hx))

        # 根据配置返回不同数量的变量
        if self.enable_v_scale:
            v_scale = self.v_scale_head(hx.detach())
            return act, None, hx, v_scale
        else:
            return act, None, hx


# ── 单元测试 ────────────────────
if __name__ == '__main__':
    B = 4
    model = Model(dim_obs=9, dim_action=6, enable_v_scale=True)

    x  = torch.randn(B, 1, 12, 16)
    v  = torch.randn(B, 9)
    hx = None

    act, _, hx, v_scale = model(x, v, hx)
    print(f"act shape    : {act.shape}")       # (4, 6)
    print(f"hx  shape    : {hx.shape}")        # (4, 192)
    print(f"v_scale range: {v_scale.min():.3f} ~ {v_scale.max():.3f}")  

    recon = model.decode(hx)
    print(f"recon shape  : {recon.shape}")     # (4, 48)  <-- 修改为 48 维了！
    print("model.py 单元测试通过 ✓")