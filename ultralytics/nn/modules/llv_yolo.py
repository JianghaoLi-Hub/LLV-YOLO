# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Paper-named LLV-YOLO modules; native operators and losses remain unchanged.

LNSR is encoded in YAML: P5 detection branch removal and GhostConv at layers 0/17.
SA_CSPP is the Python spelling of SA-CSPP. LLVYOLO_ avoids a native name collision.
"""
import torch
import torch.nn as nn

from .conv import Conv, DWConv

__all__ = ("DWPWConv", "DMHA", "EPSA", "LLVF", "LLVYOLO_SpatialAttention", "SA_CSPP")


class DWPWConv(nn.Module):
    """Depthwise + BN (no activation), then pointwise + BN + optional SiLU."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, act=True):
        super().__init__()
        self.dw = DWConv(c1, c1, k, s, act=False)
        self.pw = Conv(c1, c2, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DMHA(nn.Module):
    """Depthwise Multi-Head Attention; positional encoding acts on the value tensor."""

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        self.qkv = DWPWConv(dim, dim + nh_kd * 2, 1, act=False)
        self.proj = DWPWConv(dim, dim, 1, act=False)
        self.pe = DWPWConv(dim, dim, 3, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        q, k, v = self.qkv(x).view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


class EPSA(nn.Module):
    """Efficient Position-Sensitive Attention with a bypass and residual DMHA/FFN branch."""

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = DWPWConv(c1, 2 * self.c, 1, 1)
        self.cv2 = DWPWConv(2 * self.c, c1, 1)
        self.attn = DMHA(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(DWPWConv(self.c, self.c * 2, 1), DWPWConv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class LLVF(nn.Module):
    """Lightweight Low-level Visual Feature-driven module: spatial focusing parallel to EPSA."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(EPSA(self.c, self.c) for _ in range(max(n, 1))))
        self.spatial_gate = nn.Conv2d(2, 1, 7, 1, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        a_avg = a.mean(1, keepdim=True)
        a_max = a.amax(1, keepdim=True)
        a = a * torch.sigmoid(self.spatial_gate(torch.cat((a_avg, a_max), 1)))
        return self.cv2(torch.cat((a, self.m(b)), 1))


class LLVYOLO_SpatialAttention(nn.Module):
    """Channel-mean/max spatial gating, separate from native SpatialAttention."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        assert kernel_size > 0 and kernel_size % 2 == 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class SA_CSPP(nn.Module):
    """Spatial Attention-augmented Compact Spatial Pyramid Pooling (SA-CSPP).

    Concat order: input projection, first pool, second pool, spatially gated projection.
    k is the pooling kernel, not the attention kernel. Paper defaults: 5 and 3 respectively.
    """

    def __init__(self, c1: int, c2: int, k: int = 5, attention_k: int = 3):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.attn = LLVYOLO_SpatialAttention(attention_k)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_all = self.cv1(x)
        y = [x_all]
        y.extend(self.m(y[-1]) for _ in range(2))
        y.append(self.attn(x_all))
        return self.cv2(torch.cat(y, 1))
