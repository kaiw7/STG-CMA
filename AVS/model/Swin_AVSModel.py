import torch
import torch.nn as nn
import torchvision.models as models
from AVS.model.TPAVI import TPAVIModule
from ipdb import set_trace
import timm
import torch.nn.functional as F
from einops import rearrange, repeat
import pdb

import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is expected (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = xs
        return x


class SAdapter2(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.act = act_layer()

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = x + xs
        # x = rearrange(x, 'B N T D -> (B T) N D')
        return x


class T_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.act = act_layer()

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = xs
        # x = rearrange(x, 'B N T D -> (B T) N D')
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_ttokens, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        ## relative position bias
        self.num_ttokens = num_ttokens
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * num_ttokens - 1, num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)
        coords = torch.arange(num_ttokens)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += num_ttokens - 1
        relative_coords = relative_coords.view(-1)
        self.register_buffer("relative_coords", relative_coords)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        ## get relative pos bias
        relative_pos_bias = self.relative_position_bias_table[self.relative_coords].view(self.num_ttokens,
                                                                                         self.num_ttokens, -1).permute(
            2, 0, 1).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + relative_pos_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_ttokens, window_size, num_heads, use_temporal=True, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        ## relative pos bias for temporal dimension
        if use_temporal:
            self.num_ttokens = num_ttokens
            self.temporal_position_bias_table = nn.Parameter(torch.zeros(2 * num_ttokens - 1, num_heads))
            trunc_normal_(self.temporal_position_bias_table, std=.02)
            t_coords = torch.arange(num_ttokens)
            t_relative_coords = t_coords[:, None] - t_coords[None, :]
            t_relative_coords += num_ttokens - 1
            t_relative_coords = t_relative_coords.view(-1)
            self.register_buffer("t_relative_coords", t_relative_coords)


            self.temporal_position_bias_table_audio = nn.Parameter(torch.zeros(2 * num_ttokens - 1, num_heads))
            trunc_normal_(self.temporal_position_bias_table_audio, std=.02)
            t_coords_a = torch.arange(num_ttokens)
            t_relative_coords_a = t_coords_a[:, None] - t_coords_a[None, :]
            t_relative_coords_a += num_ttokens - 1
            t_relative_coords_a = t_relative_coords_a.view(-1)
            self.register_buffer("t_relative_coords_a", t_relative_coords_a)

    def forward(self, x, mask=None, temporal=False, signal='video'):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if temporal:
            if signal == 'video':
                relative_pos_bias = self.temporal_position_bias_table[self.t_relative_coords].view(self.num_ttokens,
                                                                                                   self.num_ttokens,
                                                                                                   -1).permute(2, 0, 1).contiguous()
            else:
                relative_pos_bias = self.temporal_position_bias_table_audio[self.t_relative_coords_a].view(self.num_ttokens,
                                                                                                   self.num_ttokens,
                                                                                                   -1).permute(2, 0,
                                                                                                               1).contiguous()
            attn = attn + relative_pos_bias.unsqueeze(0)
            attn = self.softmax(attn)
        else:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N, T, temporal=False):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        if temporal:
            flops += self.dim * N * T * T * 2
        else:
            # attn = (q @ k.transpose(-2, -1))
            flops += self.num_heads * N * (self.dim // self.num_heads) * N * T
            #  x = (attn @ v)
            flops += self.num_heads * N * N * (self.dim // self.num_heads) * T
        # x = self.proj(x)
        # flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_frames, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., t_attn=False, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapter_mlp_ratio=0.25, mode='video_adapt'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.num_frames = num_frames
        self.mode = mode

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.t_attn = t_attn
        if t_attn:
            # self.temporal_norm = norm_layer(dim)
            # self.temporal_attn = Attention(dim=dim, num_ttokens=num_frames, num_heads=num_heads, qkv_bias=qkv_bias)
            if self.mode == 'video_adapt' or self.mode == 'multimodal_adapt_no_fusion' or self.mode == 'fusion_adapt':
                self.T_Adapter = T_Adapter(D_features=dim, mlp_ratio=adapter_mlp_ratio)

            if self.mode == 'audio_adapt' or self.mode == 'multimodal_adapt_no_fusion' or self.mode == 'fusion_adapt':
                self.T_Adapter_Audio = T_Adapter(D_features=dim, mlp_ratio=adapter_mlp_ratio)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, num_ttokens=num_frames, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            use_temporal=t_attn,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.mode == 'video_adapt' or self.mode == 'multimodal_adapt_no_fusion' or self.mode == 'fusion_adapt':
            self.S_Adapter = Adapter(dim, mlp_ratio=adapter_mlp_ratio)
            self.S_Adapter2 = SAdapter2(dim, mlp_ratio=adapter_mlp_ratio)

        if self.mode == 'audio_adapt' or self.mode == 'multimodal_adapt_no_fusion' or self.mode == 'fusion_adapt':
            self.S_Adapter_Audio = Adapter(dim, mlp_ratio=adapter_mlp_ratio)
            self.S_Adapter2_Audio = SAdapter2(dim, mlp_ratio=adapter_mlp_ratio)

        self.gate_v = nn.Parameter(torch.zeros(1))
        self.gate_a = nn.Parameter(torch.zeros(1))

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        if self.mode == 'video_adapt':
            H, W = self.input_resolution
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"
            ## Temporal attention
            if self.t_attn:
                x = rearrange(x, '(b t) n c -> (b n) t c', t=self.num_frames, n=L, c=C)
                res_temporal = self.attn(self.norm1(x), temporal=True, signal='video')
                res_temporal = self.T_Adapter(res_temporal)
                x = x + self.drop_path(res_temporal)
                x = rearrange(x, '(b n) t c -> (b t) n c', t=self.num_frames, n=L, c=C)

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=self.attn_mask, signal='video')  # nW*B, window_size*window_size, C
            attn_windows = self.S_Adapter2(attn_windows)

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(B, H * W, C)
            # x = shortcut + self.drop_path(x)
            x = shortcut + x

            # FFN
            xn = self.norm2(x)
            x = x + self.mlp(xn) + self.drop_path(0.5 * self.S_Adapter(xn))
            return x

        if self.mode == 'audio_adapt':
            H, W = self.input_resolution
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"
            ## Temporal attention
            if self.t_attn:
                x = rearrange(x, '(b t) n c -> (b n) t c', t=self.num_frames, n=L, c=C)
                res_temporal = self.attn(self.norm1(x), temporal=True, signal='audio')
                res_temporal = self.T_Adapter_Audio(res_temporal)
                x = x + self.drop_path(res_temporal)
                x = rearrange(x, '(b n) t c -> (b t) n c', t=self.num_frames, n=L, c=C)

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=self.attn_mask, signal='audio')  # nW*B, window_size*window_size, C
            attn_windows = self.S_Adapter2_Audio(attn_windows)

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(B, H * W, C)
            # x = shortcut + self.drop_path(x)
            x = shortcut + x

            # FFN
            xn = self.norm2(x)
            x = x + self.mlp(xn) + self.drop_path(0.5 * self.S_Adapter_Audio(xn))
            return x

        if self.mode == 'multimodal_adapt_no_fusion':
            # finalized version for multimodal without fusion
            H, W = self.input_resolution
            v, a = x[0], x[1]
            B_v, L_v, C = v.shape
            B_a, L_a, _ = a.shape

            assert L_v == H * W, "input feature has wrong size"
            assert L_a == H * W, "input feature has wrong size"

            # todo for visual branch
            ## Temporal attention
            if self.t_attn:
                v = rearrange(v, '(b t) n c -> (b n) t c', t=self.num_frames, n=L_v, c=C)
                res_temporal_v = self.attn(self.norm1(v), temporal=True)
                res_temporal_v = self.T_Adapter(res_temporal_v)
                v = v + self.drop_path(res_temporal_v)
                v = rearrange(v, '(b n) t c -> (b t) n c', t=self.num_frames, n=L_v, c=C)

            shortcut_v = v
            v = self.norm1(v)
            v = v.view(B_v, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_v = v

            # partition windows
            v_windows = window_partition(shifted_v, self.window_size)  # nW*B, window_size, window_size, C
            v_windows = v_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows_v = self.attn(v_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            attn_windows_v = self.S_Adapter2(attn_windows_v)

            # merge windows
            attn_windows_v = attn_windows_v.view(-1, self.window_size, self.window_size, C)
            shifted_v = window_reverse(attn_windows_v, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                v = torch.roll(shifted_v, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                v = shifted_v
            v = v.view(B_v, H * W, C)
            # x = shortcut + self.drop_path(x)
            v = shortcut_v + v

            # FFN
            vn = self.norm2(v)
            vn = self.mlp(vn)
            v = v + vn + self.S_Adapter(vn) # self.drop_path(0.5 * )

            # todo for audio branch
            ## Temporal attention
            if self.t_attn:
                a = rearrange(a, '(b t) n c -> (b n) t c', t=self.num_frames, n=L_a, c=C)
                res_temporal_a = self.attn(self.norm1(a), temporal=True, signal='audio')
                res_temporal_a = self.T_Adapter_Audio(res_temporal_a)
                a = a + self.drop_path(res_temporal_a)
                a = rearrange(a, '(b n) t c -> (b t) n c', t=self.num_frames, n=L_a, c=C)

            shortcut_a = a
            a = self.norm1(a)
            a = a.view(B_a, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_a = torch.roll(a, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_a = a

            # partition windows
            a_windows = window_partition(shifted_a, self.window_size)  # nW*B, window_size, window_size, C
            a_windows = a_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows_a = self.attn(a_windows, mask=self.attn_mask, signal='audio')  # nW*B, window_size*window_size, C
            attn_windows_a = self.S_Adapter2_Audio(attn_windows_a)

            # merge windows
            attn_windows_a = attn_windows_a.view(-1, self.window_size, self.window_size, C)
            shifted_a = window_reverse(attn_windows_a, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                a = torch.roll(shifted_a, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                a = shifted_a
            a = a.view(B_a, H * W, C)
            # x = shortcut + self.drop_path(x)
            a = shortcut_a + a

            # FFN
            an = self.norm2(a)
            an = self.mlp(an)
            a = a + an + self.S_Adapter_Audio(an) # self.drop_path(0.5 * )

            x = (v, a)

            # todo multimodal without fusion, cascaded spatial and temporal MSHA, and parallelly-connected adapter with FFN
            '''
            H, W = self.input_resolution
            v, a = x[0], x[1]
            B_v, L_v, C = v.shape
            B_a, L_a, _ = a.shape

            assert L_v == H * W, "input feature has wrong size"
            assert L_a == H * W, "input feature has wrong size"

            # todo for visual branch
            ## Temporal attention
            if self.t_attn:
                v = rearrange(v, '(b t) n c -> (b n) t c', t=self.num_frames, n=L_v, c=C)
                res_temporal_v = self.attn(self.norm1(v), temporal=True)
                res_temporal_v = self.T_Adapter(res_temporal_v)
                v = v + self.drop_path(res_temporal_v)
                v = rearrange(v, '(b n) t c -> (b t) n c', t=self.num_frames, n=L_v, c=C)

            shortcut_v = v
            v = self.norm1(v)
            v = v.view(B_v, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_v = v

            # partition windows
            v_windows = window_partition(shifted_v, self.window_size)  # nW*B, window_size, window_size, C
            v_windows = v_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows_v = self.attn(v_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            attn_windows_v = self.S_Adapter2(attn_windows_v)

            # merge windows
            attn_windows_v = attn_windows_v.view(-1, self.window_size, self.window_size, C)
            shifted_v = window_reverse(attn_windows_v, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                v = torch.roll(shifted_v, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                v = shifted_v
            v = v.view(B_v, H * W, C)
            # x = shortcut + self.drop_path(x)
            v = shortcut_v + v

            # FFN
            vn = self.norm2(v)
            v = v + self.mlp(vn) + self.drop_path(0.5 * self.S_Adapter(vn))

            # todo for audio branch
            ## Temporal attention
            if self.t_attn:
                a = rearrange(a, '(b t) n c -> (b n) t c', t=self.num_frames, n=L_a, c=C)
                res_temporal_a = self.attn(self.norm1(a), temporal=True, signal='audio')
                res_temporal_a = self.T_Adapter_Audio(res_temporal_a)
                a = a + self.drop_path(res_temporal_a)
                a = rearrange(a, '(b n) t c -> (b t) n c', t=self.num_frames, n=L_a, c=C)

            shortcut_a = a
            a = self.norm1(a)
            a = a.view(B_a, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_a = torch.roll(a, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_a = a

            # partition windows
            a_windows = window_partition(shifted_a, self.window_size)  # nW*B, window_size, window_size, C
            a_windows = a_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows_a = self.attn(a_windows, mask=self.attn_mask, signal='audio')  # nW*B, window_size*window_size, C
            attn_windows_a = self.S_Adapter2_Audio(attn_windows_a)

            # merge windows
            attn_windows_a = attn_windows_a.view(-1, self.window_size, self.window_size, C)
            shifted_a = window_reverse(attn_windows_a, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                a = torch.roll(shifted_a, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                a = shifted_a
            a = a.view(B_a, H * W, C)
            # x = shortcut + self.drop_path(x)
            a = shortcut_a + a

            # FFN
            an = self.norm2(a)
            a = a + self.mlp(an) + self.drop_path(0.5 * self.S_Adapter_Audio(an))

            x = (v, a)'''
            return x

        if self.mode == 'fusion_adapt':
            # todo finalized version
            H, W = self.input_resolution
            v, a = x[0], x[1]
            B_v, L_v, C = v.shape
            B_a, L_a, _ = a.shape

            assert L_v == H * W, "input feature has wrong size"
            assert L_a == H * W, "input feature has wrong size"

            # todo for visual branch
            ## Temporal attention
            if self.t_attn:
                v = rearrange(v, '(b t) n c -> (b n) t c', t=self.num_frames, n=L_v, c=C)
                res_temporal_v = self.attn(self.norm1(v), temporal=True)
                res_temporal_v = self.T_Adapter(res_temporal_v)
                v = v + self.drop_path(res_temporal_v)
                v = rearrange(v, '(b n) t c -> (b t) n c', t=self.num_frames, n=L_v, c=C)

                a = rearrange(a, '(b t) n c -> (b n) t c', t=self.num_frames, n=L_a, c=C)
                res_temporal_a = self.attn(self.norm1(a), temporal=True, signal='audio')
                res_temporal_a = self.T_Adapter_Audio(res_temporal_a)
                a = a + self.drop_path(res_temporal_a)
                a = rearrange(a, '(b n) t c -> (b t) n c', t=self.num_frames, n=L_a, c=C)

            shortcut_v = v
            v = self.norm1(v)
            v = v.view(B_v, H, W, C)

            shortcut_a = a
            a = self.norm1(a)
            a = a.view(B_a, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                shifted_a = torch.roll(a, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_v = v
                shifted_a = a

            # partition windows
            v_windows = window_partition(shifted_v, self.window_size)  # nW*B, window_size, window_size, C
            v_windows = v_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # partition windows
            a_windows = window_partition(shifted_a, self.window_size)  # nW*B, window_size, window_size, C
            a_windows = a_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows_v = self.attn(v_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            # W-MSA/SW-MSA
            attn_windows_a = self.attn(a_windows, mask=self.attn_mask, signal='audio')  # nW*B, window_size*window_size, C

            vs_hidden = self.S_Adapter2.act(self.S_Adapter2.D_fc1(attn_windows_v))  # [n, bt, d]
            as_hidden = self.S_Adapter2_Audio.act(self.S_Adapter2_Audio.D_fc1(attn_windows_a))

            vs_fuse = vs_hidden # [bt, nv, d]
            as_fuse = as_hidden # [bt, na, d]

            attn_vs = F.softmax(torch.bmm(vs_fuse, as_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na]
            a2v_res_s = torch.bmm(attn_vs, as_fuse)  # [bt, nv, d]

            attn_as = F.softmax(torch.bmm(as_fuse, vs_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
            v2a_res_s = torch.bmm(attn_as, vs_fuse)  # [bt, na, d]

            vs_hidden = vs_hidden + self.gate_v * a2v_res_s
            as_hidden = as_hidden + self.gate_a * v2a_res_s

            attn_windows_v = attn_windows_v + self.S_Adapter2.D_fc2(vs_hidden)
            attn_windows_a = attn_windows_a + self.S_Adapter2_Audio.D_fc2(as_hidden)

            # merge windows
            attn_windows_v = attn_windows_v.view(-1, self.window_size, self.window_size, C)
            shifted_v = window_reverse(attn_windows_v, self.window_size, H, W)  # B H' W' C

            # merge windows
            attn_windows_a = attn_windows_a.view(-1, self.window_size, self.window_size, C)
            shifted_a = window_reverse(attn_windows_a, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                v = torch.roll(shifted_v, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                a = torch.roll(shifted_a, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                v = shifted_v
                a = shifted_a

            v = v.view(B_v, H * W, C)
            # x = shortcut + self.drop_path(x)
            v = shortcut_v + v

            a = a.view(B_a, H * W, C)
            # x = shortcut + self.drop_path(x)
            a = shortcut_a + a

            # FFN
            vn = self.norm2(v)
            an = self.norm2(a)

            vn = self.mlp(vn)
            an = self.mlp(an)

            vn_hidden = self.S_Adapter.act(self.S_Adapter.D_fc1(vn))
            an_hidden = self.S_Adapter_Audio.act(self.S_Adapter_Audio.D_fc1(an))
            vn_fuse = vn_hidden
            an_fuse = an_hidden

            attn_vn = F.softmax(torch.bmm(vn_fuse, an_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na]
            a2v_res_n = torch.bmm(attn_vn, an_fuse)  # [bt, nv, d]

            attn_an = F.softmax(torch.bmm(an_fuse, vn_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
            v2a_res_n = torch.bmm(attn_an, vn_fuse)  # [bt, na, d]

            vn_hidden = vn_hidden + self.gate_v * a2v_res_n
            an_hidden = an_hidden + self.gate_a * v2a_res_n

            v = v + vn + self.S_Adapter.D_fc2(vn_hidden) # self.drop_path(0.5 * )
            a = a + an + self.S_Adapter_Audio.D_fc2(an_hidden) # self.drop_path(0.5 *)

            x = (v, a)

            # TODO Sep5: 82.0, only AV interaction in FNN-parallelly adapter
            '''
            H, W = self.input_resolution
            v, a = x[0], x[1]
            B_v, L_v, C = v.shape
            B_a, L_a, _ = a.shape

            assert L_v == H * W, "input feature has wrong size"
            assert L_a == H * W, "input feature has wrong size"

            # todo for visual branch
            ## Temporal attention
            if self.t_attn:
                v = rearrange(v, '(b t) n c -> (b n) t c', t=self.num_frames, n=L_v, c=C)
                res_temporal_v = self.attn(self.norm1(v), temporal=True)
                res_temporal_v = self.T_Adapter(res_temporal_v)
                v = v + self.drop_path(res_temporal_v)
                v = rearrange(v, '(b n) t c -> (b t) n c', t=self.num_frames, n=L_v, c=C)

                a = rearrange(a, '(b t) n c -> (b n) t c', t=self.num_frames, n=L_a, c=C)
                res_temporal_a = self.attn(self.norm1(a), temporal=True, signal='audio')
                res_temporal_a = self.T_Adapter_Audio(res_temporal_a)
                a = a + self.drop_path(res_temporal_a)
                a = rearrange(a, '(b n) t c -> (b t) n c', t=self.num_frames, n=L_a, c=C)

            shortcut_v = v
            v = self.norm1(v)
            v = v.view(B_v, H, W, C)

            shortcut_a = a
            a = self.norm1(a)
            a = a.view(B_a, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                shifted_a = torch.roll(a, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_v = v
                shifted_a = a

            # partition windows
            v_windows = window_partition(shifted_v, self.window_size)  # nW*B, window_size, window_size, C
            v_windows = v_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # partition windows
            a_windows = window_partition(shifted_a, self.window_size)  # nW*B, window_size, window_size, C
            a_windows = a_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows_v = self.attn(v_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            attn_windows_v = self.S_Adapter2(attn_windows_v)
            # W-MSA/SW-MSA
            attn_windows_a = self.attn(a_windows, mask=self.attn_mask, signal='audio')  # nW*B, window_size*window_size, C
            attn_windows_a = self.S_Adapter2_Audio(attn_windows_a)

            # merge windows
            attn_windows_v = attn_windows_v.view(-1, self.window_size, self.window_size, C)
            shifted_v = window_reverse(attn_windows_v, self.window_size, H, W)  # B H' W' C
            # merge windows
            attn_windows_a = attn_windows_a.view(-1, self.window_size, self.window_size, C)
            shifted_a = window_reverse(attn_windows_a, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                v = torch.roll(shifted_v, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                a = torch.roll(shifted_a, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                v = shifted_v
                a = shifted_a

            v = v.view(B_v, H * W, C)
            # x = shortcut + self.drop_path(x)
            v = shortcut_v + v

            a = a.view(B_a, H * W, C)
            # x = shortcut + self.drop_path(x)
            a = shortcut_a + a

            # FFN
            vn = self.norm2(v)
            an = self.norm2(a)

            #vn = self.mlp(vn)
            #an = self.mlp(an)

            vn_hidden = self.S_Adapter.act(self.S_Adapter.D_fc1(vn))
            an_hidden = self.S_Adapter_Audio.act(self.S_Adapter_Audio.D_fc1(an))
            vn_fuse = vn_hidden
            an_fuse = an_hidden

            attn_vn = F.softmax(torch.bmm(vn_fuse, an_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na]
            a2v_res_n = torch.bmm(attn_vn, an_fuse)  # [bt, nv, d]

            attn_an = F.softmax(torch.bmm(an_fuse, vn_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
            v2a_res_n = torch.bmm(attn_an, vn_fuse)  # [bt, na, d]

            vn_hidden = vn_hidden + self.gate_v * a2v_res_n
            an_hidden = an_hidden + self.gate_a * v2a_res_n

            v = v + self.mlp(vn) + self.drop_path(0.5 * self.S_Adapter.D_fc2(vn_hidden))
            a = a + self.mlp(an) + self.drop_path(0.5 * self.S_Adapter_Audio.D_fc2(an_hidden))

            x = (v, a)'''
            return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        T = self.num_frames
        ## just count the FLOPs of q@k and attn@v
        # norm1
        # flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        if self.t_attn:
            flops += nW * self.attn.flops(self.window_size * self.window_size, T, temporal=True)
        flops += nW * self.attn.flops(self.window_size * self.window_size, T, temporal=False)
        # mlp
        # flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        # flops += self.dim * H * W
        return flops

# downsample the resolution of input embedding
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops = 0
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, num_frames, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 adapter_mlp_ratio=0.25, mode='video_adapt'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.mode = mode

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_frames=num_frames,
                                 num_heads=num_heads, window_size=window_size,
                                 t_attn=True if (i % 2 == 0) else False,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 adapter_mlp_ratio=adapter_mlp_ratio,
                                 mode=mode)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        #TODO save output of each layer for multiscale
        x_before_downsample = x

        if self.downsample is not None:
            if self.mode == 'video_adapt' or self.mode == 'audio_adapt':
                x = self.downsample(x)
            else:
                (v, a) = x[0], x[1]
                v = self.downsample(v)
                a = self.downsample(a)
                x = (v, a)
        return x, x_before_downsample

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        img_size = to_2tuple(img_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        patches_resolution = [img_size[0] // patch_size[1], img_size[1] // patch_size[2]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        B, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        x = rearrange(x, 'b c d h w -> (b d) (h w) c')

        return x, B, D

    def flops(self):
        return 0

class SwinTransformer2D_Adapter_AVS(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrained=None, img_size=224, patch_size=[1, 4, 4], num_frames=5, in_chans=3,
                 embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                 window_size=7, mlp_ratio=4., frozen_stages=-1, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, t_relative=True,
                 use_checkpoint=False, ftmode='videoonly', adapter_mlp_ratio = [0.25, 0.25, 0.25, 0.25],
                 channel=256, opt=None, config=None, vis_dim=[64, 128, 320, 512], tpavi_stages=[0, 1, 2, 3],
                 tpavi_vv_flag=False, tpavi_va_flag=True, **kwargs):
        super().__init__()

        self.tpavi_stages = tpavi_stages
        self.tpavi_vv_flag = tpavi_vv_flag
        self.tpavi_va_flag = tpavi_va_flag
        self.vis_dim = vis_dim
        ###################################

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) # 8*C feature dim in the last block
        self.mlp_ratio = mlp_ratio
        self.pretrained = pretrained
        self.num_frames = num_frames
        self.frozen_stages = frozen_stages
        self.patch_size = patch_size
        self.t_relative = t_relative
        self.ftmode = ftmode

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                        norm_layer=norm_layer if self.patch_norm else None)

        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution # [224//4, 224//4]= [56, 56]
        self.num_Ttokens = num_frames // patch_size[0] # 10

        # todo newly added: patch_embed for audio input
        self.f_dim, self.t_dim = self.get_shape_a(patch_size=patch_size, input_fdim=img_size,
                                                  input_tdim=img_size)
        num_patches_audio = self.f_dim * self.t_dim
        patches_resolution_audio = [self.f_dim, self.t_dim]
        self.patch_embed_audio = PatchEmbed3D(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim,
                                              norm_layer=norm_layer if self.patch_norm else None)
        self.num_patches_audio = num_patches_audio
        self.patches_resolution_audio = patches_resolution_audio
        self.num_Ttokens_audio = num_frames // patch_size[0]

        assert patches_resolution[0] == patches_resolution_audio[0]
        assert patches_resolution[1] == patches_resolution_audio[1]
        assert self.num_Ttokens == self.num_Ttokens_audio

        # absolute position embedding
        # if self.ape:
        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)

        ## temporal embedding
        if not self.t_relative:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_Ttokens, embed_dim))
            trunc_normal_(self.temporal_embedding, std=.02)

            self.temporal_embedding_audio = nn.Parameter(torch.zeros(1, self.num_Ttokens_audio, embed_dim))
            trunc_normal_(self.temporal_embedding, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule


        if self.ftmode == 'videoonly':
            # build layers
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                     patches_resolution[1] // (2 ** i_layer)),
                                   num_frames=self.num_Ttokens,
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint,
                                   adapter_mlp_ratio=adapter_mlp_ratio[i_layer],
                                   mode = 'video_adapt')
                self.layers.append(layer)

        if self.ftmode == 'audioonly':
            # build layers
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   input_resolution=(patches_resolution_audio[0] // (2 ** i_layer),
                                                     patches_resolution_audio[1] // (2 ** i_layer)),
                                   num_frames=self.num_Ttokens_audio,
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint,
                                   adapter_mlp_ratio=adapter_mlp_ratio[i_layer],
                                   mode='audio_adapt')
                self.layers.append(layer)

        if self.ftmode == 'multimodal':
            # build layers
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                     patches_resolution[1] // (2 ** i_layer)),
                                   num_frames=self.num_Ttokens_audio,
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint,
                                   adapter_mlp_ratio=adapter_mlp_ratio[i_layer],
                                   mode='multimodal_adapt_no_fusion')
                self.layers.append(layer)

        if self.ftmode == 'fusion':
            # build layers
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                     patches_resolution[1] // (2 ** i_layer)),
                                   num_frames=self.num_Ttokens_audio,
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint,
                                   adapter_mlp_ratio=adapter_mlp_ratio[i_layer],
                                   mode='fusion_adapt')
                self.layers.append(layer)

        if self.ftmode not in ['audioonly', 'videoonly', 'multimodal', 'fusion']:
            raise TypeError('ftmode is not expected !!!')

        ## final temporal blocks
        dim = int(embed_dim * 2 ** (self.num_layers - 1))

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        '''
        #todo Dropout ?  nn.LayerNorm(self.num_features*2),nn.Dropout(0.5),
        # TODO comment if the task is AVS
        if self.ftmode == 'multimodal' or self.ftmode == 'fusion':
            self.mlp_head = nn.Sequential(nn.Linear(self.num_features * 2, 512),
                                          nn.Dropout(0.5),
                                          nn.Linear(512, label_dim))
        else:
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.num_features),
                                          nn.Linear(self.num_features, label_dim))
        '''
        ############################################################
        # todo newly added and check
        self.avstask_conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel,
                                           self.vis_dim[3])
        self.avstask_conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel,
                                           self.vis_dim[2])
        self.avstask_conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel,
                                           self.vis_dim[1])
        self.avstask_conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel,
                                           self.vis_dim[0])

        self.avstask_path4 = FeatureFusionBlock(channel)
        self.avstask_path3 = FeatureFusionBlock(channel)
        self.avstask_path2 = FeatureFusionBlock(channel)
        self.avstask_path1 = FeatureFusionBlock(channel)

        self.avstask_x1_linear = nn.Linear(192, 64)  # todo input dimension 192
        self.avstask_x2_linear = nn.Linear(384, 128)  # todo input dimension 384
        self.avstask_x3_linear = nn.Linear(768, 320)  # todo input dimension 768
        self.avstask_x4_linear = nn.Linear(1536, 512)  # todo input dimension 1536

        self.avstask_audio_linear = nn.Linear(1536, 128)

        for i in self.tpavi_stages:
            setattr(self, f"avstask_tpavi_b{i + 1}", TPAVIModule(in_channels=channel, mode='dot'))
            print("==> Build TPAVI block...")

        self.avstask_output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        ################################################################################
        #self.apply(self._init_weights)
        self.initialize_weights(pretrained=self.pretrained)
        self._freeze_stages()

    def pre_reshape_for_tpavi(self, x):
        # x: [B*5, C, H, W]
        _, C, H, W = x.shape
        try:
            x = x.reshape(-1, 5, C, H, W)
        except:
            print("pre_reshape_for_tpavi: ", x.shape)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        return x

    def post_reshape_for_tpavi(self, x):
        # x: [B, C, T, H, W]
        # return: [B*T, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.view(-1, C, H, W)
        return x

    def tpavi_vv(self, x, stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'avstask_tpavi_b{stage + 1}')
        x = self.pre_reshape_for_tpavi(x)  # [B, C, T, H, W]
        x, _ = tpavi_b(x)  # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x)  # [B*T, C, H, W]
        return x

    def tpavi_va(self, x, audio, stage):
        # x: visual, [B*T, C=256, H, W]
        # audio: [B*T, 128]
        # ra_flag: return audio feature list or not
        tpavi_b = getattr(self, f'avstask_tpavi_b{stage + 1}')
        try:
            audio = audio.view(-1, 5, audio.shape[-1])  # [B, T, 128]
        except:
            print("tpavi_va: ", audio.shape)
        x = self.pre_reshape_for_tpavi(x)  # [B, C, T, H, W]
        x, a = tpavi_b(x, audio)  # [B, C, T, H, W], [B, T, C]
        x = self.post_reshape_for_tpavi(x)  # [B*T, C, H, W]
        return x, a

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def get_shape_a(self, patch_size=(2, 4, 4), input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, self.num_frames, input_fdim, input_tdim)
        test_proj = nn.Conv3d(1, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        test_out = test_proj(test_input)
        f_dim = test_out.shape[-2]
        t_dim = test_out.shape[-1]
        return f_dim, t_dim

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def initialize_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            print(f'load model from: {self.pretrained}')

            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model']

            # todo change back Conv2D in patch embed?
            state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1,self.patch_size[0], 1, 1) / \
                                                    self.patch_size[0]
            # TODO sum, mean Sep4
            state_dict['patch_embed_audio.proj.weight'] = torch.nn.Parameter(torch.mean(state_dict['patch_embed.proj.weight'].unsqueeze(2), dim=1))
            state_dict['patch_embed_audio.proj.bias'] = state_dict['patch_embed.proj.bias']
            state_dict['patch_embed_audio.norm.weight'] = state_dict['patch_embed.norm.weight']
            state_dict['patch_embed_audio.norm.bias'] = state_dict['patch_embed.norm.bias']

            '''
            # initialize the audio branch with original visual branch
            if self.ftmode == 'audioonly' or self.ftmode == 'multimodal' or self.ftmode == 'fusion':
                oritinal_keys = list(state_dict.keys())
                # print(old_keys)
                for param_key in oritinal_keys:
                    if 'layers' in param_key:
                        # print(param_key)
                        # print(weights[param_key])
                        new_key = param_key.replace('layers', 'layers_audio')
                        # print(new_atten_t_key)
                        state_dict[new_key] = state_dict[param_key]
            '''
            ## Duplicate weights for temporal attention and temporal norm
            # new_state_dict = state_dict.copy()
            # for key in state_dict:
            #     if 'blocks' in key and 'attn' in key and 'relative' not in key and 'mask' not in key:
            #         new_key = key.replace('attn','temporal_attn')
            #         if not new_key in state_dict:
            #             new_state_dict[new_key] = state_dict[key]
            #         else:
            #             new_state_dict[new_key] = state_dict[new_key]
            #     if 'blocks' in key and 'norm1' in key and 'relative' not in key and 'mask' not in key:
            #         new_key = key.replace('norm1','temporal_norm')
            #         if not new_key in state_dict:
            #             new_state_dict[new_key] = state_dict[key]
            #         else:
            #             new_state_dict[new_key] = state_dict[new_key]
            # state_dict = new_state_dict
            msg = self.load_state_dict(state_dict, strict=False)
            print('Missing keys: {}'.format(msg.missing_keys))
            print('Unexpected keys: {}'.format(msg.unexpected_keys))
            print(f"=> loaded successfully '{self.pretrained}'")
            del checkpoint
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


        for n, m in self.layers.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.layers.named_modules():
            if 'S_Adapter2' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.layers.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.layers.named_modules():
            if 'S_Adapter_Audio' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.layers.named_modules():
            if 'S_Adapter2_Audio' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.layers.named_modules():
            if 'T_Adapter_Audio' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding', 'temporal_embedding_audio'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, a, v, mode):
        if mode == 'videoonly':
            v, B, T = self.patch_embed(v)  # x in shape (BT, HW, C)

            ## Add temporal embedding
            if not self.t_relative:
                v = rearrange(v, '(b t) n c -> (b n) t c', b=B, t=T)
                v = v + self.temporal_embedding
                v = rearrange(v, '(b n) t c -> (b t) n c', b=B, t=T)

            v = self.pos_drop(v)

            for layer in self.layers:
                v = layer(v)

            v = self.norm(v) # (B T) (H W) C
            #print(v.shape)
            ## Final temporal block
            H = self.layers[-1].input_resolution[0] # (BT, HW, C); H and W is the resolution at the last layer
            #v = rearrange(v, '(B T) (H W) C -> B C T H W', T=T, H=H)
            v = rearrange(v, 'BT HW C -> BT C HW')
            v = self.avgpool(v) # [BT, C, 1]
            v = v.view(v.shape[0], -1) # [BT, C]
            v = self.mlp_head(v)  # [b*t, class_num]
            return v

        elif mode == 'audioonly':
            a = a.unsqueeze(1)  # [B, 1, T, L, H]
            a, B, T = self.patch_embed_audio(a)  # x in shape (BT, HW, C)
            ## Add temporal embedding
            if not self.t_relative:
                a = rearrange(a, '(b t) n c -> (b n) t c', b=B, t=T)
                a = a + self.temporal_embedding_audio
                a = rearrange(a, '(b n) t c -> (b t) n c', b=B, t=T)

            a = self.pos_drop(a)

            for layer in self.layers:
                a = layer(a)

            a = self.norm(a)  # (B T) (H W) C
            ## Final temporal block
            H = self.layers_audio[-1].input_resolution[0]
            #a = rearrange(a, '(B T) (H W) C -> B C T H W', T=T, H=H)
            a = rearrange(a, 'BT HW C -> BT C HW')
            a = self.avgpool(a)  # [BT, C, 1]
            a = a.view(a.shape[0], -1)  # [BT, C]
            a = self.mlp_head(a)  # [b*t, class_num]
            return a

        elif mode == 'multimodal':
            v, B, T = self.patch_embed(v)  # x in shape (BT, HW, C)
            a = a.unsqueeze(1)  # [B, 1, T, L, H]
            a, _, _ = self.patch_embed_audio(a)  # x in shape (BT, HW, C)
            ## Add temporal embedding
            if not self.t_relative:
                v = rearrange(v, '(b t) n c -> (b n) t c', b=B, t=T)
                v = v + self.temporal_embedding
                v = rearrange(v, '(b n) t c -> (b t) n c', b=B, t=T)

                a = rearrange(a, '(b t) n c -> (b n) t c', b=B, t=T)
                a = a + self.temporal_embedding_audio
                a = rearrange(a, '(b n) t c -> (b t) n c', b=B, t=T)

            v = self.pos_drop(v)
            a = self.pos_drop(a)

            for layer in self.layers:
                v, a = layer((v, a))

            v = self.norm(v)  # (B T) (H W) C
            a = self.norm(a)  # (B T) (H W) C

            v = rearrange(v, 'BT HW C -> BT C HW')
            v = self.avgpool(v)  # [BT, C, 1]
            v = v.view(v.shape[0], -1)  # [BT, C]

            a = rearrange(a, 'BT HW C -> BT C HW')
            a = self.avgpool(a)  # [BT, C, 1]
            a = a.view(a.shape[0], -1)  # [BT, C]

            out = torch.cat((a, v), dim=-1)  # [b*t, d*2]
            out = self.mlp_head(out)  # # [b*t, class_num]

            return out

        elif mode == 'fusion':
            #print(v.shape)
            #print(a.shape)
            v = rearrange(v, 'b t c h w -> b c t h w') # todo
            v, B, T = self.patch_embed(v)  # x in shape (BT, HW, C)

            a = a.unsqueeze(1)  # [B, 1, T, L, H]
            a, _, _ = self.patch_embed_audio(a)  # x in shape (BT, HW, C)
            ## Add temporal embedding
            if not self.t_relative:
                v = rearrange(v, '(b t) n c -> (b n) t c', b=B, t=T)
                v = v + self.temporal_embedding
                v = rearrange(v, '(b n) t c -> (b t) n c', b=B, t=T)

                a = rearrange(a, '(b t) n c -> (b n) t c', b=B, t=T)
                a = a + self.temporal_embedding_audio
                a = rearrange(a, '(b n) t c -> (b t) n c', b=B, t=T)

            v = self.pos_drop(v) # [bt, n, c]
            a = self.pos_drop(a) # [bt, n, c]
            x = (v, a)
            multi_scale = [] # TODO newly added

            for idx, layer in enumerate(self.layers):
                # x = (v, a)
                x, x_before_downsample = layer(x)
                # x_before_downsample = (v_before_downsample, a_before_downsample)
                if idx != len(self.layers)-1:
                    #print(x_before_downsample[0].shape)
                    multi_scale.append(x_before_downsample[0]) # append v_before_downsample of each layer except for final layer
                else:
                    multi_scale.append(self.norm(x_before_downsample[0]))
            v, a = x

            v = self.norm(v)  # (B T) (H W) C; [bt, n, c]
            a = self.norm(a)  # (B T) (H W) C; [bt, n, c]

            v = rearrange(v, 'BT HW C -> BT C HW')
            v = self.avgpool(v)  # [BT, C, 1]
            v = v.view(v.shape[0], -1)  # [BT, C]

            a = rearrange(a, 'BT HW C -> BT C HW')
            a = self.avgpool(a)  # [BT, C, 1]
            a = a.view(a.shape[0], -1)  # [BT, C]

            audio_feature = rearrange(a, '(b t) c->b t c', b=B, t=T) # [b, t, c]
            audio_feature = self.avstask_audio_linear(audio_feature)

            x1 = multi_scale[0].view(multi_scale[0].size(0), 56, 56, -1)  # todo token length = XX * XX
            x2 = multi_scale[1].view(multi_scale[1].size(0), 28, 28, -1)  # todo
            x3 = multi_scale[2].view(multi_scale[2].size(0), 14, 14, -1)  # todo
            x4 = multi_scale[3].view(multi_scale[3].size(0), 7, 7, -1)  # todo

            x1 = self.avstask_x1_linear(x1) # [b, 56, 56, 64]
            x2 = self.avstask_x2_linear(x2) # [b, 28, 28, 128]
            x3 = self.avstask_x3_linear(x3) # [b, 14, 14, 320]
            x4 = self.avstask_x4_linear(x4) # [b, 7, 7, 512]

            x1 = rearrange(x1, 'BF w h c -> BF c w h') # [b, 64, 56, 56]
            x2 = rearrange(x2, 'BF w h c -> BF c w h')
            x3 = rearrange(x3, 'BF w h c -> BF c w h')
            x4 = rearrange(x4, 'BF w h c -> BF c w h')

            '''
            # interpolation 
            x1 = F.interpolate(rearrange(x1, 'BF w h c -> BF c w h'), mode='bicubic', size=[56, 56])
            x2 = F.interpolate(rearrange(x2, 'BF w h c -> BF c w h'), mode='bicubic', size=[28, 28])
            x3 = F.interpolate(rearrange(x3, 'BF w h c -> BF c w h'), mode='bicubic', size=[14, 14])
            x4 = F.interpolate(rearrange(x4, 'BF w h c -> BF c w h'), mode='bicubic', size=[7, 7])'''

            conv1_feat = self.avstask_conv1(x1)  # BF x 256 x 56 x 56
            conv2_feat = self.avstask_conv2(x2)  # BF x 256 x 28 x 28
            conv3_feat = self.avstask_conv3(x3)  # BF x 256 x 14 x 14
            conv4_feat = self.avstask_conv4(x4)  # BF x 256 x  7 x  7

            feature_map_list = [conv1_feat, conv2_feat, conv3_feat, conv4_feat]
            a_fea_list = [None] * 4

            if len(self.tpavi_stages) > 0:
                if (not self.tpavi_vv_flag) and (not self.tpavi_va_flag):
                    raise Exception('tpavi_vv_flag and tpavi_va_flag cannot be False at the same time if len(tpavi_stages)>0, \
            					tpavi_vv_flag is for video self-attention while tpavi_va_flag indicates the standard version (audio-visual attention)')
                for i in self.tpavi_stages:
                    tpavi_count = 0
                    conv_feat = torch.zeros_like(feature_map_list[i]).to(feature_map_list[i].device)
                    if self.tpavi_vv_flag:
                        conv_feat_vv = self.tpavi_vv(feature_map_list[i], stage=i)
                        conv_feat += conv_feat_vv
                        tpavi_count += 1
                    if self.tpavi_va_flag:  # todo pass here
                        conv_feat_va, a_fea = self.tpavi_va(feature_map_list[i], audio_feature, stage=i)
                        conv_feat += conv_feat_va
                        tpavi_count += 1
                        a_fea_list[i] = a_fea
                    conv_feat /= tpavi_count
                    feature_map_list[i] = conv_feat  # update features of stage-i which conduct non-local

            conv4_feat = self.avstask_path4(feature_map_list[3])  # BF x 256 x 14 x 14
            conv43 = self.avstask_path3(conv4_feat, feature_map_list[2])  # BF x 256 x 28 x 28
            conv432 = self.avstask_path2(conv43, feature_map_list[1])  # BF x 256 x 56 x 56
            conv4321 = self.avstask_path1(conv432, feature_map_list[0])  # BF x 256 x 112 x 112

            pred = self.avstask_output_conv(conv4321)  # BF x 1 x 224 x 224
            # print(pred.shape)
            return pred, feature_map_list, a_fea_list

