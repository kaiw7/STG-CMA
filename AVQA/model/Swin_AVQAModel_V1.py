import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#from visual_net import resnet18

from ipdb import set_trace
import timm
from einops import rearrange, repeat
import timm
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import torch.nn.functional as F

def batch_organize(out_match_posi, out_match_nega):
    # audio B 512
    # posi B 512
    # nega B 512

    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0

    return out_match, batch_labels

# Question
class QstEncoder(nn.Module):
	def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

		super(QstEncoder, self).__init__()
		self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
		self.tanh = nn.Tanh()
		self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
		self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

	def forward(self, question):

		qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
		qst_vec = self.tanh(qst_vec)
		qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
		self.lstm.flatten_parameters()
		_, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
		qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
		qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
		qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
		qst_feature = self.tanh(qst_feature)
		qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

		return qst_feature


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

            v_nega = x[2] # todo newly added Dec 1
            B_v_nega, L_v_nage, _ = v_nega.shape  # todo newly added Dec 1

            assert L_v == H * W, "input feature has wrong size"
            assert L_a == H * W, "input feature has wrong size"
            assert L_v_nage == H * W, "input feature has wrong size" # todo newly added Dec 1

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

            shortcut_v_nega = v_nega # todo newly added Dec 1
            v_nega = self.norm1(v_nega) # todo newly added Dec 1
            v_nega = v_nega.view(B_v_nega, H, W, C) # todo newly added Dec 1

            # cyclic shift
            if self.shift_size > 0:
                shifted_v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                shifted_a = torch.roll(a, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                shifted_v_nega = torch.roll(v_nega, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) # todo newly added Dec 1
            else:
                shifted_v = v
                shifted_a = a
                shifted_v_nega = v_nega # todo newly added Dec 1

            # partition windows
            v_windows = window_partition(shifted_v, self.window_size)  # nW*B, window_size, window_size, C
            v_windows = v_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # partition windows
            a_windows = window_partition(shifted_a, self.window_size)  # nW*B, window_size, window_size, C
            a_windows = a_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # partition windows todo newly added Dec 1
            v_nega_windows = window_partition(shifted_v_nega, self.window_size)  # nW*B, window_size, window_size, C; todo newly added Dec 1
            v_nega_windows = v_nega_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C; todo newly added Dec 1


            # W-MSA/SW-MSA
            attn_windows_v = self.attn(v_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            # W-MSA/SW-MSA
            attn_windows_a = self.attn(a_windows, mask=self.attn_mask, signal='audio')  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA todo newly added Dec 1
            attn_windows_v_nega = self.attn(v_nega_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C; todo newly added Dec 1

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

            # merge windows  todo newly added Dec 1
            attn_windows_v_nega = attn_windows_v_nega.view(-1, self.window_size, self.window_size, C) # todo newly added Dec 1
            shifted_v_nega = window_reverse(attn_windows_v_nega, self.window_size, H, W)  # B H' W' C; todo newly added Dec 1


            # reverse cyclic shift
            if self.shift_size > 0:
                v = torch.roll(shifted_v, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                a = torch.roll(shifted_a, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                v_nega = torch.roll(shifted_v_nega, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                v = shifted_v
                a = shifted_a
                v_nega = shifted_v_nega

            v = v.view(B_v, H * W, C)
            # x = shortcut + self.drop_path(x)
            v = shortcut_v + v

            a = a.view(B_a, H * W, C)
            # x = shortcut + self.drop_path(x)
            a = shortcut_a + a

            # todo newly added Nov 1
            v_nega = v_nega.view(B_v_nega, H * W, C) # todo newly added Nov 1
            v_nega = shortcut_v_nega + self.drop_path(v_nega) # todo newly added Nov 1
            # v_nega = shortcut_v_nega + v_nega # todo newly added Nov 1

            # FFN todo newly added Nov 1
            v_nega = v_nega + self.drop_path(self.mlp(self.norm2(v_nega))) # todo newly added Nov 1

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

            x = (v, a, v_nega)

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

        #TODO multiscale

        if self.downsample is not None:
            if self.mode == 'video_adapt' or self.mode == 'audio_adapt':
                x = self.downsample(x)
            else:
                '''
                (v, a) = x[0], x[1]
                v = self.downsample(v)
                a = self.downsample(a)
                x = (v, a)'''
                (v, a, v_nega) = x[0], x[1], x[2]
                v = self.downsample(v)
                a = self.downsample(a)
                v_nega = self.downsample(v_nega)
                x = (v, a, v_nega)
        return x

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

class SwinTransformer2D_Adapter_AVQA(nn.Module):
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

    def __init__(self, grounding_pretrained=None, pretrained=None, img_size=224, patch_size=[1, 4, 4], num_frames=10, in_chans=3,
                 embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                 window_size=7, mlp_ratio=4., frozen_stages=-1, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, t_relative=True,
                 use_checkpoint=False, ftmode='videoonly', adapter_mlp_ratio = [0.25, 0.25, 0.25, 0.25], **kwargs):
        super().__init__()


        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) # 8*C feature dim in the last block
        self.mlp_ratio = mlp_ratio
        self.pretrained = pretrained
        self.grounding_pretrained = grounding_pretrained
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
        #self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        #todo Dropout ?  nn.LayerNorm(self.num_features*2),nn.Dropout(0.5),
        '''
        if self.ftmode == 'multimodal' or self.ftmode == 'fusion':
            self.mlp_head = nn.Sequential(nn.Linear(self.num_features * 2, 512),
                                          nn.Dropout(0.5),
                                          nn.Linear(512, label_dim))
        else:
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.num_features),
                                          nn.Linear(self.num_features, label_dim))
            '''
        # TODO newly added
        #self.avqatask_fc_a1 = nn.Linear(768, 1536) # comment todo whether it is required? pre-train
        self.avqatask_fc_a2 = nn.Linear(1536, 1536) # todo pre-train Dec 10

        #self.avqatask_fc_a1_pure = nn.Linear(128, 512) # comment
        #self.avqatask_fc_a2_pure = nn.Linear(512, 512) # comment
        # self.visual_net = resnet18(pretrained=True)

        #self.avqatask_fc_v = nn.Linear(2048, 512) # comment
        # self.fc_v = nn.Linear(1536, 512)
        #self.avqatask_fc_st = nn.Linear(512, 512) # comment
        self.avqatask_fc_fusion = nn.Linear(1536+1536, 1536) # todo Dec 10
        #self.avqatask_fc = nn.Linear(1024, 512) # comment
        #self.avqatask_fc_aq = nn.Linear(512, 512) # comment
        #self.avqatask_fc_vq = nn.Linear(512, 512) # comment

        self.avqatask_linear11 = nn.Linear(1536, 1536) # todo Dec 10
        self.avqatask_dropout1 = nn.Dropout(0.1) # todo Dec 10
        self.avqatask_linear12 = nn.Linear(1536, 1536) # todo Dec 10

        self.avqatask_linear21 = nn.Linear(1536, 1536) # todo Dec 10
        self.avqatask_dropout2 = nn.Dropout(0.1) # todo Dec 10
        self.avqatask_linear22 = nn.Linear(1536, 1536) # todo Dec 10
        self.avqatask_norm1 = nn.LayerNorm(1536) # todo Dec 10
        self.avqatask_norm2 = nn.LayerNorm(1536) # todo Dec 10
        self.avqatask_dropout3 = nn.Dropout(0.1) # todo Dec 10
        self.avqatask_dropout4 = nn.Dropout(0.1) # todo Dec 10
        #self.avqatask_norm3 = nn.LayerNorm(512) # comment

        self.avqatask_attn_a = nn.MultiheadAttention(1536, 4, dropout=0.1) # todo Dec 10
        self.avqatask_attn_v = nn.MultiheadAttention(1536, 4, dropout=0.1) # todo Dec 10

        # question
        self.avqatask_question_encoder = QstEncoder(93, 1536, 1536, 1, 1536) # todo Dec 10

        self.avqatask_tanh = nn.Tanh() # todo Dec 10
        #self.avqatask_dropout = nn.Dropout(0.5) # comment
        self.avqatask_fc_ans = nn.Linear(1536, 42) # todo Dec 10

        self.avqatask_avgpool = nn.AdaptiveAvgPool2d((1, 1)) # todo Dec 10
        self.avqatask_fc_gl = nn.Linear(1536+1536, 1536) # todo pre-train Dec 10

        # combine
        self.avqatask_fc1 = nn.Linear(1536+1536, 512) # todo pre-train Dec 10
        #self.avqatask_relu1 = nn.ReLU() # comment
        self.avqatask_fc2 = nn.Linear(512, 256) # todo pre-train Dec 10
        #self.avqatask_relu2 = nn.ReLU() # comment
        self.avqatask_fc3 = nn.Linear(256, 128) # todo pre-train Dec 10
        #self.avqatask_relu3 = nn.ReLU() # comment
        self.avqatask_fc4 = nn.Linear(128, 2) # todo pre-train Dec 10
        #self.avqatask_relu4 = nn.ReLU() # comment

        #self.avqatask_yb_fc_v = nn.Linear(1536, 512) # todo Dec 10
        #self.avqatask_yb_fc_a = nn.Linear(1536, 768) # todo Dec 10

        #self.apply(self._init_weights)
        self.initialize_weights(pretrained=self.pretrained, grounding_pretrained=self.grounding_pretrained)
        self._freeze_stages()

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

    def initialize_weights(self, pretrained=None, grounding_pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained and grounding_pretrained:
            self.pretrained = pretrained
            self.grounding_pretrained = grounding_pretrained # todo newly added
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            print(f'load model from: {self.pretrained}')
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model']

            ############################################################
            print(f'load grounding pretraining model from: {self.grounding_pretrained}') # todo newly added
            grounding_checkpoint = torch.load(self.grounding_pretrained, map_location='cpu')
            #print(self.state_dict().keys())
            tmp = ['module.fc_a1.weight', 'module.fc_a1.bias', 'module.fc_a2.weight', 'module.fc_a2.bias',
                   'module.fc_gl.weight',
                   'module.fc_gl.bias', 'module.fc1.weight', 'module.fc1.bias', 'module.fc2.weight', 'module.fc2.bias',
                   'module.fc3.weight', 'module.fc3.bias', 'module.fc4.weight', 'module.fc4.bias']

            tmp2 = ['module.fc_a1.weight', 'module.fc_a1.bias', 'module.fc_a2.weight', 'module.fc_a2.bias']
            pretrained_dict1 = {k: v for k, v in grounding_checkpoint.items() if k in tmp}
            pretrained_dict2 = {str(k).split('.')[0] + '.' + str(k).split('.')[1] + '_pure.' + str(k).split('.')[-1]: v
                                for k, v in grounding_checkpoint.items() if k in tmp2}

            for param_key in pretrained_dict1.keys():
                new_key = param_key.replace('module.', 'avqatask_')
                state_dict[new_key] = pretrained_dict1[param_key]

            for param_key in pretrained_dict2.keys():
                new_key = param_key.replace('module.', 'avqatask_')
                state_dict[new_key] = pretrained_dict2[param_key]


##################################################################################################


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

    # a: audio; v: visual_posi,
    def forward(self, a, v, v_nega, question, mode):
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
            # v: [B, C, T, H, W]
            v = rearrange(v, 'b t c h w -> b c t h w')  # todo
            v, BS, TT = self.patch_embed(v)  # x in shape (BT, HW, C)
            a = a.unsqueeze(1)  # [B, 1, T, L, H]
            a, _, _ = self.patch_embed_audio(a)  # x in shape (BT, HW, C)

            # todo v_nega: [B, C, T, H, W]
            v_nega = rearrange(v_nega, 'b t c h w -> b c t h w')  # todo
            v_nega, B_nega, T_nega = self.patch_embed(v_nega)  # x in shape (BT, HW, C); todo newly added Nov 1

            ## Add temporal embedding
            if not self.t_relative:
                v = rearrange(v, '(b t) n c -> (b n) t c', b=BS, t=TT)
                v = v + self.temporal_embedding
                v = rearrange(v, '(b n) t c -> (b t) n c', b=BS, t=TT)

                a = rearrange(a, '(b t) n c -> (b n) t c', b=BS, t=TT)
                a = a + self.temporal_embedding_audio
                a = rearrange(a, '(b n) t c -> (b t) n c', b=BS, t=TT)

            v = self.pos_drop(v)
            a = self.pos_drop(a)
            v_nega = self.pos_drop(v_nega) # todo newly added Nov 1

            for layer in self.layers:
                v, a, v_nega = layer((v, a, v_nega)) # todo newly added Nov 1

            f_v = self.norm(v)  # (B T) (H W) C
            f_a = self.norm(a)  # (B T) (H W) C
            visual_nega = self.norm(v_nega) # todo newly added Nov 1

            #f_v = self.avqatask_yb_fc_v(f_v) # todo [bt, n, 1536] --> [bt, n, 512]
            #f_a = self.avqatask_yb_fc_a(f_a) # todo [bt, n, 1536] --> [bt, n, 768]

            '''
            # todo feed v_nega
            with torch.no_grad():

                visual_nega = rearrange(v_nega, 'b t c h w -> (b t) c h w')
                visual_nega = self.swin.forward_features(visual_nega)'''
            ############################################

            #visual_nega = self.avqatask_yb_fc_v(visual_nega) # todo [bt, n, 1536] --> [bt, n, 512]
            #print(f_v.shape)
            visual_posi = rearrange(f_v, '(b t) (h w) c -> b t c h w', b=BS, t=TT, h=7, w=7) # todo [b, 10, 512, 7, 7]
            visual_nega = rearrange(visual_nega, '(b t) (h w) c -> b t c h w', b=B_nega, t=T_nega, h=7, w=7) # todo [b, 10, 512, 7, 7]

            ### -------> yb: cvpr use
            f_a = f_a.mean(dim=1) # todo [bt, 768]
            audio = rearrange(f_a, '(b t) c -> b t c', b=BS, t=TT) # todo [b, 10, 768]
            ### <-----

            ## question features
            qst_feature = self.avqatask_question_encoder(question)
            xq = qst_feature.unsqueeze(0)

            ## audio features  [2*B*T, 128]
            #audio_feat = F.relu(self.avqatask_fc_a1(audio)) # todo to see whether it is required? [b, 10, 1536]
            #print(audio.shape) #[b, t, 512]
            audio_feat = F.relu(audio)  # todo [b, 10, 512]
            audio_feat = self.avqatask_fc_a2(audio_feat) # todo [b, 10, 1536]
            audio_feat_pure = audio_feat # todo [b, 10, 1536]
            B, T, C = audio_feat.size()  # [B, T, C]
            audio_feat = audio_feat.view(B * T, C)  # [B*T, C] todo [b*10, 1536]

            ## visual posi [2*B*T, C, H, W]
            B, T, C, H, W = visual_posi.size() # todo [b, 10, 1536, 7, 7]
            temp_visual = visual_posi.view(B * T, C, H, W)  # [B*T, C, H, W] todo [b*10, 1536, 7, 7]
            v_feat = self.avqatask_avgpool(temp_visual)  # [B*T, C, 1, 1] # todo [b*10, 1536, 1, 1]
            visual_feat_before_grounding_posi = v_feat.squeeze()  # [B*T, C] todo [bt, 1536]

            (B, C, H, W) = temp_visual.size()
            v_feat = temp_visual.view(B, C, H * W)  # [B*T, C, HxW] todo [b*10, 1536, 49]
            v_feat = v_feat.permute(0, 2, 1)  # [B, HxW, C] todo [b*10, 49, 1536]
            visual_feat_posi = nn.functional.normalize(v_feat, dim=2)  # [B, HxW, C] todo [b*10, 49, 1536]

            ## audio-visual grounding posi
            audio_feat_aa = audio_feat.unsqueeze(-1)  # [B*T, C, 1] todo [b*10, 1536, 1]
            audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)  # [B*T, C, 1] todo [b*10, 1536, 1]

            x2_va = torch.matmul(visual_feat_posi, audio_feat_aa).squeeze()  # [B*T, HxW] todo [b*10, 49]

            x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)  # [B*T, 1, HxW] todo [b*10, 1, 49]
            visual_feat_grd = torch.matmul(x2_p, visual_feat_posi) # todo [b*10, 1, 1536]
            visual_feat_grd_after_grounding_posi = visual_feat_grd.squeeze()  # [B*T, C] todo [b*10, 1536]

            visual_gl = torch.cat((visual_feat_before_grounding_posi, visual_feat_grd_after_grounding_posi), dim=-1) # todo [b*10, 1536+1536]
            visual_feat_grd = self.avqatask_tanh(visual_gl)
            visual_feat_grd_posi = self.avqatask_fc_gl(visual_feat_grd)  # [B*T, C] todo [b*10, 1536]

            feat = torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)  # [B*T, C*2], [B*T, 1024] todo [b*10, 1536+1536]

            feat = F.relu(self.avqatask_fc1(feat))  # (1536+1536, 512)
            feat = F.relu(self.avqatask_fc2(feat))  # (512, 256)
            feat = F.relu(self.avqatask_fc3(feat))  # (256, 128)
            out_match_posi = self.avqatask_fc4(feat)  # (128, 2) # todo [b*10, 2]

            ###############################################################################################
            # visual nega
            B, T, C, H, W = visual_nega.size()
            temp_visual = visual_nega.view(B * T, C, H, W)
            v_feat = self.avqatask_avgpool(temp_visual)
            visual_feat_before_grounding_nega = v_feat.squeeze()  # [B*T, C]

            (B, C, H, W) = temp_visual.size()
            v_feat = temp_visual.view(B, C, H * W)  # [B*T, C, HxW]
            v_feat = v_feat.permute(0, 2, 1)  # [B, HxW, C]
            visual_feat_nega = nn.functional.normalize(v_feat, dim=2)

            ##### av grounding nega
            x2_va = torch.matmul(visual_feat_nega, audio_feat_aa).squeeze()
            x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)  # [B*T, 1, HxW]
            visual_feat_grd = torch.matmul(x2_p, visual_feat_nega)
            visual_feat_grd_after_grounding_nega = visual_feat_grd.squeeze()  # [B*T, C]

            visual_gl = torch.cat((visual_feat_before_grounding_nega, visual_feat_grd_after_grounding_nega), dim=-1)
            visual_feat_grd = self.avqatask_tanh(visual_gl)
            visual_feat_grd_nega = self.avqatask_fc_gl(visual_feat_grd)  # [B*T, C]

            # combine a and v
            feat = torch.cat((audio_feat, visual_feat_grd_nega), dim=-1)  # [B*T, C*2], [B*T, 1024]

            feat = F.relu(self.avqatask_fc1(feat))  # (1024, 512)
            feat = F.relu(self.avqatask_fc2(feat))  # (512, 256)
            feat = F.relu(self.avqatask_fc3(feat))  # (256, 128)
            out_match_nega = self.avqatask_fc4(feat)  # (128, 2) todo [b*10, 2]

            ###############################################################################################

            # out_match=None
            # match_label=None

            B = xq.shape[1]
            visual_feat_grd_be = visual_feat_grd_posi.view(B, -1, 1536)  # [B, T, 512]
            visual_feat_grd = visual_feat_grd_be.permute(1, 0, 2)

            ## attention, question as query on visual_feat_grd
            visual_feat_att = self.avqatask_attn_v(xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[
                0].squeeze(0)
            src = self.avqatask_linear12(self.avqatask_dropout1(F.relu(self.avqatask_linear11(visual_feat_att))))
            visual_feat_att = visual_feat_att + self.avqatask_dropout2(src)
            visual_feat_att = self.avqatask_norm1(visual_feat_att)

            # attention, question as query on audio
            audio_feat_be = audio_feat_pure.view(B, -1, 1536) # todo [b, 10, 512]
            audio_feat = audio_feat_be.permute(1, 0, 2)
            audio_feat_att = self.avqatask_attn_a(xq, audio_feat, audio_feat, attn_mask=None, key_padding_mask=None)[0].squeeze(
                0)
            src = self.avqatask_linear22(self.avqatask_dropout3(F.relu(self.avqatask_linear21(audio_feat_att))))
            audio_feat_att = audio_feat_att + self.avqatask_dropout4(src) # todo
            audio_feat_att = self.avqatask_norm2(audio_feat_att)

            feat = torch.cat((audio_feat_att + audio_feat_be.mean(dim=-2).squeeze(),
                              visual_feat_att + visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
            feat = self.avqatask_tanh(feat)
            feat = self.avqatask_fc_fusion(feat)

            ## fusion with question
            combined_feature = torch.mul(feat, qst_feature)
            combined_feature = self.avqatask_tanh(combined_feature)
            out_qa = self.avqatask_fc_ans(combined_feature)

        return out_qa, out_match_posi, out_match_nega
