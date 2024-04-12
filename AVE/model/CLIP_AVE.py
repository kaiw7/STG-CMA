from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from einops import rearrange, repeat
import clip
import loratorch as lora
import torch.nn.functional as F
# 0.03125: lightv2, Tiny
# 0.0625: light, Base
# 0.125: vanilla
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.0625, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1.,
                 num_tadapter=1, num_frames=8, drop_path=0., mode='videoonly', enable_fusion=False):
        super().__init__()
        self.mode = mode
        self.num_tadapter = num_tadapter
        self.enable_fusion = enable_fusion

        self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head
        self.scale = scale

        if self.mode == 'video_adapt' or self.mode == 'multimodal_adapt_no_fusion' or self.mode == 'fusion_adapt':
            self.MLP_Adapter = Adapter(d_model, skip_connect=False) # TODO Sep 15, comment if only audio adaptation
            self.S_Adapter = Adapter(d_model) # TODO Sep 15, comment if only audio adaptation
            self.T_Adapter = Adapter(d_model, skip_connect=False) # TODO Sep 15, comment if only audio adaptation
            if num_tadapter == 2:
                self.t_adapter_in = Adapter(d_model)

        # todo for only visual adaptation in multimodal setting
        if self.mode == 'audio_adapt' or self.mode == 'multimodal_adapt_no_fusion' or self.mode == 'fusion_adapt':
            self.S_Adapter_Audio = Adapter(d_model) # TODO Sep 15, comment if only visual adaptation
            self.MLP_Adapter_Audio = Adapter(d_model, skip_connect=False) # TODO Sep 15, comment if only visual adaptation
            self.T_Adapter_Audio = Adapter(d_model, skip_connect=False) # TODO Sep 15, comment if only visual adaptation

        #if self.mode == 'fusion_adapt' and self.enable_fusion:

            '''
            self.Adapter_AV_S = Adapter(d_model)  # spatial
            self.Adapter_AV_T = Adapter(d_model, skip_connect=False)  # spatial
            self.MLP_Adapter_AV = Adapter(d_model, skip_connect=False) # joint
            '''
            #self.Adapter_AV = Adapter(d_model) # spatial
            #self.MLP_Adapter_AV = Adapter(d_model, skip_connect=False)
            #self.MLP_Adapter_V2A = Adapter(d_model, skip_connect=False)
            #self.MLP_Adapter_A2V = Adapter(d_model, skip_connect=False)

        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        #self.ln_before_v0 = nn.LayerNorm(d_model)
        #self.ln_before_a0 = nn.LayerNorm(d_model)
        #self.ln_before_fusion = nn.LayerNorm(d_model)
        #self.ln_before_v = nn.LayerNorm(d_model)
        #self.ln_before_a = nn.LayerNorm(d_model)
        self.gate_v = nn.Parameter(torch.zeros(1))
        self.gate_a = nn.Parameter(torch.zeros(1))

        #self.my_tokens_v = nn.Parameter(torch.rand(2, d_model))
        #self.my_tokens_a = nn.Parameter(torch.rand(2, d_model))

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):

        if self.mode == 'video_adapt':
            # todo before Sep 8
            '''
            ## x shape [HW+1, BT, D]
            n, bt, d = x.shape
            ## temporal adaptation
            xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
            xt = self.T_Adapter(self.attention(self.ln_1(xt)))
            xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
            x = x + self.drop_path(xt)
            ## spatial adaptation
            x = x + self.S_Adapter(self.attention(self.ln_1(x)))
            ## joint adaptation
            xn = self.ln_2(x)
            x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))'''

            # todo visual-only branch
            ## x shape [HW+1, BT, D]
            n, bt, d = x.shape
            ## temporal adaptation
            xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
            xt = self.T_Adapter(self.attention(self.ln_1(xt)))
            xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
            x = x + self.drop_path(xt)
            ## spatial adaptation
            x = x + self.S_Adapter(self.attention(self.ln_1(x)))
            ## joint adaptation
            xn = self.ln_2(x)
            xn = self.mlp(xn)
            x = x + xn + self.MLP_Adapter(xn) # self.drop_path(self.scale * )

            return x

        elif self.mode == 'audio_adapt':
            '''
            ## x shape [HW+1, BT, D]
            n, bt, d = x.shape
            ## temporal adaptation
            xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
            xt = self.T_Adapter_Audio(self.attention(self.ln_1(xt)))
            xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
            x = x + self.drop_path(xt)
            ## spatial adaptation
            x = x + self.S_Adapter_Audio(self.attention(self.ln_1(x)))
            ## joint adaptation
            xn = self.ln_2(x)
            x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter_Audio(xn))'''

            # todo audio-only branch
            ## x shape [HW+1, BT, D]
            n, bt, d = x.shape
            ## temporal adaptation
            xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
            xt = self.T_Adapter_Audio(self.attention(self.ln_1(xt)))
            xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
            x = x + self.drop_path(xt)
            ## spatial adaptation
            x = x + self.S_Adapter_Audio(self.attention(self.ln_1(x)))
            ## joint adaptation
            xn = self.ln_2(x)
            xn = self.mlp(xn)
            x = x + xn + self.MLP_Adapter_Audio(xn) #self.drop_path(self.scale * )
            return x

        elif self.mode == 'multimodal_adapt_no_fusion':
            '''
            v, a = x[0], x[1]

            ## v shape [HW+1, BT, D] Visual branch TODO
            nv, bt, d = v.shape

            ## temporal adaptation
            vt = rearrange(v, 'n (b t) d -> t (b n) d', t=self.num_frames)
            vt = self.T_Adapter(self.attention(self.ln_1(vt)))
            vt = rearrange(vt, 't (b n) d -> n (b t) d', n=nv)
            v = v + self.drop_path(vt)
            ## spatial adaptation
            v = v + self.S_Adapter(self.attention(self.ln_1(v)))
            ## joint adaptation
            vn = self.ln_2(v)
            v = v + self.mlp(vn) + self.drop_path(self.scale * self.MLP_Adapter(vn))

            # a shape [HW+1, BT, D] audio branch
            na = a.shape[0]

            ## temporal adaptation
            at = rearrange(a, 'n (b t) d -> t (b n) d', t=self.num_frames)
            at = self.T_Adapter_Audio(self.attention(self.ln_1(at)))
            at = rearrange(at, 't (b n) d -> n (b t) d', n=na)
            a = a + self.drop_path(at)
            ## spatial adaptation
            a = a + self.S_Adapter_Audio(self.attention(self.ln_1(a)))
            ## joint adaptation
            an = self.ln_2(a)
            a = a + self.mlp(an) + self.drop_path(self.scale * self.MLP_Adapter_Audio(an))

            x = (v, a)'''

            # todo: our multimodal setting

            v, a = x[0], x[1]

            ## v shape [HW+1, BT, D] Visual branch TODO
            nv, bt, d = v.shape

            ## temporal adaptation
            vt = rearrange(v, 'n (b t) d -> t (b n) d', t=self.num_frames)
            vt = self.T_Adapter(self.attention(self.ln_1(vt)))
            vt = rearrange(vt, 't (b n) d -> n (b t) d', n=nv)
            v = v + self.drop_path(vt)
            ## spatial adaptation
            v = v + self.S_Adapter(self.attention(self.ln_1(v)))
            ## joint adaptation
            vn = self.ln_2(v)
            vn = self.mlp(vn)
            v = v + vn + self.MLP_Adapter(vn) # self.drop_path(self.scale * )

            # a shape [HW+1, BT, D] audio branch
            na = a.shape[0]

            ## temporal adaptation
            at = rearrange(a, 'n (b t) d -> t (b n) d', t=self.num_frames)
            at = self.T_Adapter_Audio(self.attention(self.ln_1(at)))
            at = rearrange(at, 't (b n) d -> n (b t) d', n=na)
            a = a + self.drop_path(at)
            ## spatial adaptation
            a = a + self.S_Adapter_Audio(self.attention(self.ln_1(a)))
            ## joint adaptation
            an = self.ln_2(a)
            an = self.mlp(an)
            a = a + an + self.MLP_Adapter_Audio(an) # self.drop_path(self.scale * )
            x = (v, a)

            # TODO: only having visual adaptation in multimodal setting
            '''
            v, a = x[0], x[1]

            ## v shape [HW+1, BT, D] Visual branch TODO
            nv, bt, d = v.shape

            ## temporal adaptation
            vt = rearrange(v, 'n (b t) d -> t (b n) d', t=self.num_frames)
            vt = self.T_Adapter(self.attention(self.ln_1(vt)))
            vt = rearrange(vt, 't (b n) d -> n (b t) d', n=nv)
            v = v + self.drop_path(vt)
            ## spatial adaptation
            v = v + self.S_Adapter(self.attention(self.ln_1(v)))
            ## joint adaptation
            vn = self.ln_2(v)
            vn = self.mlp(vn)
            v = v + vn + self.MLP_Adapter(vn)  # self.drop_path(self.scale * )

            # a shape [HW+1, BT, D] audio branch
            na = a.shape[0]

            ## temporal adaptation
            at = rearrange(a, 'n (b t) d -> t (b n) d', t=self.num_frames)
            #at = self.T_Adapter_Audio(self.attention(self.ln_1(at))) #todo
            at = self.attention(self.ln_1(at))
            at = rearrange(at, 't (b n) d -> n (b t) d', n=na)
            a = a + self.drop_path(at)
            ## spatial adaptation
            #a = a + self.S_Adapter_Audio(self.attention(self.ln_1(a))) #todo
            a = a + self.attention(self.ln_1(a))
            ## joint adaptation
            an = self.ln_2(a)
            an = self.mlp(an)
            a = a + an #+ self.MLP_Adapter_Audio(an)  # self.drop_path(self.scale * )

            x = (v, a)'''

            # TODO: only having audio adaptation in multimodal setting

            '''
            v, a = x[0], x[1]
            ## v shape [HW+1, BT, D] Visual branch TODO
            nv, bt, d = v.shape
            ## temporal adaptation
            vt = rearrange(v, 'n (b t) d -> t (b n) d', t=self.num_frames)
            #vt = self.T_Adapter(self.attention(self.ln_1(vt)))
            vt = self.attention(self.ln_1(vt)) # todo Sep 15
            vt = rearrange(vt, 't (b n) d -> n (b t) d', n=nv)
            v = v + self.drop_path(vt)
            ## spatial adaptation
            #v = v + self.S_Adapter(self.attention(self.ln_1(v)))
            v = v + self.attention(self.ln_1(v)) # todo Sep 15
            ## joint adaptation
            vn = self.ln_2(v)
            vn = self.mlp(vn)
            # todo Sep 15
            v = v + vn #+ self.MLP_Adapter(vn)  # self.drop_path(self.scale * )

            # a shape [HW+1, BT, D] audio branch
            na = a.shape[0]

            ## temporal adaptation
            at = rearrange(a, 'n (b t) d -> t (b n) d', t=self.num_frames)
            at = self.T_Adapter_Audio(self.attention(self.ln_1(at)))
            at = rearrange(at, 't (b n) d -> n (b t) d', n=na)
            a = a + self.drop_path(at)
            ## spatial adaptation
            a = a + self.S_Adapter_Audio(self.attention(self.ln_1(a)))
            ## joint adaptation
            an = self.ln_2(a)
            an = self.mlp(an)
            a = a + an + self.MLP_Adapter_Audio(an)  # self.drop_path(self.scale * )
            x = (v, a)'''
            return x

        elif self.mode == 'fusion_adapt':

            '''
            v, a = x[0], x[1]
            v_ori, a_ori = v, a

            ## TODO v shape [N_v, BT, D] Visual branch
            nv, bt, d = v.shape

            ## temporal adaptation
            vt = rearrange(v, 'n (b t) d -> t (b n) d', t=self.num_frames)
            vt = self.T_Adapter(self.attention(self.ln_1(vt)))
            vt = rearrange(vt, 't (b n) d -> n (b t) d', n=nv)
            v = v + self.drop_path(vt)
            ## spatial adaptation
            v = v + self.S_Adapter(self.attention(self.ln_1(v)))
            ## joint adaptation
            vn = self.ln_2(v)
            v = v + self.mlp(vn) + self.drop_path(self.scale * self.MLP_Adapter(vn))

            # TODO a shape [N_a, BT, D] audio branch
            na = a.shape[0]

            ## temporal adaptation
            at = rearrange(a, 'n (b t) d -> t (b n) d', t=self.num_frames)
            at = self.T_Adapter_Audio(self.attention(self.ln_1(at)))
            at = rearrange(at, 't (b n) d -> n (b t) d', n=na)
            a = a + self.drop_path(at)
            ## spatial adaptation
            a = a + self.S_Adapter_Audio(self.attention(self.ln_1(a)))
            ## joint adaptation
            an = self.ln_2(a)
            a = a + self.mlp(an) + self.drop_path(self.scale * self.MLP_Adapter_Audio(an))

            # visual branch: [nv, bt, d]
            # audio branch: [na, bt, d]
            '''

            # TODO Aug 31 Cross-Modal Adapter, our final fusion version !!!

            v, a = x[0], x[1]

            ## TODO v shape [N_v, BT, D] Visual branch
            nv, bt, d = v.shape
            # TODO a shape [N_a, BT, D] audio branch
            na = a.shape[0]

            ## temporal adaptation for visual
            vt = rearrange(v, 'n (b t) d -> t (b n) d', t=self.num_frames)
            vt = self.T_Adapter(self.attention(self.ln_1(vt)))
            vt = rearrange(vt, 't (b n) d -> n (b t) d', n=nv)
            v = v + self.drop_path(vt)
            ## temporal adaptation for audio
            at = rearrange(a, 'n (b t) d -> t (b n) d', t=self.num_frames)
            at = self.T_Adapter_Audio(self.attention(self.ln_1(at)))
            at = rearrange(at, 't (b n) d -> n (b t) d', n=na)
            a = a + self.drop_path(at)

            ## spatial adaptation for visual
            vs =self.attention(self.ln_1(v))
            vs_hidden = self.S_Adapter.act(self.S_Adapter.D_fc1(vs)) # [n, bt, d]

            a_s = self.attention(self.ln_1(a))
            as_hidden = self.S_Adapter_Audio.act(self.S_Adapter_Audio.D_fc1(a_s))

            vs_fuse = rearrange(vs_hidden, 'n bt d -> bt n d')
            as_fuse = rearrange(as_hidden, 'n bt d -> bt n d')

            attn_vs = F.softmax(torch.bmm(vs_fuse, as_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na]
            a2v_res_s = torch.bmm(attn_vs, as_fuse)  # [bt, nv, d]
            a2v_res_s = rearrange(a2v_res_s, 'bt n d -> n bt d')

            attn_as = F.softmax(torch.bmm(as_fuse, vs_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
            v2a_res_s = torch.bmm(attn_as, vs_fuse)  # [bt, na, d]
            v2a_res_s = rearrange(v2a_res_s, 'bt n d -> n bt d')

            vs_hidden = vs_hidden + self.gate_v * a2v_res_s
            as_hidden = as_hidden + self.gate_a * v2a_res_s

            v = v + vs + self.S_Adapter.D_fc2(vs_hidden)
            a = a + a_s + self.S_Adapter_Audio.D_fc2(as_hidden)

            ## joint adaptation for visual
            vn = self.ln_2(v)
            vn = self.mlp(vn)
            vn_hidden = self.MLP_Adapter.act(self.MLP_Adapter.D_fc1(vn))

            an = self.ln_2(a)
            an = self.mlp(an)
            an_hidden = self.MLP_Adapter_Audio.act(self.MLP_Adapter_Audio.D_fc1(an))

            vn_fuse = rearrange(vn_hidden, 'n bt d -> bt n d')
            an_fuse = rearrange(an_hidden, 'n bt d -> bt n d')

            attn_vn = F.softmax(torch.bmm(vn_fuse, an_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na]
            a2v_res_n = torch.bmm(attn_vn, an_fuse)  # [bt, nv, d]
            a2v_res_n = rearrange(a2v_res_n, 'bt n d -> n bt d')

            attn_an = F.softmax(torch.bmm(an_fuse, vn_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
            v2a_res_n = torch.bmm(attn_an, vn_fuse)  # [bt, na, d]
            v2a_res_n = rearrange(v2a_res_n, 'bt n d -> n bt d')

            vn_hidden = vn_hidden + self.gate_v * a2v_res_n
            an_hidden = an_hidden + self.gate_a * v2a_res_n

            v = v + vn + self.MLP_Adapter.D_fc2(vn_hidden)# self.drop_path(), self.scale *

            ## joint adaptation for audio
            a = a + an + self.MLP_Adapter_Audio.D_fc2(an_hidden)# self.drop_path(), self.scale *


            # TODO Sep 16, visual-to-audio direction
            '''
            v, a = x[0], x[1]

            ## TODO v shape [N_v, BT, D] Visual branch
            nv, bt, d = v.shape
            # TODO a shape [N_a, BT, D] audio branch
            na = a.shape[0]

            ## temporal adaptation for visual
            vt = rearrange(v, 'n (b t) d -> t (b n) d', t=self.num_frames)
            vt = self.T_Adapter(self.attention(self.ln_1(vt)))
            vt = rearrange(vt, 't (b n) d -> n (b t) d', n=nv)
            v = v + self.drop_path(vt)
            ## temporal adaptation for audio
            at = rearrange(a, 'n (b t) d -> t (b n) d', t=self.num_frames)
            at = self.T_Adapter_Audio(self.attention(self.ln_1(at)))
            at = rearrange(at, 't (b n) d -> n (b t) d', n=na)
            a = a + self.drop_path(at)

            ## spatial adaptation for visual
            vs = self.attention(self.ln_1(v))
            vs_hidden = self.S_Adapter.act(self.S_Adapter.D_fc1(vs))  # [n, bt, d]

            a_s = self.attention(self.ln_1(a))
            as_hidden = self.S_Adapter_Audio.act(self.S_Adapter_Audio.D_fc1(a_s))

            vs_fuse = rearrange(vs_hidden, 'n bt d -> bt n d')
            as_fuse = rearrange(as_hidden, 'n bt d -> bt n d')

            #attn_vs = F.softmax(torch.bmm(vs_fuse, as_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na] #todo
            #a2v_res_s = torch.bmm(attn_vs, as_fuse)  # [bt, nv, d] #todo
            #a2v_res_s = rearrange(a2v_res_s, 'bt n d -> n bt d') #todo

            attn_as = F.softmax(torch.bmm(as_fuse, vs_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
            v2a_res_s = torch.bmm(attn_as, vs_fuse)  # [bt, na, d]
            v2a_res_s = rearrange(v2a_res_s, 'bt n d -> n bt d')

            vs_hidden = vs_hidden  #+ self.gate_v * a2v_res_s #todo
            as_hidden = as_hidden + self.gate_a * v2a_res_s

            v = v + vs + self.S_Adapter.D_fc2(vs_hidden)
            a = a + a_s + self.S_Adapter_Audio.D_fc2(as_hidden)

            ## joint adaptation for visual
            vn = self.ln_2(v)
            vn = self.mlp(vn)
            vn_hidden = self.MLP_Adapter.act(self.MLP_Adapter.D_fc1(vn))

            an = self.ln_2(a)
            an = self.mlp(an)
            an_hidden = self.MLP_Adapter_Audio.act(self.MLP_Adapter_Audio.D_fc1(an))

            vn_fuse = rearrange(vn_hidden, 'n bt d -> bt n d')
            an_fuse = rearrange(an_hidden, 'n bt d -> bt n d')

            #attn_vn = F.softmax(torch.bmm(vn_fuse, an_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na] #todo
            #a2v_res_n = torch.bmm(attn_vn, an_fuse)  # [bt, nv, d] #todo
            #a2v_res_n = rearrange(a2v_res_n, 'bt n d -> n bt d') #todo

            attn_an = F.softmax(torch.bmm(an_fuse, vn_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
            v2a_res_n = torch.bmm(attn_an, vn_fuse)  # [bt, na, d]
            v2a_res_n = rearrange(v2a_res_n, 'bt n d -> n bt d')

            vn_hidden = vn_hidden #+ self.gate_v * a2v_res_n #todo
            an_hidden = an_hidden + self.gate_a * v2a_res_n

            v = v + vn + self.MLP_Adapter.D_fc2(vn_hidden)  # self.drop_path(), self.scale *

            ## joint adaptation for audio
            a = a + an + self.MLP_Adapter_Audio.D_fc2(an_hidden)  # self.drop_path(), self.scale *
            '''
            # TODO Sep 16, audio-to-visual direction
            '''
            v, a = x[0], x[1]

            ## TODO v shape [N_v, BT, D] Visual branch
            nv, bt, d = v.shape
            # TODO a shape [N_a, BT, D] audio branch
            na = a.shape[0]

            ## temporal adaptation for visual
            vt = rearrange(v, 'n (b t) d -> t (b n) d', t=self.num_frames)
            vt = self.T_Adapter(self.attention(self.ln_1(vt)))
            vt = rearrange(vt, 't (b n) d -> n (b t) d', n=nv)
            v = v + self.drop_path(vt)
            ## temporal adaptation for audio
            at = rearrange(a, 'n (b t) d -> t (b n) d', t=self.num_frames)
            at = self.T_Adapter_Audio(self.attention(self.ln_1(at)))
            at = rearrange(at, 't (b n) d -> n (b t) d', n=na)
            a = a + self.drop_path(at)

            ## spatial adaptation for visual
            vs = self.attention(self.ln_1(v))
            vs_hidden = self.S_Adapter.act(self.S_Adapter.D_fc1(vs))  # [n, bt, d]

            a_s = self.attention(self.ln_1(a))
            as_hidden = self.S_Adapter_Audio.act(self.S_Adapter_Audio.D_fc1(a_s))

            vs_fuse = rearrange(vs_hidden, 'n bt d -> bt n d')
            as_fuse = rearrange(as_hidden, 'n bt d -> bt n d')

            attn_vs = F.softmax(torch.bmm(vs_fuse, as_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na] #todo
            a2v_res_s = torch.bmm(attn_vs, as_fuse)  # [bt, nv, d] #todo
            a2v_res_s = rearrange(a2v_res_s, 'bt n d -> n bt d') #todo

            #attn_as = F.softmax(torch.bmm(as_fuse, vs_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
            #v2a_res_s = torch.bmm(attn_as, vs_fuse)  # [bt, na, d]
            #v2a_res_s = rearrange(v2a_res_s, 'bt n d -> n bt d')

            vs_hidden = vs_hidden + self.gate_v * a2v_res_s #todo
            as_hidden = as_hidden #+ self.gate_a * v2a_res_s

            v = v + vs + self.S_Adapter.D_fc2(vs_hidden)
            a = a + a_s + self.S_Adapter_Audio.D_fc2(as_hidden)

            ## joint adaptation for visual
            vn = self.ln_2(v)
            vn = self.mlp(vn)
            vn_hidden = self.MLP_Adapter.act(self.MLP_Adapter.D_fc1(vn))

            an = self.ln_2(a)
            an = self.mlp(an)
            an_hidden = self.MLP_Adapter_Audio.act(self.MLP_Adapter_Audio.D_fc1(an))

            vn_fuse = rearrange(vn_hidden, 'n bt d -> bt n d')
            an_fuse = rearrange(an_hidden, 'n bt d -> bt n d')

            attn_vn = F.softmax(torch.bmm(vn_fuse, an_fuse.permute(0, 2, 1)), dim=-1)  # [bt nv na] #todo
            a2v_res_n = torch.bmm(attn_vn, an_fuse)  # [bt, nv, d] #todo
            a2v_res_n = rearrange(a2v_res_n, 'bt n d -> n bt d') #todo

            #attn_an = F.softmax(torch.bmm(an_fuse, vn_fuse.permute(0, 2, 1)), dim=-1)  # [bt na nv]
            #v2a_res_n = torch.bmm(attn_an, vn_fuse)  # [bt, na, d]
            #v2a_res_n = rearrange(v2a_res_n, 'bt n d -> n bt d')

            vn_hidden = vn_hidden  + self.gate_v * a2v_res_n #todo
            an_hidden = an_hidden #+ self.gate_a * v2a_res_n

            v = v + vn + self.MLP_Adapter.D_fc2(vn_hidden)  # self.drop_path(), self.scale *

            ## joint adaptation for audio
            a = a + an + self.MLP_Adapter_Audio.D_fc2(an_hidden)  # self.drop_path(), self.scale *
            '''

            '''
            ## spatial adaptation
            v = v + self.S_Adapter(self.attention(self.ln_1(v)))
            ## joint adaptation
            vn = self.ln_2(v)
            v = v + self.mlp(vn) + self.drop_path(self.scale * self.MLP_Adapter(vn))

            
            ## spatial adaptation
            a = a + self.S_Adapter_Audio(self.attention(self.ln_1(a)))
            ## joint adaptation
            an = self.ln_2(a)
            a = a + self.mlp(an) + self.drop_path(self.scale * self.MLP_Adapter_Audio(an))
            '''

            # TODO Aug 31
            '''
            # visual branch: [nv, bt, d]
            # audio branch: [na, bt, d]
            v_fuse = rearrange(v, 'n bt d -> bt n d')
            a_fuse = rearrange(a, 'n bt d -> bt n d')

            #v_fuse = rearrange(v_ori, 'n bt d -> bt n d')  # [bt, nv, d]
            #a_fuse = rearrange(a_ori, 'n bt d -> bt n d')  # [bt, na, d]

            attn_v = F.softmax(torch.bmm(v_fuse, a_fuse.permute(0, 2, 1)), dim=-1) # [bt nv na]
            a2v_res = torch.bmm(attn_v, a_fuse) # [bt, nv, d]
            a2v_res = rearrange(a2v_res, 'bt n d -> n bt d')

            attn_a = F.softmax(torch.bmm(a_fuse, v_fuse.permute(0, 2, 1)), dim=-1) # [bt na nv]
            v2a_res = torch.bmm(attn_a, v_fuse) # [bt, na, d]
            v2a_res = rearrange(v2a_res, 'bt n d -> n bt d')

            v = v + self.gate_v * a2v_res
            a = a + self.gate_a * v2a_res
            '''
            '''
            if self.enable_fusion:
            # todo option
                v_ori = rearrange(v_ori, 'n bt d -> bt n d') # [bt, n1, d]
                a_ori = rearrange(a_ori, 'n bt d -> bt n d') # [bt, n2, d]

                rep_token_v = repeat(self.my_tokens_v, 't d -> b t d', b=v_ori.size(0))  # [bt, 2, d]
                attn_v_token = F.softmax(torch.bmm(rep_token_v, v_ori.permute(0, 2, 1)), dim=-1)  # [bt, 2, n1]
                attn_v_token = rep_token_v + torch.bmm(attn_v_token, v_ori)  # [b, 2, d]

                attn_a = F.softmax(torch.bmm(a_ori, attn_v_token.permute(0, 2, 1)), dim=-1)  # [bt, n2, 2]
                attn_a_res = torch.bmm(attn_a, attn_v_token)  # [bt, n2, d]

                a_res = a_ori + self.gate_a * attn_a_res # [bt, n2, d]
                a_res = self.MLP_Adapter_A2V(self.ln_before_a(a_res))
                a_res = rearrange(a_res, 'bt n d -> n bt d')  # [n2, bt, d]

                rep_token_a = repeat(self.my_tokens_a, 't d -> b t d', b=a_ori.size(0))  # [bt, 2, d]
                attn_a_token = F.softmax(torch.bmm(rep_token_a, a_ori.permute(0, 2, 1)), dim=-1)  # [bt, 2, n2]
                attn_a_token = rep_token_a + torch.bmm(attn_a_token, a_ori)  # [b, 2, d]

                attn_v = F.softmax(torch.bmm(v_ori, attn_a_token.permute(0, 2, 1)), dim=-1)  # [bt, n1, 2]
                attn_v_res = torch.bmm(attn_v, attn_a_token)  # [bt, n1, d]

                v_res = v_ori + self.gate_v * attn_v_res  # [bt, n1, d]
                v_res = self.MLP_Adapter_V2A(self.ln_before_v(v_res))
                v_res = rearrange(v_res, 'bt n d -> n bt d')  # [n2, bt, d]

                v = v + v_res
                a = a + a_res
            '''
            '''
            if self.enable_fusion:
                # todo option
                # v_ori: [N_v, bt, d], a_ori: [N_a, bt, d]
                av_ori = torch.cat((v_ori, a_ori), dim=0) # [N_v+N_a, bt, d]
                av_t_fusion = rearrange(av_ori, 'n (b t) d -> t (b n) d', t=self.num_frames)
                av_t_fusion = self.Adapter_AV_T(self.attention(self.ln_1(av_t_fusion)))
                av_t_fusion = rearrange(av_t_fusion, 't (b n) d -> n (b t) d', n=av_ori.shape[0])
                av_ori = av_ori + self.drop_path(av_t_fusion)

                av_s_fusion = av_ori + self.Adapter_AV_S(self.attention(self.ln_1(av_ori)))
                av_joint = self.ln_2(av_s_fusion)
                av_res = av_s_fusion + self.mlp(av_joint) + self.drop_path(self.scale * self.MLP_Adapter_AV(av_joint))

                v_res = av_res[:nv, :, :]  # [N_v, bt, d]
                a_res = av_res[nv:, :, :]  # [N_a, bt, d]
                v = v + self.gate_v * v_res
                a = a + self.gate_a * a_res
            '''

            '''
            # todo option1
            v_t_fusion = v_ori.mean(dim=0)  # [bt, d]
            a_t_fusion = a_ori.mean(dim=0)  # [bt, d]
            v_t_fusion = rearrange(v_t_fusion, '(b t) d -> t b d', t=self.num_frames)
            a_t_fusion = rearrange(a_t_fusion, '(b t) d -> t b d', t=self.num_frames)
            av_t_fusion = torch.cat((v_t_fusion, a_t_fusion), dim=0)  # [2t, b, d]
            av_t_fusion = av_t_fusion + self.drop_path(self.Adapter_AV_T(self.attention(self.ln_1(av_t_fusion))))  # [2t, b, d]
            v_t_fusion_out = av_t_fusion[:self.num_frames, :, :] # [t, b, d]
            a_t_fusion_out = av_t_fusion[self.num_frames:, :, :] # [t, b, d]
            v_t_fusion_out = rearrange(v_t_fusion_out, 't b d -> (b t) d', t=self.num_frames)
            a_t_fusion_out = rearrange(a_t_fusion_out, 't b d -> (b t) d', t=self.num_frames)

            v_s_fusion = v_ori + v_t_fusion_out.unsqueeze(dim=0) #[N_v, bt, d]
            a_s_fusion = a_ori + a_t_fusion_out.unsqueeze(dim=0) #[N_a, bt, d]
            #print(a_s_fusion)

            av_s_fusion = torch.cat((v_s_fusion, a_s_fusion), dim=0) # [N_v+N_a, bt, d]
            av_s_fusion = av_s_fusion + self.Adapter_AV_S(self.attention(self.ln_1(av_s_fusion))) # [N_v+N_a, bt, d]

            av_joint = self.ln_2(av_s_fusion)
            av_res = av_s_fusion + self.mlp(av_joint) + self.drop_path(self.scale * self.MLP_Adapter_AV(av_joint))

            v_res = av_res[:nv, :, :] # [N_v, bt, d]
            a_res = av_res[nv:, :, :] # [N_a, bt, d]
            #print(a_res)
            v = v + self.gate_v * v_res
            a = a + self.gate_a * a_res
            '''


            #v = self.ln_before_v0(v)
            #a = self.ln_before_a0(a)
            #print(v)

            x = (v, a)
            return x

class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 num_tadapter=1, scale=1., drop_path=0.1, mode='cascaded', enable_fusion_idx=6):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter,
                                                                num_frames, dpr[i], mode=mode,
                                                                enable_fusion=True if i >= enable_fusion_idx-1 else False) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class MM_CLIP_AVE(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, label_dim, input_resolution=224, audio_length=1024, num_video_frames=10, patch_size=16, embed_dim=768,
                 layers=12, heads=8, drop_path_rate=0.2, num_tadapter=1, adapter_scale=0.5, pretrained=None, ftmode='videoonly'):
        super().__init__()
        self.ftmode = ftmode
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.embed_dim = embed_dim
        self.ori_num_patches = (input_resolution // patch_size) ** 2
        self.oringal_hw = int(self.ori_num_patches ** 0.5)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = embed_dim ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim)) # class token for visual and audio
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, embed_dim)) # positional embed
        self.ln_pre = LayerNorm(embed_dim)

        # for audio signal
        # patch embedding for audio signals
        self.f_dim, self.t_dim = self.get_shape_a(fstride=patch_size, tstride=patch_size, input_fdim=128,
                                                  input_tdim=int(audio_length * (1/10)))
        self.num_patches_a = self.f_dim * self.t_dim
        self.conv1_audio = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
                               bias=False)  # # newly added
        self.positional_embedding_audio = nn.Parameter(scale * torch.randn(self.num_patches_a + 1, embed_dim))  # newly added


        self.num_video_frames = num_video_frames # num of video frames == num of audio clips

        # newly added
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_video_frames, embed_dim))

        # newly added
        self.temporal_embedding_audio = nn.Parameter(torch.zeros(1, num_video_frames, embed_dim))

        if self.ftmode == 'videoonly':
            self.transformer = Transformer(num_video_frames, embed_dim, layers, heads, num_tadapter=num_tadapter,
                                           scale=adapter_scale, drop_path=drop_path_rate, mode='video_adapt')

        elif self.ftmode == 'audioonly':
            self.transformer = Transformer(num_video_frames, embed_dim, layers, heads, num_tadapter=num_tadapter,
                                           scale=adapter_scale, drop_path=drop_path_rate, mode='audio_adapt')

        elif self.ftmode == 'multimodal':
            self.transformer = Transformer(num_video_frames, embed_dim, layers, heads, num_tadapter=num_tadapter,
                                           scale=adapter_scale, drop_path=drop_path_rate, mode='multimodal_adapt_no_fusion')


        elif self.ftmode == 'fusion':
            self.transformer = Transformer(num_video_frames, embed_dim, layers, heads, num_tadapter=num_tadapter,
                                           scale=adapter_scale, drop_path=drop_path_rate, mode='fusion_adapt',
                                           enable_fusion_idx=int(layers*2 / 3)) # int(layers*2 / 3) or 0

        else:
            raise TypeError('ftmode is not expected !!!')

        # newly added
        self.ln_post = LayerNorm(embed_dim)

        # mlp head
        # todo nn.LayerNorm(embed_dim*2), Sep5
        if self.ftmode == 'multimodal' or self.ftmode == 'fusion':
            self.mlp_head = nn.Sequential(nn.Linear(embed_dim * 2, 512),
                                          nn.Dropout(0.5),
                                          nn.Linear(512, label_dim))
        else:
            self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim),
                                          nn.Linear(embed_dim, label_dim))

        self.initialize_weights(pretrained=self.pretrained)
        print('Visual Positional Embedding Shape:', self.positional_embedding.shape)
        print('Temporal Positional Embedding Shape:', self.temporal_embedding.shape)
        print('Audio Positional Embedding Shape:', self.positional_embedding_audio.shape)

    def get_shape_a(self, fstride=16, tstride=16, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.embed_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

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
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                clip_model, preprocess = clip.load("ViT-B/16", device="cpu", download_root=self.pretrained)
            else:
                clip_model, preprocess = clip.load("ViT-L/14", device="cpu", download_root=self.pretrained)
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']

            pretrain_dict['conv1_audio.weight'] = torch.nn.Parameter(
                torch.sum(pretrain_dict['conv1.weight'], dim=1).unsqueeze(1))

            ori_pos_embed = pretrain_dict['positional_embedding']  # [196, 768]
            ori_pos_embed = ori_pos_embed.unsqueeze(dim=0) # [1, 196, 768]
            a_pos_embed = ori_pos_embed[:, 1:, :].detach().reshape(1, self.ori_num_patches, self.embed_dim).transpose(1, 2).reshape(
                1, self.embed_dim, self.oringal_hw, self.oringal_hw)

            if self.t_dim <= self.oringal_hw:
                a_pos_embed = a_pos_embed[:, :, :,
                              int(self.oringal_hw / 2) - int(self.t_dim / 2): int(self.oringal_hw / 2) - int(
                                  self.t_dim / 2) + self.t_dim]
            else:
                a_pos_embed = torch.nn.functional.interpolate(a_pos_embed, size=(self.oringal_hw, self.t_dim),
                                                              mode='bilinear')

            if self.f_dim <= self.oringal_hw:
                a_pos_embed = a_pos_embed[:, :,
                              int(self.oringal_hw / 2) - int(self.f_dim / 2): int(self.oringal_hw / 2) - int(
                                  self.f_dim / 2) + self.f_dim, :]
            else:
                a_pos_embed = torch.nn.functional.interpolate(a_pos_embed, size=(self.f_dim, self.t_dim),
                                                              mode='bilinear')
            a_pos_embed = a_pos_embed.reshape(1, self.embed_dim, self.num_patches_a).transpose(1, 2)

            pretrain_dict['positional_embedding_audio'] = nn.Parameter(torch.cat([ori_pos_embed[:, :1, :].detach(), a_pos_embed], dim=1).squeeze(dim=0))


            msg = self.load_state_dict(pretrain_dict, strict=False)
            print('Missing keys: {}'.format(msg.missing_keys))
            print('Unexpected keys: {}'.format(msg.unexpected_keys))
            print(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)


        # S_Adapter_Audio, MLP_Adapter_Audio, Adapter_AV, MLP_Adapter_AV, MLP_Adapter_V2A, MLP_Adapter_A2V
        for n, m in self.transformer.named_modules():
            if 'S_Adapter_Audio' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.transformer.named_modules():
            if 'T_Adapter_Audio' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.transformer.named_modules():
            if 'S_Adapter_in' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter_Audio' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.transformer.named_modules():
            if 'Adapter_AV' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
# Adapter_AV_S
        for n, m in self.transformer.named_modules():
            if 'Adapter_AV_S' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.transformer.named_modules():
            if 'Adapter_AV_T' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter_AV' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter_V2A' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter_A2V' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding', 'positional_embedding_audio', 'temporal_embedding_audio'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, a, v, mode):
        if mode == 'videoonly':
            B, C, T, H, W = v.shape
            v = rearrange(v, 'b c t h w -> (b t) c h w')
            v = self.conv1(v) #
            v = v.reshape(v.shape[0], v.shape[1], -1)
            v = v.permute(0, 2, 1)
            v = torch.cat([self.class_embedding.to(v.dtype) + torch.zeros(v.shape[0], 1, v.shape[-1], dtype=v.dtype, device=v.device), v], dim=1)
            v = v + self.positional_embedding.to(v.dtype)

            n = v.shape[1]
            v = rearrange(v, '(b t) n d -> (b n) t d', t=self.num_video_frames)
            v = v + self.temporal_embedding
            v = rearrange(v, '(b n) t d -> (b t) n d', n=n)

            v = self.ln_pre(v) # [b*t, N+1, D]

            v = v.permute(1, 0, 2)  # NLD -> LND  # [N+1, b*t, D]
            v = self.transformer(v) # [N+1, b*t, D]
            v = v.permute(1, 0, 2)  # [b*t, N+1, D]
            v = self.ln_post(v)  # [b*t, N+1, D]

            #x = x[:, 1:, :]  # [b*t N d]
            #x = x.mean(dim=1)  # [b*t, d]

            v = v[:, 0] # TODO, [b*t, d]

            v = self.mlp_head(v)  # [b*t, class_num]
            return v

        elif mode == 'audioonly':
            B, T, L, H = a.shape
            a = a.unsqueeze(1) # [B, 1, T, L, H]
            a = rearrange(a, 'b c t h w -> (b t) c h w')
            a = self.conv1_audio(a)
            a = a.reshape(a.shape[0], a.shape[1], -1)  # [B*T, D, N]
            a = a.permute(0, 2, 1)
            a = torch.cat([self.class_embedding.to(a.dtype) + torch.zeros(a.shape[0], 1, a.shape[-1], dtype=a.dtype,
                                                                          device=a.device), a], dim=1)
            a = a + self.positional_embedding_audio.to(a.dtype)  # [B*T, N+1, D]

            n = a.shape[1]
            a = rearrange(a, '(b t) n d -> (b n) t d', t=self.num_video_frames)
            a = a + self.temporal_embedding_audio
            a = rearrange(a, '(b n) t d -> (b t) n d', n=n)

            a = self.ln_pre(a)  # [B N+1, D]

            a = a.permute(1, 0, 2)  # [N+1, B*T, D]
            a = self.transformer(a)  # [N+1, B*T, D]
            a = a.permute(1, 0, 2)  # [B*T, N+1, D]
            a = self.ln_post(a)  # [B*T, N+1, D]

            a = a[:, 0, :]  # [b*T, d]

            a = self.mlp_head(a)
            return a

        elif mode == 'multimodal':
            # todo for visual
            B, C, T, H, W = v.shape
            v = rearrange(v, 'b c t h w -> (b t) c h w')
            v = self.conv1(v)  #
            v = v.reshape(v.shape[0], v.shape[1], -1)
            v = v.permute(0, 2, 1)
            v = torch.cat([self.class_embedding.to(v.dtype) + torch.zeros(v.shape[0], 1, v.shape[-1], dtype=v.dtype,
                                                                          device=v.device), v], dim=1)
            v = v + self.positional_embedding.to(v.dtype)

            n = v.shape[1]
            v = rearrange(v, '(b t) n d -> (b n) t d', t=self.num_video_frames)
            v = v + self.temporal_embedding
            v = rearrange(v, '(b n) t d -> (b t) n d', n=n)

            v = self.ln_pre(v)  # [b*t, N+1, D]
            v = v.permute(1, 0, 2)  # NLD -> LND  # [N+1, b*t, D]

            # todo for audio
            a = a.unsqueeze(1)  # [B, 1, T, L, H]
            a = rearrange(a, 'b c t h w -> (b t) c h w')
            a = self.conv1_audio(a)
            a = a.reshape(a.shape[0], a.shape[1], -1)  # [B*T, D, N]
            a = a.permute(0, 2, 1)
            a = torch.cat([self.class_embedding.to(a.dtype) + torch.zeros(a.shape[0], 1, a.shape[-1], dtype=a.dtype,
                                                                          device=a.device), a], dim=1)
            a = a + self.positional_embedding_audio.to(a.dtype)  # [B*T, N+1, D]

            na = a.shape[1]
            a = rearrange(a, '(b t) n d -> (b n) t d', t=self.num_video_frames)
            a = a + self.temporal_embedding_audio
            a = rearrange(a, '(b n) t d -> (b t) n d', n=na)

            a = self.ln_pre(a)  # [B N+1, D]
            a = a.permute(1, 0, 2)  # [N+1, B*T, D]

            v, a = self.transformer((v, a))  # [N+1, b*t, D]
            v = v.permute(1, 0, 2)  # [b*t, N+1, D]
            v = self.ln_post(v)  # [b, N+1, D]

            a = a.permute(1, 0, 2)  # [B*T, N+1, D]
            a = self.ln_post(a)  # [b, N+1, D]

            v = v[:, 0]  # TODO [b*t, D]
            a = a[:, 0]  # TODO [b*t, D]

            out = torch.cat((a, v), dim=-1)  # [b*t, d*2]

            out = self.mlp_head(out) # # [b*t, class_num]
            return out

        elif mode == 'fusion':
            # todo for visual
            B, C, T, H, W = v.shape
            v = rearrange(v, 'b c t h w -> (b t) c h w')
            v = self.conv1(v)  #
            v = v.reshape(v.shape[0], v.shape[1], -1)
            v = v.permute(0, 2, 1)
            v = torch.cat([self.class_embedding.to(v.dtype) + torch.zeros(v.shape[0], 1, v.shape[-1], dtype=v.dtype,
                                                                          device=v.device), v], dim=1)
            v = v + self.positional_embedding.to(v.dtype)

            n = v.shape[1]
            v = rearrange(v, '(b t) n d -> (b n) t d', t=self.num_video_frames)
            v = v + self.temporal_embedding
            v = rearrange(v, '(b n) t d -> (b t) n d', n=n)

            v = self.ln_pre(v)  # [b*t, N+1, D]
            v = v.permute(1, 0, 2)  # NLD -> LND  # [N+1, b*t, D]

            # todo for audio
            a = a.unsqueeze(1)  # [B, 1, T, L, H]
            a = rearrange(a, 'b c t h w -> (b t) c h w')
            a = self.conv1_audio(a)
            a = a.reshape(a.shape[0], a.shape[1], -1)  # [B*T, D, N]
            a = a.permute(0, 2, 1)
            a = torch.cat([self.class_embedding.to(a.dtype) + torch.zeros(a.shape[0], 1, a.shape[-1], dtype=a.dtype,
                                                                          device=a.device), a], dim=1)
            a = a + self.positional_embedding_audio.to(a.dtype)  # [B*T, N+1, D]

            na = a.shape[1]
            a = rearrange(a, '(b t) n d -> (b n) t d', t=self.num_video_frames)
            a = a + self.temporal_embedding_audio
            a = rearrange(a, '(b n) t d -> (b t) n d', n=na)

            a = self.ln_pre(a)  # [B N+1, D]
            a = a.permute(1, 0, 2)  # [N+1, B*T, D]

            v, a = self.transformer((v, a))  # [N+1, b*t, D]

            v = v.permute(1, 0, 2)  # [b*t, N+1, D]
            v = self.ln_post(v)  # [b, N+1, D]

            a = a.permute(1, 0, 2)  # [B*T, N+1, D]
            a = self.ln_post(a)  # [b, N+1, D]

            v = v[:, 0]  # TODO [b*t, D]
            a = a[:, 0]  # TODO [b*t, D]

            out = torch.cat((a, v), dim=-1)  # [b*t, d*2]

            out = self.mlp_head(out)  # # [b*t, class_num]
            return out
'''
ftmode = 'fusion'
pretrained = '/nobackup1/kai/research/multimodal/AudioVisual/pretrained_model/CLIP'

model = MM_CLIP(label_dim=101, input_resolution=224, audio_length=1024, num_video_frames=10, patch_size=16, embed_dim=768,
                 layers=12, heads=8, drop_path_rate=0.2, num_tadapter=1, adapter_scale=0.5, pretrained=pretrained,
                ftmode=ftmode)

video = torch.randn((1, 3, 10, 224, 224))
audio = torch.randn((1, 1024, 128))
output = model(a=audio, v=video, mode=ftmode)
print(output.shape)'''

'''
ftmode = 'multimodal'
pretrained = '/nobackup1/kai/research/multimodal/AudioVisual/pretrained_model/CLIP'

model = MM_CLIP_AVE(label_dim=101, input_resolution=224, audio_length=1024, num_video_frames=10, patch_size=14, embed_dim=1024,
                 layers=24, heads=16, drop_path_rate=0.2, num_tadapter=1, adapter_scale=0.5, pretrained=pretrained,
                ftmode=ftmode)

video = torch.randn((1, 3, 10, 224, 224))
audio = torch.randn((1, 10, 102, 128))
output = model(a=audio, v=video, mode=ftmode)
print(output.shape)'''
