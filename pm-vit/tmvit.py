import torch
import torch.nn as nn
import logging
from functools import partial
from timm.models.layers import DropPath, PatchEmbed, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import _cfg, checkpoint_filter_fn, _load_weights
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg, load_pretrained

_logger = logging.getLogger(__name__)

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import numpy as np
import math



default_cfgs = {'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
}


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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

    def flops(self, N):
        flops = 0
        (in_dim, mlp_dim) = self.fc1.weight.data.shape
        flops += N * in_dim * mlp_dim
        flops += N * mlp_dim * in_dim
        return flops


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_tokens=1, num_heads=8, channel=None, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = 64
        self.in_dim = dim
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, (num_heads * self.head_dim) * 3, bias=qkv_bias)
        # print(self.qkv.weight.data.shape)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_heads * self.head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if channel is None:
            self.channel = num_patches + num_tokens
        else:
            self.channel = channel

        self.merge_matrix = nn.Parameter(torch.eye(self.channel, num_patches + num_tokens))
        self.recover_matrix = nn.Parameter(torch.eye(num_patches + num_tokens, self.channel))
        # self.token_split = nn.Parameter(torch.range(1, self.channel + 1, dtype=int), requires_grad=False)
        self.token_mask = nn.Parameter(torch.ones(num_patches + num_tokens), requires_grad=False)
        self.bias = nn.Parameter(torch.ones(self.channel), requires_grad=False)
        # self.bias = None
        self.attn = None
        self.seq_ranks = None
        self.cnt = 0

    def forward(self, x, compute_taylor_attn=False, bias=None):
        B, N, C = x.shape
        # B,N,C --> B,N,3C --> B,N,3,h,dv --> 3,B,h,N,dv
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # B,h,N,dv
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # add bias
        # if self.bias is not None:
        #     attn = attn + self.bias

        attn = attn.softmax(dim=-1)

        if compute_taylor_attn:
            self.attn = attn

            attn.register_hook(self.compute_rank_attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)

        # if compute_taylor_attn:
        #     self.attn = x
        #     x.register_hook(self.compute_out)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def compute_rank_attn(self, grad):
        # grad = torch.abs(grad)
        # self.attn = torch.abs(self.attn)
        values = torch.sum(grad * self.attn, dim=0, keepdim=True) \
                      .sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)[0, 0, 0, 1:].data
        # print(values.shape)
        # print('attn')
        values = values / (self.attn.size(0) * self.attn.size(1) * self.attn.size(2))
        if self.seq_ranks is None:
            self.seq_ranks = torch.zeros(grad.size(3) - 1).cuda()

        self.seq_ranks += torch.abs(values)

    # def compute_out(self, grad):
    #     # print(grad.sum(), self.attn.sum())
    #     B, N, d = grad.shape
    #     out = torch.abs(self.attn).sum(dim=0).sum(dim=-1)[1:].data
    #     grad = torch.abs(grad).sum(dim=0).sum(dim=-1)[1:].data
    #     values = out * grad
    #     # values = torch.sum(torch.abs(grad) * torch.abs(self.attn), dim=0, keepdim=True).sum(dim=2)[0, 1:].data
    #     values = values / (B * N)
    #     # print(values.shape)
    #     if self.seq_ranks is None:
    #         self.seq_ranks = torch.zeros(N - 1).cuda()
    #
    #     # print(values.sum())
    #
    #     self.seq_ranks += torch.abs(values)

    def reset_rank(self):
        self.attn = None
        self.seq_ranks = None
        self.cnt = 0

    def flops(self):
        flops = 0
        total_dim = self.head_dim * self.num_heads
        # q.k.v dot x
        flops += 3 * self.channel * total_dim * self.in_dim
        # attn = q matmul k.transpose
        flops += self.channel * total_dim * self.channel
        # softmax
        flops += self.num_heads * self.channel * self.channel
        # out = attn matmul v
        flops += self.channel * total_dim * self.channel
        # self.out dot out
        flops += self.channel * total_dim * self.in_dim
        return flops


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_patches, num_tokens=1, mlp_ratio=4., channel=None, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, merge=False, recover=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.channel = channel
        self.attn = Attention(dim, num_patches, num_heads=num_heads, num_tokens=num_tokens,
                              channel=channel, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.merge = merge
        self.recover = recover
        if not self.merge:
            del self.attn.token_mask
            del self.attn.merge_matrix
            del self.attn.recover_matrix
            self.attn.bias = None
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, compute_taylor_attn=False):
        # merge token
        if self.merge:
            # get the redundant for shortcut
            # this step is not necessary
            # x_keep = torch.tensordot(torch.diag(self.attn.token_mask), x, dims=([1], [1])).permute(1, 0, 2)
            x_res = torch.tensordot(torch.diag(1 - self.attn.token_mask), x, dims=([1], [1])).permute(1, 0, 2)
            # print('x_res: ', x_res.sum(dim=-1))
            #
            x = torch.tensordot(self.attn.merge_matrix, x, dims=([1], [1])).permute(1, 0, 2)

        # original
        x = x + self.drop_path(self.attn(self.norm1(x), compute_taylor_attn))
        # print('x msa:', x.sum(dim=-1))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print('x_mlp:', x.sum(dim=-1))

        # token recover
        if self.merge:
            x = torch.tensordot(self.attn.recover_matrix, x, dims=([1], [1])).permute(1, 0, 2)
            # add the shortcut feature
            x = x + x_res
            # assert 0
        return x

    def flops(self):
        flops = 0
        flops += self.attn.flops()
        flops += self.mlp.flops(self.channel)
        return flops, self.attn.flops()


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, embed_dims=None, depth=12,
                 num_heads=12, mlp_ratio=4., channels=None, qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, merge_list=[], recover_list=[], weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            channels (list): number of k,v patches
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.embed_dim = embed_dim
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if channels is None:
            channels = [self.num_patches + self.num_tokens] * depth
            # channels = [197, 197, 118, 118, 118, 78, 78, 78, 39, 39, 39, 197]
        assert len(channels) == depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for i in range(depth):

            merge = True if i in merge_list else False
            recover = True if i in recover_list else False

            self.blocks.append(
                Block(
                    dim=embed_dim, num_heads=num_heads, num_patches=self.num_patches, num_tokens=self.num_tokens,
                    mlp_ratio=mlp_ratio, channel=channels[i], qkv_bias=qkv_bias, attn_drop=attn_drop_rate,
                    drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, merge=merge,
                    recover=recover)
            )

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, compute_taylor_attn=False):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x, compute_taylor_attn)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, compute_attn=False):
        # print("fi{}".format(x.dtype))
        x = self.forward_features(x, compute_taylor_attn=compute_attn)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

    def flops(self):
        embed_dim, in_dim, fw, fh = self.patch_embed.proj.weight.data.shape
        num_class, _ = self.head.weight.data.shape
        # flop_embedding = self.num_patches * in_dim * embed_dim * (fw * fh)
        flop_classify = self.num_patches * num_class * embed_dim
        # print(flop_embedding, flop_classify)
        flops = flop_classify #+ flop_embedding
        attn_f = 0
        for layer in self.blocks:
            flops += layer.flops()[0]
            attn_f += layer.flops()[1]
        return flops, attn_f
        # return self.blocks.flops()  + flop_classify


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    # kwargs.pop('pretrained_cfg')

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        # pretrained_cfg=default_cfg,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model

# @register_model
# def deit_base_patch16_224(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     if 'num_heads' in kwargs:
#         model_kwargs = dict(patch_size=16, depth=12, embed_dim=768, **kwargs)
#     else:
#         model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, pretrained_strict=False, **model_kwargs)
#     return model
#
# @register_model
# def deit_small_patch16_224(pretrained=False, **kwargs):
#     """ ViT-Tiny (ViT-Ti/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     if 'num_heads' in kwargs:
#         model_kwargs = dict(patch_size=16, depth=12, embed_dim=384, **kwargs)
#     else:
#         model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, pretrained_strict=False, **model_kwargs)
#     return model
#
# @register_model
# def deit_tiny_patch16_224(pretrained=False, **kwargs):
#     """ ViT-Tiny (ViT-Ti/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     if 'num_heads' in kwargs:
#         model_kwargs = dict(patch_size=16, depth=12, embed_dim=192, **kwargs)
#     else:
#         model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
#     model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, pretrained_strict=False, **model_kwargs)
#     return model
#
# @register_model
# def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
#     """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
#     ImageNet-1k weights from https://github.com/facebookresearch/deit.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer(
#         'deit_small_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
#     return model
#
#
# @register_model
# def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
#     """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
#     ImageNet-1k weights from https://github.com/facebookresearch/deit.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer(
#         'deit_base_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
#     return model

@register_model
def tmvit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (ViT-Ti/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """

    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, pretrained_strict=False, **model_kwargs)
    return model

@register_model
def tmvit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (ViT-Ti/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    if 'num_heads' in kwargs:
        model_kwargs = dict(patch_size=16, depth=12, embed_dim=192, **kwargs)
    else:
        model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, pretrained_strict=False, **model_kwargs)
    return model

@register_model
def tmvit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (ViT-Ti/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    if 'num_heads' in kwargs:
        model_kwargs = dict(patch_size=16, depth=12, embed_dim=384, **kwargs)
    else:
        model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, pretrained_strict=False, **model_kwargs)
    return model