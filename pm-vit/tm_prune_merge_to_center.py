import numpy as np
import torch
import torch.nn as nn
# from model import VisionTransformer, SelfAttention
from tmvit import VisionTransformer, Attention
# from prune_by_layer import get_new_attn, get_new_out

# merge_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# recover_list = [4, 7, 10]

eps = 1e-8
sigma = 2

def normalize(score):
    v = np.abs(score)
    v = v / (np.sum(v) + eps)
    return v


def gauss_norm(score, center_score, center):
    if score is []:
        assert 0
    # return score / (score + eps)
    # gauss_filter = [np.exp((-(i-center)**2) / (2*sigma**2)) if score[i] != 0 else 0 for i in range(len(score))]
    # print('gauss_filter: ', gauss_filter)

    # score = np.abs(score * gauss_filter)
    # print("score_before:", score)
    score = score / (center_score + eps)
    bias = score.sum()
    # print('score: ', score)
    return score, bias


def set1_except0(score, center_score, center):
    # print('score:', score)
    value = np.zeros(score.shape[0])
    for i in range(len(score)):
        if score[i] != 0:
            value[i] = 1
    # print('value:', value)
    return value


def normalize_ranks_per_layer(layer_ranks):
    for i in range(len(layer_ranks)):
        v = torch.abs(layer_ranks[i])
        v = v / (torch.sqrt(torch.sum(v * v)) + eps)
        layer_ranks[i] = v
    return layer_ranks


def get_new_qkv(module, head, head_dim, idx):
    # print(module.weight.data.size())
    in_dim = module.weight.data.size(1)
    # print('head:{}  in_dim:{}'.format(head, in_dim))
    new_out_dim = len(idx) * head_dim

    weight = module.weight.data.reshape(3, head, head_dim, in_dim)
    bias = module.bias.data.reshape(3, head, head_dim)

    new_weight = weight[:, idx, :, :].clone()
    new_bias = bias[:, idx, :].clone()

    new_qkv = nn.Linear(in_dim, new_out_dim)
    new_qkv.weight.data = new_weight.reshape(-1, in_dim)
    new_qkv.bias.data = new_bias.reshape(-1)

    return new_qkv


def get_new_proj(module, head, idx):
    out_dim, in_dim = module.weight.data.size()
    head_dim = in_dim // head
    new_in_dim = len(idx) * head_dim

    weight = module.weight.data.reshape(out_dim, head, head_dim)
    bias = module.bias.data

    new_weight = weight[:, idx, :].clone()

    new_proj = nn.Linear(new_in_dim, out_dim)
    new_proj.weight.data = new_weight.reshape(out_dim, -1)
    new_proj.bias.data = bias.clone()

    return new_proj


def get_new_conv(model, idx):
    num_seq = len(idx)
    new_conv = nn.Conv2d(num_seq, num_seq, kernel_size=1, stride=1, bias=None, groups=num_seq)
    new_weight = model.weight.data[idx, :, :, :].clone()
    new_conv.weight.data = new_weight.clone()
    return new_conv


def prune_tokens(model, masks):
    model.cpu()
    layer = 0
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            # idx = np.squeeze(np.argwhere(np.asarray(masks[layer])))
            # # add class token to index
            # idx = np.concatenate([[1], idx + 1], axis=0)
            #
            # module.index = nn.Parameter(torch.tensor(idx), requires_grad=False)
            mask = masks[layer]
            module.token_index = nn.Parameter(torch.cat([torch.tensor([True]), mask], dim=0), requires_grad=False)
            layer += 1
            # for i in range(layer, len(masks)):
            #     masks[i] = torch.cat([masks[i][mask], masks[i][~mask]], dim=0)
            # print(module.index)
            module.reset_rank()


def prune_attention(model, masks):
    model.cpu()
    layer = 0
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            idx = np.squeeze(np.argwhere(np.asarray(masks[layer])))

            new_qkv = get_new_qkv(module.qkv, module.num_heads, module.head_dim, idx)
            new_proj = get_new_proj(module.proj, module.num_heads, idx)

            module.qkv = new_qkv
            module.proj = new_proj
            module.num_heads = len(idx)

            module.reset_rank()

            layer += 1


def prune_token_by_layer(model, layer, layer_num):
    # print(layer)
    for name, module in model.named_modules():
        if isinstance(module, Attention) and '.{}.attn'.format(layer) in name:
            # print(name)
            layer_rank = module.seq_ranks.cpu().clone()
            smallest = np.sort(np.asarray(layer_rank))[-layer_num]
            # print(smallest)
            mask = layer_rank > smallest
            # print(mask)
            # idx = np.argwhere(np.asarray(mask))
            module.token_index = nn.Parameter(torch.cat([torch.tensor([True]), mask], dim=0), requires_grad=False)
            # print(module.token_index)
            module.reset_rank()
            break


def get_merge_matrix(model, masks, scores, center_list, merge_list, mode='attn'):
    model.cpu()
    stage = 0
    channels = [len(scores[0]) + 1] * 12
    for name, module in model.named_modules():
        layer = merge_list[stage]
        mask = masks[stage]
        score = scores[stage]
        center = center_list[stage]
        square = int(np.sqrt(len(score)))

        # print(mask)

        if isinstance(module, Attention) and '.{}.attn'.format(layer) in name:
            # get zero mask
            if mode == 'clus':
                token_mask = torch.tensor(mask, dtype=torch.float32)
            else:
                token_mask = torch.cat([torch.tensor([1], dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)], dim=0)
            # print(token_mask)
            module.token_mask = nn.Parameter(token_mask, requires_grad=False)

            channel = int(len(center)) + 1
            # print('layer {}: {}'.format(layer, channel))
            channels[layer] = channel

            # init merge matrix
            matrix = np.zeros((channel, len(score) + 1))
            matrix[0][0] = 1
            # print(matrix.shape)

            # init bias
            bias = np.ones(channel)

            left = 1

            for i in range(len(center) - 1):
                line = (center[i] + 1) // square
                right = int(np.ceil((center[i] + center[i + 1]) / 2) + 1)
                # print('before: left:{}, right:{}, next:{}'.format(left, right, center[i + 1] + 1))
                line_r = (right - 1) // square
                if line < line_r:
                    right = (line + 1) * square + 1
                # print('after: left:{}, right:{}, next:{}'.format(left, right, center[i + 1] + 1))
                matrix[i + 1][left:right], bias[i+1] = gauss_norm(score[left - 1:right - 1], score[center[i]], center[i]-left+1)
                # bias[i+1] = (score[left - 1:right - 1] != 0).sum()
                left = right
            matrix[-1][left:], bias[-1] = gauss_norm(score[left - 1:], score[center[-1]], center[-1]-left+1)
            # bias[-1] = (score[left - 1:] != 0).sum()

            # assert 0
            # print(matrix)
            # print(bias)
            bias = np.log(np.sqrt(bias))
            # print(bias)
            module.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

            matrix = torch.tensor(matrix, dtype=torch.float32)
            module.merge_matrix = nn.Parameter(matrix)
            # get recover matrix
            recover_matrix = torch.linalg.pinv(matrix)
            # print(recover_matrix[:, :2])
            module.recover_matrix = nn.Parameter(recover_matrix)
            # update stage
            if stage < len(merge_list) - 1:
                stage += 1

    return channels


def tm_prune(model, num_prune, num_keep, merge_list, mode='attn'):
    seq_ranks = []
    masks = []
    recovers = []
    if mode == 'head':
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                seq_ranks.append(module.head_ranks)

    elif mode == 'attn':
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                seq_ranks.append(module.seq_ranks.cpu())
    elif mode == 'random':
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                num = module.token_index.data.shape[0] - 1
                seq_ranks.append(torch.rand(num))
    elif mode == 'clus':
        # assert len(num_keep) == len(merge_list)
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                if 'token_mask' in module.state_dict().keys():
                    masks.append(np.asarray(module.token_mask.cpu()))

                if 'recover_matrix' in module.state_dict().keys():
                    recovers.append(np.asarray(module.recover_matrix.cpu()))
                seq_ranks.append(module.seq_ranks.cpu())
                # print(module.seq_ranks)
    else:
        print('no such mode')
        assert 0
    normalize_ranks_per_layer(seq_ranks)
    # print(layer_ranks)`q
    layer_ranks = []
    for i in merge_list:
        layer_ranks.append(np.asarray(seq_ranks[i]) * (1-0.025*i))

    layer_ranks = np.asarray(layer_ranks)

    if mode == 'random':
        layers = len(layer_ranks)
        for layer in np.random.randint(layers, size=num_prune):
            length = len(layer_ranks[layer])
            id = np.random.randint(length)
            layer_ranks[layer][id] = 0
        masks = torch.tensor([layer_rank != 0 for layer_rank in layer_ranks])

    elif mode == 'attn':
        # global
        num_prune = num_prune * len(merge_list)
        num_keep = num_keep * len(merge_list)
        prune_thr = np.sort(np.hstack(layer_ranks))[num_prune]
        keep_thr = np.sort(np.hstack(layer_ranks))[-num_keep]
        masks = [layer_rank >= prune_thr for layer_rank in layer_ranks]
        # local
        # prune_thr = np.sort(layer_ranks[merge_list], axis=-1)[:, num_prune]
        # keep_thr = np.sort(layer_ranks[merge_list], axis=-1)[:, -num_keep]
        # # print(smallest)
        # masks = [layer_rank >= thr for layer_rank, thr in zip(layer_ranks[merge_list], prune_thr)]

        scores = []
        for layer_rank, mask in zip(layer_ranks, masks):
            scores.append(layer_rank * mask)
        # print(scores)
        # center_list = [(np.where(score >= thr)[0]) for score, thr in zip(scores, keep_thr)]
        center_list = [(np.where(score >= keep_thr)[0]) for score in scores]
    elif mode == 'clus':
        masks = np.asarray(masks)
        # recovers = np.asarray(recovers)

        scores = []
        for layer_rank, recover, mask in zip(layer_ranks, recovers, masks):

            # layer_rank = np.concatenate([[1.0], layer_rank], axis=0)
            # print(layer_rank)
            score = layer_rank @ recover.T[1:, 1:]
            scores.append(score * mask[1:])
        # print(layer_ranks.shape)
        # layer rank
        # keep_thr = np.sort(layer_ranks, axis=-1)[:, -num_keep]
        # center_list = [(np.where(score >= thr)[0]) for score, thr in zip(scores, keep_thr)]
        # global rank
        num_keep = num_keep * len(merge_list)
        keep_thr = np.sort(np.hstack(layer_ranks))[-num_keep]
        # print(len(num_keep), layer_ranks.shape)
        # keep_thr = [np.sort(layer_ranks, axis=-1)[i, -num_keep[i]] for i in range(len(num_keep))]
        center_list = [(np.where(score >= keep_thr)[0]) for score in scores]
        # center_list = [(np.where(score >= thr)[0]) for score, thr in zip(scores, keep_thr)]

    if mode == 'random':
        prune_tokens(model, masks)
        # for i in range(12):
        #     print(model.transformer.encoder_layers[i].attn.index.data.shape)
        channels = [mask.sum() + 1 for mask in masks]
        print(channels)
        return channels

    elif mode == 'attn' or mode == 'clus':

        # print('center list: ', center_list)
        channels = get_merge_matrix(model, masks, scores, center_list, merge_list, mode=mode)

        print(channels)

        return channels
