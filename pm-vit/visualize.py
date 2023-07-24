
import os
import torch
import cv2
import numpy as np

from timm.models import create_model
from tmvit import VisionTransformer

# from src/config import get_eval_config
# from src/data_loaders import *

image_path = '/home/maojunzhu/pycharmprojects/deit/data/imagenet/val/n01530575/ILSVRC2012_val_00025347.JPEG'
# image_path = '/home/maojunzhu/pycharmprojects/tm-vit/data/imagenet/val/n01667114/ILSVRC2012_val_00013394.JPEG'
# ImageNet ti16-attn25
path = '/home/maojunzhu/pycharmprojects/tm-vit/experiments/b16_augreg_01_06/best.pth'

# res_path = './vis_results/turtle394'
res_path = './vis_results/bird347'

border_color = [114, 144, 114]


def load_matrix(path):
    print('load matrix')
    if path.endswith('pth'):
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint['model']

        # assert len(state_dict['blocks.0.attn.token_index']) == \
        #        len(model.blocks[0].attn.token_index), 'token number must be the same'
        channels = checkpoint['channels']

        print('channels: ', channels)

        merge_matrix = []
        recover_matrix = []
        for (name, param) in state_dict.items():
            if 'merge_matrix' in name:
                merge_matrix.append(param)
            if "recover_matrix" in name:
                recover_matrix.append(param)
        # for i, module in enumerate(model.blocks):
        #     module.attn.token_index.data = index_list[i]

        # channels = [int(index.sum()) for index in index_list]
        #
        # print(channels)

        return channels, merge_matrix, recover_matrix

        # print(model.blocks[-1].attn.token_index)

    else:
        assert 0, 'only support .pth weight'


def split_image(image, block_size):
    height, width, _ = image.shape
    block_height = height // block_size
    block_width = width // block_size

    blocks = []

    for i in range(block_height):
        for j in range(block_width):
            block = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, :]
            blocks.append(block)

    return blocks


def process_blocks(blocks):
    processed_blocks = []

    for index, block in enumerate(blocks):
        # block_mean = np.mean(block, axis=(0, 1), keepdims=True)
        # processed_block = np.ones_like(block) * block_mean
        processed_block = block

        # # 计算渐变颜色
        # start_color = [0, 0, 0]  # 起始颜色为红色
        # end_color = [0, 255, 255]  # 结束颜色为蓝色
        # gradient = index / len(blocks)  # 渐变系数，根据索引计算
        #
        # # 计算当前小块的边框颜色
        # border_color = [
        #     int(start_color[0] * (1 - gradient) + end_color[0] * gradient),
        #     int(start_color[1] * (1 - gradient) + end_color[1] * gradient),
        #     int(start_color[2] * (1 - gradient) + end_color[2] * gradient)
        # ]

        # 在小块边缘添加边框
        processed_block[0, :, :] = border_color  # 顶部边框为渐变颜色
        processed_block[block.shape[0] - 1, :, :] = border_color  # 底部边框为渐变颜色
        processed_block[:, 0, :] = border_color  # 左侧边框为渐变颜色
        processed_block[:, block.shape[1] - 1, :] = border_color  # 右侧边框为渐变颜色

        processed_blocks.append(processed_block)

    return processed_blocks


def merge_blocks(processed_blocks, image_size, block_size):
    block_height = image_size[0] // block_size
    block_width = image_size[1] // block_size
    merged_image = np.zeros((image_size[0], image_size[1], 3))
    index = 0

    for i in range(block_height):
        for j in range(block_width):
            merged_image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, :] = \
            processed_blocks[index]
            index += 1

    return merged_image


def main():

    if not os.path.exists(res_path):
        os.mkdir(res_path)

    channels, merge_matrix, recover_matrix = load_matrix(path)

    image = cv2.imread(image_path)

    image = cv2.resize(image, (224, 224))

    cv2.imwrite(os.path.join(res_path, 'original.jpg'), image)

    # 将图像分割成小块
    block_size = 16
    blocks = split_image(image, block_size)

    # 处理小块
    processed_blocks = process_blocks(blocks)

    # 合并小块成图像
    merged_image = merge_blocks(processed_blocks, image.shape[:2], block_size)

    # 显示结果
    cv2.imwrite(os.path.join(res_path, 'blocked.jpg'), merged_image)

    layer = 0
    for layer, (merge, recover, channel) in enumerate(zip(merge_matrix, recover_matrix, channels)):
        merge = merge[1:, 1:].cpu()
        recover = recover[1:, 1:].cpu()
        # image_n = image.copy()
        # print(recover.shape)

        processed_blocks = np.asarray(processed_blocks)

        new_blocks = np.ones(processed_blocks.shape) * 255

        act_blocks = []
        rec_blocks = np.ones(processed_blocks.shape) * 255
        # 在小块边缘添加边框
        new_blocks[:, 0, :, :] = border_color  # 顶部边框为渐变颜色
        new_blocks[:, block_size - 1, :, :] = border_color  # 底部边框为渐变颜色
        new_blocks[:, :, 0, :] = border_color  # 左侧边框为渐变颜色
        new_blocks[:, :, block_size - 1, :] = border_color  # 右侧边框为渐变颜色
        # print(new_blocks.shape)
        # assert 0
        # new_blocks = processed_blocks(new_blocks)

        for i in range(channel-1):
            line = merge[i]

            r_line = recover.T[i]

            c_idx = np.max(r_line)

            index = np.where(line != 0)[0]

            act_block = np.dot(line, processed_blocks.transpose((1, 2, 0, 3)))
            # act_blocks.append(act_block)

            new_block = act_block / index.shape[0]

            # print(new_block.shape)

            # print(index.shape[0])

            if len(index) > 1:
                new_block_l = new_block.copy()
                new_block_l[1:-1, new_block.shape[1] - 1, :] = new_block[8, 8, :]
                new_blocks[index[0]] = new_block_l

                if len(index) > 2:
                    new_block_c = new_block.copy()
                    new_block_c[1:-1, new_block.shape[1] - 1, :] = new_block[8, 8, :]
                    new_block_c[1:-1, 0, :] = new_block[8, 8, :]

                    new_blocks[index[1:-1]] = new_block_c

                new_block_r = new_block.copy()
                new_block_r[1:-1, 0, :] = new_block[8, 8, :]
                new_blocks[index[-1]] = new_block_r

            elif len(index) == 1:
                new_blocks[index] = new_block
            else:
                assert 0, 'Wrong line'

        # get recovered blocks
        act_blocks = np.asarray(act_blocks)
        recovered = np.dot(recover, act_blocks.transpose((1, 2, 0, 3)))

        # print(recovered.shape)
        # merge blocks to image
        merged_image = merge_blocks(new_blocks, image.shape[:2], block_size)

        recover_image = merge_blocks(recovered, image.shape[:2], block_size)

        cv2.imwrite(os.path.join(res_path, 'layer{}.jpg'.format(layer)), merged_image)
        cv2.imwrite(os.path.join(res_path, 'recovered_layer{}.jpg'.format(layer)), recover_image)


if __name__ == '__main__':
    main()
