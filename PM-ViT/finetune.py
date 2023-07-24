import torch
import os
import numpy as np
from pathlib import Path
import utils
import time
import torch
import torch.backends.cudnn as cudnn
import json

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler
from config import get_prune_args
from losses import DistillationLoss
from speed_test import speed_test
from torch import nn
# from ps_vit import VisionTransformer
from tmvit import VisionTransformer
from tm_prune_merge_to_center import tm_prune, prune_token_by_layer

debug = 0

if debug:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def main(args):
    device = torch.device(args.device)

    utils.init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size*1.5),
        # batch_size=256,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'channels' in checkpoint:
            channels = checkpoint['channels']
            print('channels:{}'.format(channels))
        else:
            channels = None
        checkpoint_model = checkpoint['model']
        # for n, p in checkpoint_model.items():
        #     print(n, p.data.shape)
    else:
        channels = None
        checkpoint_model = None

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
        channels=channels,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        merge_list=args.merge_list,
        recover_list=args.recover_list,
    )

    model.to(device)

    num_patches = model.patch_embed.num_patches
    new_length = model.pos_embed.shape[-2]
    if checkpoint_model is not None:
        ori_length = checkpoint_model['pos_embed'].shape[-2]
        if new_length != ori_length:
            pos_embed_checkpoint = checkpoint_model['pos_embed'].permute(0, 2, 1)

            new_pos_embed = torch.nn.functional.interpolate(pos_embed_checkpoint, size=(new_length),
                                                            mode='linear', align_corners=False)
            checkpoint_model['pos_embed'] = new_pos_embed.permute(0, 2, 1)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops, attn_flops = model_without_ddp.flops()
    print('number of params:', n_parameters)
    print('flops:', flops)

    if args.speed_only:
        # test model throughput for three times to ensure accuracy
        inference_speed = speed_test(model, batchsize=args.batch_size)
        print('inference_speed (inaccurate):', inference_speed, 'images/s')
        inference_speed = speed_test(model, batchsize=args.batch_size)
        print('inference_speed:', inference_speed, 'images/s')
        inference_speed = speed_test(model, batchsize=args.batch_size)
        print('inference_speed:', inference_speed, 'images/s')

        return

    # model_without_ddp.set_impact_cal()

    if args.pretrain is not '':
        # checkpoint = torch.load(args.pretrain, map_location='cpu')
        model_without_ddp.load_pretrained(args.pretrain)
    elif args.resume is not '':
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    else:
        print('None checkpoint to load!')
        assert 0

    # print(model_without_ddp.blocks[0].attn.recover_matrix.requires_grad)
    # print(model_without_ddp.blocks[0].attn.bias.requires_grad)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        print('using ema')
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512
    args.lr = 1e-6
    # optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    optimizer = create_optimizer(args, model_without_ddp)

    base_criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        base_criterion = SoftTargetCrossEntropy()
        # base_criterion = torch.nn.CrossEntropyLoss()
    elif args.smoothing:
        base_criterion = torch.nn.CrossEntropyLoss()
        # base_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        base_criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    criterion = DistillationLoss(
        base_criterion, teacher_model, 'none', args.distillation_alpha, args.distillation_tau
    )

    if args.resume:
        if'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    output_dir = Path(args.output_dir)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        # checkpoint_paths = [output_dir / 'augreg.pth']
        # for checkpoint_path in checkpoint_paths:
        #     utils.save_on_master({
        #         'model': model_without_ddp.state_dict(),
        #         'channels': channels
        #     }, checkpoint_path)
        #     print('model saved on ', checkpoint_path)
        return

    if args.prune:

        result = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, 0, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                mode=args.prune_mode,
                set_training_mode=False,  # keep in eval mode during learning prune plan
                prune_iter=args.prune_iter
            )

        # merge_list = args.merge_list
        num_prune = int(num_patches * args.prune_rate)
        num_keep = int(num_patches * args.keep_rate)
        # num_keep = [int(num_patches * rate) for rate in args.keep_rate]
        channels = tm_prune(model, num_prune, num_keep, args.merge_list, mode=args.prune_mode)

        model.to(device)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write('channels' + str(channels) + "\n")

            checkpoint_paths = [output_dir / 'pruned_{}_{}.pth'.
                format(args.prune_rate, args.keep_rate)]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'channels': channels
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        return

    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            # global_pool='avg'
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'], strict=False)
        teacher_model.to(device)
        teacher_model.eval()

    optimizer.param_groups[0]["lr"] = linear_scaled_lr

    args.lr = linear_scaled_lr
    print('lr:', args.lr)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = DistillationLoss(
        base_criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    max_accuracy = 0.0

    if args.freeze_param:
        for n, p in model_without_ddp.named_parameters():
            if 'merge_matrix' not in n and 'recover_matrix' not in n:
                p.requires_grad = False
        print('param: ', model_without_ddp.blocks[0].attn.qkv.weight.requires_grad)
    if args.freeze_matrix:
        for n, p in model_without_ddp.named_parameters():
            if 'recover_matrix' in n or 'merge_matrix' in n:
                p.requires_grad = False
        print('matrix: ', model_without_ddp.blocks[0].attn.recover_matrix.requires_grad)

    for epoch in range(args.final_finetune):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=True  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)

        if not debug:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best = False
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                best = True
            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir:
            if best:
                checkpoint_paths = [output_dir / 'best.pth']
            else:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    # 'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    # 'emb_dim': emb_dims,
                    # 'num_heads': num_heads,
                    'channels': channels,
                    'args': args,
                }, checkpoint_path)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    args = get_prune_args()
    args.epochs = args.final_finetune
    if args.prune:
        args.mixup = 0
        args.reprob = 0
        args.cutmix = 0
    args.aa = None
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
