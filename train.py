import sys, os
import cv2
import torch
import argparse
import timeit
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data

from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict, poly_lr_scheduler, AverageMeter
from ptsemseg.loss import *
from ptsemseg.augmentations import *

torch.backends.cudnn.benchmark = True

def train(args):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # Setup Augmentations
    data_aug = Compose([
                        RandomHorizontallyFlip(),
                        RandomRotate90x(),
                        RandomSizedCrop(size=args.img_rows, change_ar=False, min_area=0.95**2),
                        RandomRotate(degree=20),
                       ], is_random_aug=False)

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split='train_v2', img_size=(args.img_rows, args.img_cols), augmentations=data_aug, num_k_split=args.num_k_split, max_k_split=args.max_k_split, seed=args.seed)
    te_loader = data_loader(data_path, is_transform=True, split='train-empty_v2', img_size=(args.img_rows, args.img_cols), augmentations=data_aug, num_k_split=args.num_k_split, max_k_split=args.max_k_split, seed=args.seed)
    v_loader = data_loader(data_path, is_transform=True, split='val_v2', img_size=(args.img_rows, args.img_cols), num_k_split=args.num_k_split, max_k_split=args.max_k_split, seed=args.seed)
    ve_loader = data_loader(data_path, is_transform=True, split='val-empty_v2', img_size=(args.img_rows, args.img_cols), num_k_split=args.num_k_split, max_k_split=args.max_k_split, seed=args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=int(args.batch_size*args.batch_ratio), num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    train_e_loader = data.DataLoader(te_loader, batch_size=args.batch_size-int(args.batch_size*args.batch_ratio), num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=2, shuffle=True, pin_memory=True)
    val_e_loader = data.DataLoader(ve_loader, batch_size=args.batch_size, num_workers=2, shuffle=True, pin_memory=True)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup Model
    model = get_model(args.arch, n_classes)
    model.cuda()

    # Check if model has custom optimizer / loss
    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer
    else:
        milestones = [x for x in range(args.n_epoch//5, args.n_epoch, args.n_epoch//5)]
        gamma = 0.5

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.l_rate, weight_decay=args.weight_decay)
        if args.num_cycles > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch*len(trainloader)//args.num_cycles, eta_min=args.l_rate*1e-1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if hasattr(model, 'loss'):
        print('Using custom loss')
        loss_fn = model.loss
    else:
        loss_fn = cross_entropy2d

    start_epoch = 0
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            model_dict = model.state_dict()
            model_dict.update(convert_state_dict(checkpoint['model_state']))
            model.load_state_dict(model_dict)

            print("Loaded checkpoint '{}' (epoch {}, map {})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['map']))
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 


    scale_weight = torch.tensor([1.0, 0.4, 0.4], device=torch.device('cuda'))
    best_map = -100.0
    best_epoch = -1
    for epoch in range(start_epoch, args.n_epoch):
        start_train_time = timeit.default_timer()

        if args.num_cycles == 0:
            scheduler.step(epoch)

        model.train()
        for i, (images, labels, _) in enumerate(trainloader):
            if args.num_cycles > 0:
                iter_num = i + epoch * len(trainloader)
                scheduler.step(iter_num % (args.n_epoch * len(trainloader) // args.num_cycles)) # Cosine Annealing with Restarts

            optimizer.zero_grad()

            images = images.cuda()
            labels = labels.cuda()

            try:
                e_images, e_labels, _ = next(train_e_loader_iter)
            except:
                train_e_loader_iter = iter(train_e_loader)
                e_images, e_labels, _ = next(train_e_loader_iter)

            e_images = e_images.cuda()
            e_labels = e_labels.cuda()

            images = torch.cat([images, e_images], dim=0)
            labels = torch.cat([labels, e_labels], dim=0)

            outputs = model(images, output_size=(args.img_rows, args.img_cols))

            loss_seg = loss_fn(outputs, labels, lambda_ce=args.lambda_ce, lambda_fl=args.lambda_fl, lambda_dc=args.lambda_dc, lambda_lv=args.lambda_lv, scale_weight=scale_weight)

            loss_seg.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Iter [%6d/%6d] Loss: %.4f" % (epoch+1, args.n_epoch, i+1, len(trainloader), loss_seg))


        map = AverageMeter() # mean F2 score
        mean_loss_seg_val = AverageMeter()
        model.eval()
        with torch.no_grad():
            for loader in [val_e_loader, valloader]:
                running_metrics.reset()
                map.reset()
                mean_loss_seg_val.reset()

                for i_val, (images_val, labels_val, _) in tqdm(enumerate(loader)):
                    images_val = images_val.cuda()
                    labels_val = labels_val.cuda()

                    outputs_val = model(images_val, output_size=(args.img_rows, args.img_cols))

                    loss_seg_val = loss_fn(outputs_val, labels_val, lambda_ce=args.lambda_ce, lambda_fl=args.lambda_fl, lambda_dc=args.lambda_dc, lambda_lv=args.lambda_lv, scale_weight=scale_weight)
                    mean_loss_seg_val.update(loss_seg_val, n=images_val.size(0))

                    pred = outputs_val.max(1)[1]

                    running_metrics.update(labels_val, pred)

                    map_val = running_metrics.comput_map(labels_val, pred)
                    map.update(map_val.mean(), n=map_val.size)

                    if (i_val+1) * args.batch_size >= args.val_size:
                        break

                print('Mean average precision: {:.5f}'.format(map.avg))
                print('Mean val loss: {:.4f}'.format(mean_loss_seg_val.avg))

                score, class_iou = running_metrics.get_scores()

                for k, v in score.items():
                    print(k, v)

                for i in range(n_classes):
                    print(i, class_iou[i])

        state = {'epoch': epoch+1,
                 'model_state': model.state_dict(),
                 #'optimizer_state' : optimizer.state_dict(),
                 'map': map.avg,}
        torch.save(state, "checkpoints/{}_{}_{}_{}-{}_model.pth".format(args.arch, args.dataset, epoch+1, args.num_k_split, args.max_k_split))
        if map.avg >= best_map:
            best_map = map.avg
            best_epoch = epoch+1
            torch.save(state, "checkpoints/{}_{}_best_{}-{}_model.pth".format(args.arch, args.dataset, args.num_k_split, args.max_k_split))

        elapsed_train_time = timeit.default_timer() - start_train_time
        print('Training time (epoch {0:5d}): {1:10.5f} seconds'.format(epoch+1, elapsed_train_time))

    print('best map: {}, epoch: {}'.format(best_map, best_epoch))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='bisenet',
                        help='Architecture to use [\'fcn8s, unet, segnet, pspnet, icnet, etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='airbus',
                        help='Dataset to use [\'pascal, camvid, ade20k, cityscapes, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=385,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=385,
                        help='Width of the input image')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=10,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-4,
                        help='Weight Decay')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')

    parser.add_argument('--seed', nargs='?', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--num_cycles', nargs='?', type=int, default=1,
                        help='Cosine Annealing Cyclic LR')
    parser.add_argument('--batch_ratio', nargs='?', type=float, default=1.0,
                        help='Ratio of samples in a single batch for ship masks')
    parser.add_argument('--val_size', nargs='?', type=int, default=8000,
                        help='Size for random selected validation samples')

    parser.add_argument('--lambda_ce', nargs='?', type=float, default=0.0,
                        help='Weight for cross entropy loss')
    parser.add_argument('--lambda_fl', nargs='?', type=float, default=1.0,
                        help='Weight for focal loss')
    parser.add_argument('--lambda_dc', nargs='?', type=float, default=0.0,
                        help='Weight for dice loss')
    parser.add_argument('--lambda_lv', nargs='?', type=float, default=1.0,
                        help='Weight for lovasz softmax loss')

    parser.add_argument('--num_k_split', nargs='?', type=int, default=0,
                        help='The K-th fold cross validation')
    parser.add_argument('--max_k_split', nargs='?', type=int, default=0,
                        help='The total K fold cross validation')

    args = parser.parse_args()
    print(args)
    train(args)
