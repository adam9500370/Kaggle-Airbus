import sys, os
import cv2
import torch
import argparse
import timeit
import random
import collections
from skimage.morphology import label
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.backends import cudnn
from torch.utils import data

from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.models.utils import flip
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict, make_result_dir, AverageMeter
from ptsemseg.metrics import runningScore

cudnn.benchmark = True

def test(args):
    result_root_path = make_result_dir(args.dataset, args.split)

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols), no_gt=args.no_gt, seed=args.seed, num_k_split=args.num_k_split, max_k_split=args.max_k_split)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    n_classes = loader.n_classes
    testloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # Setup Model
    model = get_model(model_name, n_classes)
    model.cuda()

    checkpoint = torch.load(args.model_path)
    state = convert_state_dict(checkpoint['model_state'])
    model_dict = model.state_dict()
    model_dict.update(state)
    model.load_state_dict(model_dict)

    print("Loaded checkpoint '{}' (epoch {}, map {})".format(args.model_path, checkpoint['epoch'], checkpoint['map']))

    running_metrics = runningScore(n_classes)

    rm = 0
    pred_list = []
    map = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (images, labels, names) in tqdm(enumerate(testloader)):
            images = images.cuda()
            if args.tta:
                images_flip = flip(images, dim=3)

            outputs = model(images)
            if args.tta:
                outputs_flip = model(images_flip)

            if outputs.shape[2] != 768 or outputs.shape[3] != 768:
                outputs = F.interpolate(outputs, size=(768, 768), mode='bilinear', align_corners=True)
                if args.tta:
                    outputs_flip = F.interpolate(outputs_flip, size=(768, 768), mode='bilinear', align_corners=True)

            prob = F.softmax(outputs, dim=1)
            if args.tta:
                prob_flip = F.softmax(outputs_flip, dim=1)
                prob_flip_flip = flip(prob_flip, dim=3)
                prob = (prob + prob_flip_flip) / 2.0

            prob = prob[:, 1, :, :] # ship probability
            pred = torch.where(prob < args.pred_thr, torch.zeros_like(prob), torch.ones_like(prob))

            if not args.no_gt:
                running_metrics.update(labels, pred)

                map_val = running_metrics.comput_map(labels, pred)
                map.update(map_val.mean(), n=map_val.size)

            pred = pred.cpu().numpy()
            prob = prob.cpu().numpy()
            for k in range(pred.shape[0]):
                id = names[k][0].split('.')[0]

                decoded = loader.decode_segmap(pred[k])
                if decoded.sum() > 0:
                    decoded_instances = label(decoded, connectivity=2)
                    rle_mask_counts = 0
                    for n in range(1, decoded_instances.max()+1):
                        img_mask = (decoded_instances == n).astype(np.uint8)

                        """
                        # Find min area of a rectangle (bbox)
                        _, contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        cnt = contours[0]
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        img_mask = cv2.drawContours(img_mask, [box], 0, 1, cv2.FILLED)
                        """

                        if img_mask.sum() >= loader.lbl_thr: # Remove ship mask for area of ship < a threshold lbl_thr
                            rle_mask = loader.RLenc(img_mask)
                            rle_mask = rle_mask if len(rle_mask) > 0 else np.nan
                            rle_mask_counts += 1
                            pred_list.append({'ImageId': names[k][0], 'EncodedPixels': rle_mask})
                    if rle_mask_counts == 0:
                        pred_list.append({'ImageId': names[k][0], 'EncodedPixels': np.nan})
                        rm += 1
                else:
                    pred_list.append({'ImageId': names[k][0], 'EncodedPixels': np.nan})
                    rm += 1

                save_result_path = os.path.join(result_root_path, id + '_' + str(args.num_k_split) + '_' + str(args.max_k_split) + '.png')
                cv2.imwrite(save_result_path, decoded)

    if not args.no_gt:
        print('Mean Average Precision: {:.5f}'.format(map.avg))

        score, class_iou = running_metrics.get_scores()

        for k, v in score.items():
            print(k, v)

        for i in range(n_classes):
            print(i, class_iou[i])

        running_metrics.reset()
        map.reset()

    # Create submission
    sub = pd.DataFrame(pred_list)
    sub = sub[['ImageId', 'EncodedPixels']]
    sub.to_csv(args.split + '_' + str(args.num_k_split) + '_' + str(args.max_k_split) + '.csv', index=False)

    print('To black: ', rm)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='bisenet_airbus_best_0-0_model.pth',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='airbus',
                        help='Dataset to use [\'pascal, camvid, ade20k, cityscapes, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=769,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=769,
                        help='Width of the input image')

    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test',
                        help='Split of dataset to test on')

    parser.add_argument('--no_gt', dest='no_gt', action='store_true',
                        help='Disable verification | True by default')
    parser.add_argument('--gt', dest='no_gt', action='store_false',
                        help='Enable verification | True by default')
    parser.set_defaults(no_gt=True)

    parser.add_argument('--seed', nargs='?', type=int, default=1234,
                        help='Random seed')

    parser.add_argument('--num_k_split', nargs='?', type=int, default=0,
                        help='K-th fold cross validation')
    parser.add_argument('--max_k_split', nargs='?', type=int, default=0,
                        help='Total K fold cross validation')

    parser.add_argument('--pred_thr', nargs='?', type=float, default=0.5,
                        help='Threshold of salt probability prediction')

    parser.add_argument('--tta', dest='tta', action='store_true',
                        help='Enable Test Time Augmentation (TTA) with horizontal flip | True by default')
    parser.add_argument('--no_tta', dest='tta', action='store_false',
                        help='Disable Test Time Augmentation (TTA) with horizontal flip | True by default')
    parser.set_defaults(tta=True)

    args = parser.parse_args()
    print(args)
    test(args)
