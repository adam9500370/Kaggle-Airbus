import os
import random
import cv2
import torch
import pandas as pd
import numpy as np

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *


class airbusLoader(data.Dataset):
    def __init__(self, root, split="train_v2", is_transform=True,
                 img_size=(769, 769), augmentations=None,
                 no_gt=False, num_k_split=0, max_k_split=0, seed=1234):

        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.no_gt = no_gt
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225] # torchvision pretrained image transform
        self.files = {}

        self.images_base = os.path.join(self.root, self.split if 'test' in self.split else self.split.replace('val', 'train').replace('all-empty', 'train').replace('train-empty', 'train').replace('val-empty', 'train'))
        if not self.no_gt:
            self.annotations_filename = 'train_ship_segmentations_v2.csv' if 'test' not in split else 'test_ship_segmentations_v2.csv'
            self.annotations_df = pd.read_csv(os.path.join(self.root, self.annotations_filename), index_col=0)

        if max_k_split == 0 or num_k_split == 0 or 'test' in split: # Select all files in the split
            list_filename = os.path.join('data_list', 'list_{}'.format(split.replace('val', 'train')))
        else: # Select the k-th fold files in the split
            list_filename = os.path.join('data_list', 'list_{}_{}_{}'.format(split, num_k_split, max_k_split))

        with open(list_filename) as f:
            self.files[split] = f.read().splitlines()

        self.valid_classes = [0, 255]
        self.ignore_index = 250
        self.lbl_thr = 30

        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))
        self.label_colours = dict(zip(range(self.n_classes), self.valid_classes))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()

        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if not self.no_gt:
            rle_mask = self.annotations_df.loc[os.path.basename(img_path), 'EncodedPixels']
            lbl = self.encode_segmap(self.decode_RLenc(rle_mask, img.shape[:2]).astype(np.uint8))
        else:
            lbl = []

        name = [os.path.basename(img_path)]

        if not self.no_gt and self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, name

    def transform(self, img, lbl):
        if img.shape[0] != self.img_size[0] or img.shape[1] != self.img_size[1]:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR) # cv2.resize shape: (W, H)

        if len(img.shape) == 3:
            img = img[:, :, ::-1] # RGB -> BGR

        img = img.astype(np.float64)
        img = img.astype(float) / 255.0 # Rescale images from [0, 255] to [0, 1]
        img = (img - self.mean_rgb) / self.std_rgb

        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1) # NHWC -> NCHW
        else:
            img = np.expand_dims(img, axis=0)

        img = torch.from_numpy(img).float()

        if not self.no_gt:
            if lbl.shape[0] != self.img_size[0] or lbl.shape[1] != self.img_size[1]:
                lbl = cv2.resize(lbl, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST) # cv2.resize shape: (W, H)

            if 'train' in self.split and lbl.sum() <= self.lbl_thr:
                lbl[lbl == 1] = self.ignore_index

            lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp, img_norm=False):
        out = np.zeros((temp.shape[0], temp.shape[1]))
        for l in range(self.n_classes):
            out[temp == l] = self.label_colours[l]

        out = out / 255.0 if img_norm else out
        return out

    def encode_segmap(self, mask):
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


    def RLenc(self, img, format=True):
        # ref: https://www.kaggle.com/stainsby/fast-tested-rle
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]

        if format:
            return ' '.join(str(rr) for rr in runs)
        else:
            return runs

    def decode_single_mask(self, rle_mask, img_size):
        img = np.zeros(img_size[0] * img_size[1])
        for i, v in enumerate(rle_mask.split()):
            if i % 2 == 0:
                pos = int(v)-1
            else:
                r = int(v)
                img[pos:pos+r] = 1
        return img

    def decode_RLenc(self, rle_masks, img_size, order='F'):
        img = np.zeros(img_size[0] * img_size[1])
        if isinstance(rle_masks, basestring): # py2: basestring, py3: str
            img = img + self.decode_single_mask(rle_masks, img_size)
        elif not isinstance(rle_masks, float): # multiple instances
            for k in range(len(rle_masks.iloc[:])):
                img = img + self.decode_single_mask(rle_masks.iloc[k], img_size)
        return self.decode_segmap(img.reshape(img_size[1], img_size[0]).T)
