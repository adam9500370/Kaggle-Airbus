# Kaggle-Airbus
PyTorch implementation for [Kaggle Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)


## Model Design
A modified BiSeNet [1] model (backbone: ResNet18) with
* Pyramid pooling module in context path and auxiliary training [2]
* Feature fusion modules as decoder in spatial path


## Training
* Augmentation
    - Random horizontally flip
    - Random rotate (0, 90, 180, 270) degrees
    - Random rotate +- 20 degrees
    - Random crop and resize
* Optimizer: Adam
* Weight decay: 1e-4
* Loss functions
    - 1.00 Focal loss [3]
    - 1.00 Lovász-softmax loss [4]
* Using all the training data


## Final prediction
* A single fold model with TTA (horizontal flip)
* Threshold of ship probability: 0.5
* Remove small (< 30 pixels) ship masks
* Average inference time (769x769, on a single 1080TI)
    - Without TTA: 30 FPS
    - With TTA: 18 FPS


## Progress
+ Stage 1
    * Input size (resized): 385x385
    * Batch size (ship/empty): 32 (32/0)
    * Epochs: 10
    * Learning rate: cosine annealing from 1e-3 to 1e-4
    * Best score on public/private board: 0.68808/0.82467
+ Stage 2
    * Input size (resized): 769x769
    * Batch size (ship/empty): 10 (7/3)
    * Epochs: 5
    * Learning rate: cosine annealing from 1e-4 to 1e-5
    * Best score on public/private board: 0.71187/0.83442
+ Stage 3
    * Input size (resized): 769x769
    * Batch size (ship/empty): 10 (5/5)
    * Epochs: 5
    * Learning rate: cosine annealing from 1e-5 to 1e-6
    * Best score on public/private board: 0.70798/0.83678


## Requirements
* pytorch 0.4.1
* torchvision 0.2.0
* numpy
* opencv
* scikit-image
* tqdm
* pandas

`pip install -r requirements.txt`


## Usage

### Data
* Download data from [Kaggle Airbus competition](https://www.kaggle.com/c/airbus-ship-detection/data)
* Extract train/test.zip
* Modify the path appropriately in `config.json`

### To train the model
```
python train.py --arch bisenet --dataset airbus \
                --img_rows 385 --img_cols 385 \
                --n_epoch 10 --batch_size 32 --seed 1234 \
                --l_rate 1e-3 --weight_decay 1e-4 \
                --lambda_fl 1.0 --lambda_lv 1.0 \
                --num_k_split 0 --max_k_split 0 --num_cycles 1 \
                --batch_ratio 1.0 --val_size 8000
```
`python train.py -h` for more details

### To test the model
```
python test.py --model_path checkpoints/bisenet_airbus_10_0-0_model.pth --dataset airbus \
               --img_rows 769 --img_cols 769 --seed 1234 \
               --batch_size 1 --split test_v2 --num_k_split 0 --max_k_split 0 --tta
```
`python test.py -h` for more details


## Reference
[1] [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)

[2] [Pyramid Scene Parsing Network (PSPNet)](https://arxiv.org/abs/1612.01105)

[3] [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

[4] [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790)

[5] Part of code adapted from [meetshah1995/pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)

[6] Official implementation for [Lovász-Softmax loss](https://github.com/bermanmaxim/LovaszSoftmax)
