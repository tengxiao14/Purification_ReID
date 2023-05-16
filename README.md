# Purification

> **[Learning to Purification for Unsupervised Person Re-identification]**<br>
> Long Lan, Xiao Teng, Jing Zhang, Xiang Zhang and Dacheng Tao<br>
> *IEEE Transactions on Image Processing (**TIP**) 2023 *<br>

# Environment

numpy, torch, torchvision,

six, h5py, Pillow, scipy,

scikit-learn, metric-learn, 

faiss_gpu

### Installation

```shell
git clone https://github.com/tengxiao14/Purification_ReID.git
cd Purification_ReID
python setup.py develop
```

### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the person datasets Market-1501 and MSMT17 datasets.
Then unzip them under the directory like

```
Purification/examples/data
├── market1501
│   └── Market-1501-v15.09.15
└── msmt17
    └── MSMT17_V1
```

## Training

We utilize 4 NVIDIA TITAN V GPUs for training.

**examples:**

Market-1501:

1. Training teacher model:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/Purification_train_usl_teacher.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --lambda1 0.15 --lambda2 0.2
```

2. Training student model:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/Purification_train_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --lambda1 0.15 --lambda2 0.2 --mu 1.0 --teacher-path logs/market_resnet50/model_teacher.pth.tar
```

MSMT17:

1. Training teacher model:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/Purification_train_usl_teacher.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.7 --num-instances 16 --lambda1 0.15 --lambda2 0.2
```

2. Training student model:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/Purification_train_usl.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.7 --num-instances 16 --lambda1 0.15 --lambda2 0.2 --mu 1.0 --teacher-path logs/msmt_resnet50/model_teacher.pth.tar
```

## Evaluation
To evaluate the model, run:
```
CUDA_VISIBLE_DEVICES=0 python examples/test.py -d $DATASET --resume $PATH
```
**Some examples:**
```
### Market-1501 ###
CUDA_VISIBLE_DEVICES=0 python examples/test.py -d market1501 --resume logs/market_resnet50/model_best.pth.tar
```


# Acknowledgements

Thanks to [ClusterContrast](https://github.com/alibaba/cluster-contrast-reid) and [SpCL](https://github.com/yxgeee/SpCL) for the opening source.
