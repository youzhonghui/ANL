To run the demo:

1. mkdir data
2. python main.py --config ./configs/vgg.json

Some training logs can be found at ./logs



| Model (+ANL and w/o finetune) | Accuracy (CIFAR-10) |
| ----------------------------- | ------------------- |
| VGG-16                        | 94.65               |
| MobileNet-V2                  | 94.91               |
| WRN-28-10                     | 96.56               |


Citation:
```
@inproceedings{ANL,
  author    = {Zhonghui You and
               Jinmian Ye and
               Kunming Li and
               Ping Wang},
  title     = {Adversarial Noise Layer: Regularize Neural Network By Adding Noise},
  booktitle = {{IEEE} International Conference on Image Processing (ICIP)},
  year      = {2019}
}
```