# CIFAR10 ResNet18 Classification with PyTorch
用PyTorch实现CIFAR10数据集上的图像分类。

## 项目介绍
本项目基于开源pytorch-cifar代码二次开发，主要使用模型SimpleDLA在CIFAR10数据集完成图像分类。（文件夹Models中有其他模型，想试试其他的在main.py的net那里改一下就行）
- 修改源码适配Windows系统，修复Linux专属stty（我直接把stty这段删了）、多进程相关报错
- 在训练代码中新增日志记录，自动将每轮epoch的训练/测试loss、acc保存到CSV文件
- 独立编写绘图脚本，读取CSV绘制loss和accuracy变化曲线，直观观察模型收敛与过拟合现象
- 支持断点续训，可加载保存的模型权重继续训练

## 环境依赖
- Python >=3.6
- PyTorch >=1.0
- torchvision
- matplotlib
- numpy


## Accuracy
> 下表为原项目基准测试结果；我在ResNet中增加Dropout层，可进一步优化模型准确率（只训练200轮不太看得出来差别）。

| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |

