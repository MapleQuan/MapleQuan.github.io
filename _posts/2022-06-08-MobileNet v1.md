---
layout: post
title: MobileNet v1
tags: Deep Learnig
author: Maple Quan
math: true
data: 2022-06-08 17:12 +0800
---

# MobileNetv1

MobileNet v1是google在2017年提取，其是一个专门为移动和嵌入式设备设计的CNN模型。

论文链接：https://arxiv.org/abs/1704.04861

###### 摘要：

摘要部分主要讲了MobileNet v1是基于流线型结构，其使用深度可分离卷积去构建轻量化神经网络。作者提出两个简单的全局超参数来高效的平衡延迟和准确性，而超参数允许模型构建者基于问题的约束来为其应用选择合适的模型大小。

###### 介绍：

作者提出问题：现在的趋势是为了达到更高的准确率，网络结构越来越深、越来越复杂，但是实际上这些都没有必要的。在真实的应用中，识别任务是需要及时地展现在被限制的计算平台上。

###### 网络结构:

1.深度可分离卷积：深度可分离卷积可以分解为deepwise convolution 和pointwise convolution。这样的分解是对计算量的减少和模型的大小有着很大影响的。

<img src="Pictures\image-20220324163437351.png" alt="image-20220324163437351" style="zoom: 80%;" />

------

deepwise convolution: 将卷积和变成单通道，输入有M个通道数，就需要M个卷积核，每个通道分别进行卷积，最后做叠加。如图：

<img src="Pictures\image-20220324164049495.png" alt="image-20220324164049495" style="zoom:80%;" />

pointwise convolution: 用1x1的卷积核进行卷积，作用是对卷积和的特征进行升维。如图：

<img src="Pictures\image-20220324164427435.png" alt="image-20220324164427435" style="zoom:80%;" />

------

设输入为  DF x DF x M 的特征图F，产生以一个DF x DF x N的特征图G

对于标准卷积DK x DK，计算量为：
$$
D_F \cdot  D_F \cdot M \cdot N \cdot D_K \cdot D_K
$$
对于深度可分离卷积，计算量为：
$$
deepwise convolution: D_F \cdot D_F \cdot M \cdot D_K \cdot D_K\\

	pointwise convolution: N \cdot M \cdot D_F \cdot D_F\\
	\longrightarrow depthwise separable convolution: D_F \cdot D_F \cdot M \cdot D_K \cdot D_K + N \cdot M \cdot D_F \cdot D_F
$$
减少的计算量：
$$
\frac{D_F \cdot D_F \cdot M \cdot D_K \cdot D_K + N \cdot M \cdot D_F \cdot D_F}{D_F \cdot  D_F \cdot M \cdot N \cdot D_K \cdot D_K}=\frac{1}{N} + \frac{1}{D^2_F}
$$

2.网络结构和训练：所有的卷积层除了最后一个全连接层后面都有BN层和ReLU。

标准卷积与深度可分离卷积块对比：

<img src="Pictures\image-20220324171022162.png" alt="image-20220324171022162" style="zoom:80%;" />

MobileNet v1网络结构:

<img src="Pictures\image-20220324171149284.png" alt="image-20220324171149284" style="zoom:80%;" />

MobileNet v1训练使用的是RMSprop 优化算法，并且使用更少的正则化和数据增强，这是由于小模型过拟合的问题更少。

3.Width Multiplier-- Thinner Models：作者提出一个称为width multiplier 简单的参数α，其作用是均匀的细化每一层网络。给定输入通道数为M ，通过α的作用后输入通道数变为αM，输出通道数变成αN。应用width multiplier后计算量为：
$$
D_k \cdot D_k \cdot αM \cdot D_F \cdot D_F + αM \cdot αN \cdot D_F \cdot D_F
$$
其中α ∈ (0,1]，典型的取值有1，0.75，0.5和0.25。width multiplier能够有效的减少计算量和参数的数量。

4.Resolution Mutipliter--Reduced Representation: 提出的resolution multiplier ρ 是为了减少神经网络的计算量。应用width multiplier 和 resolution multiplier后的计算量为：
$$
D_k \cdot D_k \cdot αM \cdot ρD_F \cdot ρD_F + αM \cdot αN \cdot ρD_F \cdot ρD_F
$$
其中ρ ∈ (0, 1]，典型的特征图大小较少为224，192，160，128。

<img src="Pictures\image-20220325100108220.png" alt="image-20220325100108220" style="zoom:80%;" />



###### 代码实现：

```python
import nn
import torch.nn as nn

#标准卷积
def conv_bn(inputs, outputs, strides):
    return nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, strides, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True))
#深度可分离卷积
def conv_dw(inputs, outputs, strides):
    return nn.Sequential(
                nn.Conv2d(inputs, inputs, 3, strides, 1, groups=inputs, bias=False),
                nn.BatchNorm2d(inputs),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inputs, outputs, 1,1,0,bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=True))
#模型框架：α与ρ均取1
class MobileNetv1(nn.Module):
    def __init__(self):
        super(MobileNetv1,self).__init__()
        self.model = nn.Sequential(
                        conv_bn(3, 32, 2),
                        conv_dw(32, 64, 1),
                        conv_dw(64, 128, 2),
                        conv_dw(128, 128, 1),
                        conv_dw(128, 256, 2),
                        conv_dw(256, 256, 1),
                        conv_dw(256, 512, 2),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 512, 1),
                        conv_dw(512, 1024, 2),
                        conv_dw(1024, 1024, 1))
                        #nn.AvgPool2d(7))
        self.fc = nn.Linear(1024, 10)
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

```
