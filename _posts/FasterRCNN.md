---
layout: post
title: Faster RCNN
tags: Deep Learnig
author: Maple Quan
math: true
data: 2022-06-08 17:12 +0800
---

# Faster RCNN

Faster RCNN是对Fast RCCN进行的改进，发表与2015年。

论文链接：[https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)

###### 摘要：

摘要部分主要讲其使用的RPN能够整张图片卷积特征共享，能够接近零消耗区域建议。RPN还是个全卷积网络能够实时预测各个目标的边界和目标的分数。

###### 介绍：

Fast RCNN为了实现实时率使用了非常深的网络，但忽略了在 region proposals 上所耗费的时间。proposals 是在SOTA 目标检测系统中的一个计算瓶颈。Region  方法依赖的是便宜的特征和经济的推荐计划。常见的Region proposals 有Selective Search(SS) 和 EdgeBoxes，其都存在一个消耗大量的时间在目标检测网络中。

###### Region Proposal Networks:

**RPN**：RPN的核心思想是使用CNN卷积神经网络直接产生 region proposals，其本质是在卷积特征图输出层上使用滑动窗口。每个滑动窗口(n=3)映射到低维向量，然后送入两个相同的全连接层—— a box-regression layer(reg) 和 a box-classification layer(cls)。

<img src="Pictures\image-20220403205548277.png" alt="image-20220403205548277" style="zoom: 67%;" />

**Translation-Invarivant Anchors：**在每个滑动窗口的位置上，会同时预测 k 个 region proposals，所以 reg layer 有 4k 个输出来编码 k 个 boxes 的坐标，cls layer 输出 2k 个分数来评估每个 proposal 是否有目标。论文使用的是3个尺度(原文为128²，256²，512²)和3个纵横比(原文为 1:1，1:2，2:1)，所以在每个滑动位置生成 k=9 个 anchors。对于卷积特征图的大小为 W x H(典型值为2400)，总共大约有 WHk 个 anchors。通过卷积核进行尺度核纵横比的采样，使用3种尺度和3种纵横比来产生的 anchors 是具有平移不变性的。

------
在计算机视觉中的一个挑战就是平移不变性:比如人脸识别任务中，小的人脸(24x24的分辨率)和大的人脸(1080x720)如何在同一个训练好权值的网络中都能正确识别。若是平移了图像中的目标，则建议框也应该平移，也应该能用同样的函数预测建议框。传统有两种主流的解决方式：
第一、对图像或feature map层进行尺度\宽高的采样;
第二、对滤波器进行尺度\宽高的采样(或可以认为是滑动窗口).

------

**A Loss Function for Learning Region Proposals:**

对于训练 RPNs，对于每个 anchor 采用的二分类标签。如何定义 positive label:

- 与 gt(ground-truth) box 最高IoU 重叠的 anchor / anchors()
- 与任意 gt box 的IoU 重叠超过0.7的 anchor

***事实上，采用第②个规则基本上可以找到足够的正样本，但是对于一些极端情况，例如所有的Anchor对应的anchor box与groud truth的IoU不大于0.7,可以采用第一种规则生成。***

negative labe: 对于所有的 gt boxes 其 IoU 小于0.3 的 anchor

而对于既不是 positive label 也不是 negative label 的 anchor，以及跨越图像边界的 anchor，应该将其舍弃。

根据 Fast RCNN 中的 multi-task loss，其损失函数为：
$$
L({p_i},{t_i})= \frac{1}{N_{cls}}\sum_{i}{L_{cls}(p_i,p_{i}^{*})+λ\frac{1}{N_{reg}}\sum_{i}{p^*}{L_{reg}(t_i,{t_{i}^{*})}}}\\
其中i 为一个 mini-batch中的第i个 anchor;\\
p_i是 anchor i是目标的预测概率;\\
p_{i}^{*}是 gt label, 如果为1，则该 anchor 为 positive，反之为 negative;\\
t_i是预测的 bounding\, box 的参数化坐标向量;\\
t_{i}^{*}是 positive \,anchor 对应的 gt\,box 的坐标向量;\\
L_{cls}是两个类别(object \,vs.\,not object) 的对数损失：\\
L_{cls}(p_{i},p_{i}^{*})=-log[p_{i}^{*}p_i+(1-p_{i}^{*})(1-p_i)]\\
L_{reg}(t_i,t_{i}^{*})是回归损失，用L_{reg}(t_i,t_{i}^{*})=R(t_i-t_{i}^{*})来计算，R是 smooth \,L_1函数;\\
p_{i}^{*}L_{reg}意味着只有前景 anchor (p_{i}^{*}=1)才有回归损失;\\
λ为平衡权重
$$


***注意：Lcls在pytorch实现代码中所使用的是BSE损失，所以cls layer 只预测 k scores***

![image-20220404101452635](Pictures\image-20220404101452635.png)

其中x, y, w, h是 box 的中心坐标，宽，高；x是 predicted box，xa为anchor box，x﹡为ground-truth box

###### Training RPNs:

RPN 通过反向传播和随机梯度下降进行端到端训练。每个 mini-batch 包含一张图像中随机采样256个 anchor 来计算损失函数，positive 和 negative anchors 的比例为 1:1。如果 positive samples 少于128，则使用negative ones 来填充。

 新增的层的参数用均值为0，标准差为0.01的高斯分布来进行初始化，其余层参数用InageNet 分类预训练模型来初始化。在PASCAL数据集上(*Caffe*)：前60k个mini-batch进行迭代，学习率设为0.001；后20k个mini-batch进行迭代，学习率设为0.0001；设置动量momentum=0.9，权重衰减weightdecay=0.0005。

###### Share Features for RPN and Fast R-CNN:

RPN在提取得到proposals后，作者选择使用Fast-R-CNN实现最终目标的检测和识别。RPN和Fast-R-CNN共用了13个VGG的卷积层，显然将这两个网络完全孤立训练不是明智的选择，作者采用交替训练（Alternating training）阶段卷积层特征共享：

第一步，我们依上述训练RPN，该网络用ImageNet预训练的模型初始化，并端到端微调用于区域建议任务；

第二步，我们利用第一步的RPN生成的建议框，由Fast R-CNN训练一个单独的检测网络，这个检测网络同样是由ImageNet预训练的模型初始化的，这时候两个网络还没有共享卷积层；

第三步，我们用检测网络初始化RPN训练，但我们固定共享的卷积层，并且只微调RPN独有的层，现在两个网络共享卷积层了；

第四步，保持共享的卷积层固定，微调Fast R-CNN的fc层。这样，两个网络共享相同的卷积层，构成一个统一的网络。

###### Faster RCNN的架构：

<img src="Pictures\format,png.png" alt="a.jpg" style="zoom: 50%;" />

**RoI Pooling**: 提取一个固定长度的特征向量，每个特征会输入到一系列全连接层，得到一个RoI特征向量。其中一个是传统的softmax层分类，输出有K个类别加上背景类；另一个是 bounding box regressor。RoI Pooling 就是将不同大小的roi 池化成大小相同的feature map，利于输出到下一层fc全连接网络中

***目的： 其是一个简单版本的SPP，目的是为了减少计算时间并且得出固定长度的向量。原来SPP是金字塔 4x4，2x2，1x1，而RoI 改为单个4x4或者KxM。***

<img src="Pictures\image-20220408110838143.png" alt="image-20220408110838143" style="zoom: 50%;" />

<img src="Pictures\image-20220408111128262.png" alt="image-20220408111128262" style="zoom: 33%;" />



------

<img src="Pictures\image-20220416100316317.png" alt="image-20220416100316317" style="zoom:200%;" />

------