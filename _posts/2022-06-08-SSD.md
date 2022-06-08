---
layout: post
title: SSD
tags: Deep Learnig
author: Maple Quan
math: true
data: 2022-06-08 17:12 +0800
---

# SSD： Single Shot MultiBox Detector

SSD为one-stage方法，如Yolo和SSD，其主要思路是均匀地在图片的不同位置进行密集抽样，抽样时可以采用不同尺度和长宽比，然后利用CNN提取特征后直接进行分类与回归，整个过程只需要一步，所以其优势是速度快，但是均匀的密集采样的一个重要缺点是训练比较困难，这主要是因为正样本与负样本（背景）极其不均衡，导致模型准确度稍低。

论文链接：https://arxiv.org/pdf/1512.02325v5.pdf

###### Abstract

作者提出SSD，其可以将边界框的输出空间离散为一组默认框，每个特征地图位置的不同纵横比和比例。在预测时，网络对在每个默认 box 存在的每个目标分类生成分数，同时为了更好地匹配目标形状对 box 进行调整。该网络结合了来自不同分辨率的多个特征地图的预测，从而自然地处理各种大小的物体。SSD完全消除了 proposal generation 和之后的 pixel 或者 feature 的重采样阶段，并且将所有的计算封装在一个网络中。

###### Introduction

对于如Faster RCNN这样的网络，其对于嵌入式系统来说是太**计算密集**的，甚至对于实时的应用，及时使用更高端的硬件也是很慢的。作者提出的方法不为包围盒假设和对像素或特征进行重新采样。本文提出的改进包括使用一个**小型卷积滤波器**来预测边界盒位置的对象类别和偏移量，使用单独的**预测器(过滤器)**来进行不同的纵横比检测，并将这些滤波器应用于网络后期的多个特征映射，以便在多个尺度上执行检测。在这些改进中，对不同的尺度使用多层来预测能够实现低分辨率下的高准确率，进一步地提高了检测速度。

###### The Single Shot Detector (SSD)

<img src="Pictures\image-20220410195345639.png" alt="image-20220410195345639" style="zoom: 67%;" />

对不同大小物体的识别 feature map 不同，对于每个 default box，预测所有目标类别的形状偏移量和置信度

SSD是基于前馈卷积网络来产生一个固定大小的 bounding box 的集合和为在 boxes 中的目标类别实例进行打分，接着通过 NMS 来产生最后的目标。整体的网络结构为 base network + auxiliary structure(***产生具有以下的关键检测的特征***)：

- **Multi-scale feature maps for detection:** 在截断的 base network 后面加上卷积特征层，这些层的大小是逐渐减小，可以进行多尺度的预测；
- **Convolutional predictors for detection:** 每个添加的特征层(或者从 base network可选的一个存在的特征层)能够用一系列的卷积核产生一组固定的预测目标。对于一个大小为 m x n、p通道的特征图，使用 3 x 3 x p的小核能够产生某一类别的得分或者与 default box coordinates 相关的偏移量；
- **Default boxes and aspect ratios:** default box 以一种卷积方式展平特征图，这样每个 box 的位置相对于其相应的格子是固定的。在每个特征映射格子中，其预测相对于单元格中默认框形的偏移量，以及表示每个这些框中存在一个类实例的分数。对于每个位置预测 k 个box，计算 c 个类别的得分和与 original default box 形状相关的四个偏移量。这样每个位置需要(c + 4)k 个 filters，对于一张 m x n 的特征图产生(c + 4)kmn 个输出。

**SSD 模型结构图**

<img src="Pictures\image-20220410210918321.png" alt="image-20220410210918321" style="zoom:67%;" />

训练：

- **Matching strategy:** 对于每个 gt box，选择位置、aspect ratio 和 scale 不同的 default box。将每个 gt box 与 jaccard overlap 高于阈值(*0.5*)的进行匹配。同时，允许网络对多个重叠的 default boxes 进行预测分数，而不是只选择最大重叠的一个。

- **Training objective:** 设 ![img](D:\Note\Pictures\20170619101004297.png) 为一个 indicator 对第i个 default box 与第j个类别p的 gt box 的匹配。总的损失函数为 localization loss (loc) 与 confidence loss (conf) 的加权和：
  $$
  L(x,c,l,g)=\frac{1}{N}(L_{conf}{(x,c) + \alpha{L_{loc}{x,l,g}}})\\
  N为匹配的 default\, boxes 的数量。若N=0，则loss = 0\\
  $$
  <img src="D:\Note\Pictures\image-20220411225013423.png" alt="image-20220411225013423" style="zoom:80%;" />
  
- **Choosing scales and aspect ratios for default boxes:** 同时使用 lower 和 upper feature maps 进行预测。在每个特征图上， default box 的尺寸计算如下：
  $$
  s_k = s_{min} + \frac{s_{max}-s_{min}}{m-1}(k-1), k∈[1, m]\\
  其中s_{min}=0.2,s_{max}=0.9\\
  aspect\, ratios \,a_r∈[{1, 2, 3 ,\frac{1}{2},\frac{1}{3}}]\\
  对每个 default \, box 计算 width(a_{k}^{a}=s_{k}\sqrt{a_{r}})和高度(h_{k}^{a}=s_{k}\sqrt{a_{r}})\\
  对于 aspect \,ration为1，额外增加一个default box，该box的尺度为s_{k}^{'}=\sqrt{s_{k}{s_{k+1}}}
  $$



​			每一个default box，宽度、高度、中心点计算如下：
$$
w_{k}^{a}=s_{k}\sqrt{a_r}\\
h_{k}^{a}=s_{k}\sqrt{a_r}\\
(\frac{i+0.5}{|f_k|},\frac{j+0.5}{|f_k|})
$$

- **Hard negative mining:**  经过matching后，很多default box是负样本，这将导致正样本、负样本不均衡，训练难以收敛。因此，该论文将负样本根据置信度进行排序，选取最高的那几个，并且保证负样本、正样本的比例为3：1。

- **Data augmentation:** 

- 为了使得模型对目标的尺度、大小更加鲁棒，该论文对训练图像做了data augmentation。每一张训练图像，由以下方法随机产生：

     1）使用原始图像;

     2）采样一个path，与目标的最小jaccard overlap为0.1、0.3、0.5、0.7、0.9  ;

     3）随机采样一个path。

  ​    采样得到的path，其大小为原始图像的[0.1, 1]，aspect ratio在1/2与2之间。当groundtruth box的中心在采样的path中时，保留重叠部分。经过上述采样之后，将每个采样的pathresize到固定大小，并以0.5的概率对其水平翻转。

![image-20220412121854988](Pictures\image-20220412121854988.png)

-------

<img src="Pictures\res50_ssd.png" alt="res50_ssd" style="zoom:200%;" />


-------

