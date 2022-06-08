---
layout: post
title: Feature Pyramid NetWorks
tags: Deep Learnig
author: Maple Quan
math: true
data: 2022-06-08 17:12 +0800
---



# Feature Pyramid Networks

论文链接：https://arxiv.org/pdf/1612.03144.pdf

###### Abstract:

特征金字塔是识别系统中来检测不同尺度目标的基本组成。而深度学习的目标检测也在避免使用特征金字塔来表示，部分的原因是其计算和内存是密集型的。本文将研究内在的多尺度、深度卷积网络的金字塔分级来构造具有很少额外成本的特征金字塔。

###### Introduction:

总结了几种解决识别物体尺寸差异很大的方法，这也是计算机视觉所面临的难题之一。

<img src="Pictures\image-20220406153216377.png" alt="image-20220406153216377" style="zoom:67%;" />



图 (a)  **Featurized image pyramid**中方法优点有：

- 对每种尺度的图像进行特征提取，能够产生多尺度的特征表示，并且所有等级的特征图都具有较强的语义信息，甚至包括一些高分辨率的特征图。

缺点：

- 推理时间大幅度增加；
- 由于内存占用巨大，用图像金字塔的形式训练一个端到端的深度神经网络变得不可行；
- 如果只在测试阶段使用图像金字塔，那么会造成一个问题：由于训练时,网络只是针对于某一个特点的分辨率进行训练，推理时运用图像金字塔,可能会在训练与推理时产生“矛盾”。

图 (b) **Single feature map**  是利用单个的高层特征图来进行预测(Faster RCNN)

图 (c) **Pyramidal feature hierarchy** :深层ConvNet逐层计算特征层级，而对于下采样的特征等级有内在的多尺度和金字塔形状。这样在网络中的特征等级产生不同空间分辨率的特征映射，但引入了由不同深度引起的较大的语义差异。高分辨率映射具有损害其目标识别表示能力的低级特征。

图 (d) **Feature Pyramid Network** :  采用一种结构，它将低分辨率，具有高分辨率的强大语义特征，语义上的弱特征通过自顶向下的路径和横向连接相结合。其结果是一个特征金字塔，在所有级别都具有丰富的语义，并且可以从单个输入图像尺度上进行快速构建。

<img src="Pictures\image-20220406160112016.png" alt="image-20220406160112016" style="zoom:80%;" />

###### Related Work:

<img src="Pictures\image-20220406160615294.png" alt="image-20220406160615294" style="zoom:80%;" />

- **Bottom-up pathway**: 前馈 backbone 的一部分，每一级往上用 step = 2 的降采样。选择每个 stage 的最后一层作为特征图的 reference set。对于Resnets，作者选择 residual blocks 为 {C₂，C₃，C₄，C₅}作为stage，分别对应输入图片的下采样倍数为{4，8，16，32}，不选择C₁是由于语义低和占较大的内存。
- **Top-down pathway and lateral connections**: 通过从更高的金字塔层次上对空间上更粗，但语义上更强的特征图进行采样，从而产生更高分辨率的特征。有了更粗的空间分辨率，上采样将空间分辨率提高两倍，即放大到上一个 stage 的特征图一样的大小。上采样的方法使用的是最近邻插值法。具体过程为：C5层先经过1 x 1卷积，改变特征图的通道数(文章中设置d=256，与Faster R-CNN中RPN层的维数相同便于分类与回归)。M5通过上采样，再加上(特征图中每一个相同位置元素直接相加)C4经过1 x 1卷积后的特征图，得到M4。这个过程再做两次，分别得到M3，M2。M层特征图再经过3 x 3卷积(减轻最近邻近插值带来的混叠影响，周围的数都相同)，得到最终的P2，P3，P4，P5层特征。

<img src="Pictures\v2-3a1a33dab4980683b5ebe73d43c7396b_720w.jpg" alt="img" style="zoom: 67%;" />

<img src="Pictures\v2-fe85fb352b9c212fb6d5416330fad9d2_720w.jpg" alt="img" style="zoom:67%;" />