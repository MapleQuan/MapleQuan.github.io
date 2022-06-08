---
layout: post
title: ShuffleNet v2
tags: Deep Learnig
author: Maple Quan
math: true
data: 2022-06-08 17:12 +0800
---

# ShuffleNet v2

论文链接：https://arxiv.org/pdf/1807.11164v1.pdf

**Abstruct**

神经网络结构的设计一般是由间接的计算复杂度主导(i.e. FLOPs)，但是直接的度量(i.e. speed) 还取决于其他因素，如内存获取的消耗和平台的特性。

**Introduction**

除了准度率，计算复杂度是另外的一个值得重点考虑的。而在真实的场景中，在计算预算限制的条件下获得最好的准确率，指定目标平台和应用场景。为了衡量计算复杂度，广泛采用的度量方式是浮点运算的次数 (**FLOPs**)。

<img src="Pictures\image-20220408160509216.png" alt="image-20220408160509216" style="zoom: 50%;" />

从图中可以看出具有相同的 FLOPs 的模型运行速度也会相差很多，所以只用 FLOPs 来衡量计算复杂度是不充分的，也会导致得不到最优模型的设计。导致这种差异的主要原因有两个：

- ​	对速度有重大影响的几个重要的因素没有被 FLOPs 考虑到，比如 memory access cost (MAC)、并行度；
- 具有相同的 FLOPs 的模型在不同平台上的运行速度可能是不同的。

作者提出设计有效网络结构的原则：

- 用直接度量来代替间接度量；
- 直接在目标平台上评估。

**Practical Guidelines for Efficient Network Design**

![image-20220408161936587](Pictures\image-20220408161936587.png)

从上图可以得到卷积部分占据了 FLOPs 度量的大部分，其他操作包括数据输入输出，数据打乱和逐元素的一些操作也占据了相当大部分时间。所以 FOLPs 不是一个实际运行时间的准确估计。

根据这两个原则，提出四种有效的网络设计原则：

- **G1) Equal channel width minimizes memory access cost (MAC)**: 深度可分离卷积中 pointwise convolution 占了大部分的复杂度。作者研究 1 x 1卷积，输入输出通道数为 c₁ 和 c₂，而 h 和 w 为特征图的空间大小，则1 x 1卷积的 FLOPs 为 B= hwc₁c₂ 。
  $$
  MAC = hw(c_1 + c_2) + c_1 + c_2\\
  等式两项分别代表输入输出特征图和权重参数的代价\\
  由均值不等式\, \frac{c_1 + c_2}{2} {\geq}\sqrt{c_1 c_2}得出:\\
  MAC \geq 2hw\sqrt{c_1 c_2} + c_1 c_2\geq 2\sqrt{hwB} + \frac{B}{hw}
  $$
  MAC有一个由 FLOPs 给出的下限，当且仅当输入输出通道数相等时，其等于下限。当 c₁ : c₂ = 1:1时，网络的MAC更小，评估速度更快。
  
- **Excessive group convolution increases MAC：**分组卷积是现代网络体系结构的核心。它通过将所有通道之间的密集卷积改变为稀疏卷积(仅在通道组内)来降低计算复杂度(FLOPs)。一方面，它允许在一个固定的FLOPs下使用更多的channels，并增加网络容量(从而提高准确性)。然而，另一方面，增加的通道数量导致更多的MAC。
  $$
  MAC = hw(c_1 +c_2) + \frac{c_1 c_2}{g}\\
  =hwc_1 + \frac{Bg}{c_1} + \frac{B}{hw}\\
  其中g 为分组数， B=hwc_1c_2/g 为FLOPs
  $$
  能够得出，给定固定的输入形状 c₁ x h x w 和计算代价B，MAC随着g的增长而增加。很明显使用大量的分组数会显著降低运行速度。
  
- **Network fragmentation reduces degree of parallelism：**尽管碎片化结构已经被证明有利于提高准确性，但其可能会降低效率，因为他对GPU等具有强大的并行计算能力的设备并不友好。而且还引入了额外的开销，例如内核启动和同步。

- **Element-wise operations are non-negligible：** 逐元素操作占据了相当大量的时间，尤其是在GPU上。逐元素算子包括 ReLU、AddTensor、AddBias等，它们的 FLOPs 相对较小，但是 MAC 较大。

**ShuffleNet v2: an Efficient Architecture**

在给定预算（FLOPs）的情况下，特征图的通道数也是受限制的。为了在不显著增加 FLOPs 计算量的情况下提高通道数，ShuffleNet v1论文采用了两种技术：**逐点组卷积和类瓶颈结构**，然后引入 "channel shuffle" 操作，使得不同组的通道之间能进行信息交流，提高精度。<img src="D:\Note\Pictures\image-20220409103802340.png" alt="image-20220409103802340" style="zoom: 67%;" />

为了实现较高的模型容量和效率，关键问题是如何维持大量且同样宽的通道，同时没有密集卷积(**1 x 1 卷积**)也没有太多的分组。

在 ShuffleNet v1 block的基础上，**ShuffleNet v2 block 引入通道分割（Channel Split）这个简单的算子**来实现上述目的，如图 (c) 所示。在每个单元 (block) 的开始，我们将输入特征图的  c 个通道切分成 (split) 两个分支 (branches)： c - c'个通道和 c' 个通道。根据 **G3** 网络碎片尽可能少，其中一个分支保持不变（shortcut connection），另外一个分支包含三个通道数一样的卷积来满足 **G1**。和 v1 不同，v2 block 的两个 卷积不再使用分组卷积，一部分原因是为了满足 **G2**，另外一部分原因是一开始的通道切分 （split)操作已经完成了分组效果。

最后，对两个分支的结果进行拼接（concatnate），这样对于卷积 block 来说，输入输出通道数是一样的，符合 **G1** 原则。和 ShuffleNet v1 一样都使用通道打乱（channel shuffle）操作来保证两个分支的信息进行交互。" Add" 操作不在使用，逐元素操作算子：ReLU 和 depthwise convolution 只存在右边分支。同时，三个连续的逐元素操作算子：**Concat, Channel Shuffle and Channel Split** 合并为一个逐元素算子。

对于需要空间下采样的 block，卷积单元进行略微的修改，channel split 被移除，然后 block 的输出通道数变为两倍，如图 d。

**ShuffleNet v2 架构：**![image-20220409105719288](Pictures\image-20220409105719288.png)

ShuffleNet v2 不仅高效，而且准确。主要的两个原因是：

- 首先，每个构建块的高效率使使用更多的特征通道和更大的网络容量成为可能
- 第二，在每个块中，有一半的特征通道直接穿过该块并加入下一个块。这可以看作是一种特性重用，就像DenseNet和CondenseNet的思想一样。