---
layout: post
title: Machine Learning
tags: mathjax
author: Maple Quan
data: 2022-06-08 13:56 +0800
---

# 机器学习

#### 单变量线性回归

两类机器学习算法：监督学习和无监督学习

------

**监督学习**：回归问题，分类问题
**无监督学习**：聚类算法（未知前提下分类）

------

软件：Matlab和Octave

------

**代价函数(平方误差函数)**：



------

------

等高线图(等高图像)



------

**梯度下降法('Batch' Gradient Descent)**：最小代价函数，不断改变参数的值

定义：



------

##### 线性回归的梯度算法：

------

线性回归方程始终是一个凸函数(没有局部最优解，只有一个全局最优解) 

------

------

#### 线性代数

**Matrix:** Rectangular array of numbers

**Dimension**: number of  rows x number of columns
$$
A = \left[
\matrix{
  a_{11} & a_{12}&a_{13}\\
  a_{21} & a_{22}&a_{23}\\
  a_{31} & a_{32}&a_{33} 
}
\right]
$$

------

**Vector**: An n x 1 matrix
$$
y = \left[
\matrix{
  a_{1} \\
  a_{2} \\
  a_{3} 
}
\right]
$$
Matrix Addition: two same dimensions can add

Scalar Multiplication: every elements to multiply the number

Matrix Multiplication:

------

矩阵乘法：



矩阵乘法特性：

​		1.不满足交换律(单位矩阵除外)；

​		2.满足结合律；

单位矩阵：A X I=I X A=A

矩阵特性：AA^-1^=A^-1^A=I

矩阵的转置：A ------A^T^------>(A~ij~=A^T^ ~ji~)







------

#### 多变量线性回归

特征缩放（归一化）：将特征取值缩放到-1<=x~i~<=1左右

学习率足够小，代价函数每次迭代都会下降

多项式回归

正规方程：
$$
\theta=(X^{T}X)^{-1}X^{T}y
$$

#### Logistic回归

logistic regression(classification)

决策边界->假设函数属性
