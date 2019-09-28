# paper-note

nips2015: _Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting_

## 1.概述

论文目的是利用历史数据预测往后降水量 （_precipitation nowcasting_）

作者将此问题视为输入和输出都是时空序列的 **时空序列预测问题**。

通过给**FC-LSTM**的 输入到隐层 和隐层到隐层 转移都加上卷积机构，作者提出了 *convolutional LSTM* (convLSTM)

作者将实验结果与FC-LSTM与state-of-art方法ROVER算法比较，convLSTM更胜一筹。

## 2. 绪言

### 2.1 问题形式化

将空间区域划分为$M\times N$的网格，每个网格中有$P$个观测指标，则一个观测值可表示为$\mathcal{X}\in\mathrm{R}^{P\times M \times N}$，则给定前$J$个观测值，预测后$K$个值的问题可形式化为如下：
$$
\large
\tilde{\mathcal{X}}_{t+1},\cdots,\tilde{\mathcal{X}}_{t+K} = \text{arg max}_{\mathcal{X}_{t+1},\cdots,\mathcal{X}_{t+K}}\quad p(\mathcal{X}_{t+1}, \cdots, \mathcal{X}_{t+K} | \hat{\mathcal{X}}_{t-J+1}, \hat{\mathcal{X}}_{t-J+2},\cdots,\hat{\mathcal{X}}_{t})
$$

在本问题中，每个时间点上获得的观测值为一个二维的雷达图像(radar echo map)，文中将图像划分为网格，并且将每个网格中的==？？？==（**view the pixels inside a patch as its measurements** ），则此问题转化为一个**时空序列预测问题**。

问题复杂度与可解性：

> Although the number of free variables in a length-K sequence can be up to O(MKNKPK), in practice we may exploit the structure of the space of possible predictions to reduce the dimensionality and hence make the problem tractable.

### 2.2 LSTM 与FC-LSTM

略去对LSTM的介绍，下面对**FC-LSTM**作简要介绍：$\circ$符号为==?Hadamard乘==，
$$
\begin{aligned}
i_t =& \sigma(W_{xi}x_t+W_{hi}h_{t-1}+\textcolor{red}{W_{ci}\circ{c}_{t-1}}+b_i)\\
f_t =& \sigma(W_{xf}x_t+W_{hf}h_{t-1}+\textcolor{red}{W_{cf}\circ{c}_{t-1}}+b_f)\\
c_t =& f_t \circ c_{t-1} + i_t \circ \tanh (W_{xc}x_t+W_{hc}h_{t-1}+b_c)\\
o_t =& \sigma(W_{xo}x_t+W_{ho}h_{t-1}+\textcolor{red}{W_{co}\circ{c}_{t}}+b_o)
\end{aligned}
$$
 即为在LSTM单元内部添加了三个peepholes，如下图所示：

![LSTM with peepholes](/home/feng/Documents/paper-notes/image/LSTM3-var-peepholes.png)

> 图片来自[link](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)。

其中，

> the input, cell output and states are all 1D vectors.



## 3. 模型介绍

FC-LSTM缺点：

- 尽管能发掘时序相关性，但包含过多的空间信息冗余(contains too much redundancy for spatial data)

- 在input-to-state和state-to-state转移中使用的全连接并未包含任何的空间信息。

convLSTM特性：所有输入$\mathcal{X}_1, \cdots,\mathcal{X}_t$，cell输出$\mathcal{C}_1,\cdots,\mathcal{C}_t$，隐层状态$\mathcal{H}_1,\cdots,\mathcal{H}_t$，以及门输出$i_t, f_t, o_t$均为三维张量，其中最后两个维度为空间信息。
$$
\begin{aligned} 
i_{t} &=\sigma\left(W_{x i} * \mathcal{X}_{t}+W_{h i} * \mathcal{H}_{t-1}+W_{c i} \circ \mathcal{C}_{t-1}+b_{i}\right) \\ 
f_{t} &=\sigma\left(W_{x f} * \mathcal{X}_{t}+W_{h f} * \mathcal{H}_{t-1}+W_{c f} \circ \mathcal{C}_{t-1}+b_{f}\right) \\ 
\mathcal{C}_{t} &=f_{t} \circ \mathcal{C}_{t-1}+i_{t} \circ \tanh \left(W_{x c} * \mathcal{X}_{t}+W_{h c} * \mathcal{H}_{t-1}+b_{c}\right) \\ 
o_{t} &=\sigma\left(W_{x o} * \mathcal{X}_{t}+W_{h o} * \mathcal{H}_{t-1}+W_{c o} \circ \mathcal{C}_{t}+b_{o}\right) \\ 
\mathcal{H}_{t} &=o_{t} \circ \tanh \left(\mathcal{C}_{t}\right) 
\end{aligned}
$$


# ! TODO

