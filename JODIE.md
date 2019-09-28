# paper-note

KDD2019: *Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks*

Stanford

---

[TOC]



## 1. INTRO

用户与物品之间发生序列性交互，即：在一段时间内，用户会相继与不同物品发生交互。实例：电商平台，MOOC，社交协作平台。由此，产生了一类网络：*network of temporal interactions between users and items*。

![图1：用户物品交互](/home/feng/Documents/paper-notes/image/usr-image-sequential-interact.png)

如上图所示：

+ 左图：每个物品与用户的交互信息上均带有标签，标签信息包括：访问时间，和一个*feature vector*，如评论和购买数量等
+ 右图：本文工作，学习一个嵌入投影算子(*embedding projection operator*)来预测用户未来轨迹。

表征学习(*Representation learning*)，即习得实体的低维embedding，能很好的表示用户/物品的属性的演化过程。但现有的方法面临如下四个方面的挑战：

1. 大多数算法只有在用户发生行为时才产生一个embedding。例如：某用户今天买了一个物品，于是模型更新了他的embedding，用户一个月以后再次登录系统，模型对他的embedding还是没有变化。这类模型并没有考虑时间的变化可能导致用户的兴趣变化。<font color='red'>本文使用投影操作来预测用户未来embedding轨迹。</font>
2. 实体既包括不变的属性，也有动态演化的属性。许多模型在嵌入时只考虑了其中一种。<font color='red'>本文同时考虑两者，为每个实体维护两部分的embedding。</font>
3. 许多模型预测用户行为时，针对任一用户，都会为每一个物品打分。当物品数量巨大时，时间复杂度高，不实用。<font color='red'>本文改善方法：直接输出最有可能交互的物品的embedding，而不是输出概率值。</font>
4. 许多模型为了保留时序依赖，一次只处理一个用户行为，可扩展性很差。<font color='red'>本文方法使用t-batch批训练。</font>

本文模型JODIE[[注1]](#foot1)：通过时序的交互信息习得用户和物品的embedding，并且在用户发生行为和投影算子预测用户未来embedding轨迹时更新用户和物品的embedding。

> The embeddings of the user and item are updated when a user takes an action and a projection operator predicts the future embedding trajectory of the user.

在此模型中，每个用户和物品都用两个embedding，一个静态的和一个动态的，静态代表其属性中长期的不发生变化的部分，而动态的则代表其属性中随时间发生改变的部分。后者通过JODIE算法学习得到。在预测用户轨迹时，两部分的embedding都会使用。

JODIE模型由两部分组成：更新操作和投影操作。

+ 更新操作：此部分使用两个RNN生成用户和物品的embedding，两个RNN之间相互耦合，以便整合利用物品与用户间的依赖关系，每当发生交互时，用户部分的RNN利用物品的embedding来更新用户的embedding，物品RNN则会利用用户embedding来更新物品embedding。
+ 投影操作（<font color='red'>模型主要创新点</font>）：用来预测用户未来的embedding轨迹。直观来讲，短时间后，用户的embedding会发生微小变化；长时间后则会发生较大变化。因此，JODIE训练一层时间注意力来对$\Delta$时间后用户的embedding来进行投影。
+ 投影后的用户embedding则被用来预测用户未来最有可能交互的物品。预测结果并不输出每个物品的概率，而是直接输出一个物品embedding，可以通过locality sensitive hashing (LSH)在常数时间内完成。
+ 本文使用批训练算法*t-batch*来加快训练过程，增加可扩展性。将交互过程（即交互网络中的边）分批，任一批中，每个用户和每种物品至多出现一次，并且将交互过程按时间排序。

JODIE模型很容易扩展到多类实体交互的情形，只需要为每一类实体训练一个RNN来产生和更新embedding即可。

## 2. 相关工作

+ DRNN： **RNN**，**Time-LSTM**[[2]](#ref2)  和 **LatentCross**[[3]](#ref3)：
  + 使用one-hot编码物品，只使用了物品的id信息，而无视了物品的现有状态(embedding)
  + 只动态学习用户embedding，而不包括物品
+ Dynamic co-evolutionmodels: **DeepCoevolve**[[4]](#ref4)
  + 都考虑了交互过程中用户和物品之间的相互影响
  + 区别在于投影操作，输出物品embedding和批训练
+ Temporal network embedding models: **CTDNE**[[5]](#ref5), **IGE**[[6]](#ref6)
  + 只为实体生成一个最终的静态的embedding，当有新的交互信息加入后都需要重新训练

## 3. JODIE

#### 模型介绍

![model illustration](/home/feng/Documents/paper-notes/image/JODIE-model-illustration.png)



+ 当$t$时刻的一个交互发生时，RNN~U~ 利用交互信息<font color='red'>$f$</font>，前一时刻的用户embedding <font color='red'>$u(t^-)$</font>，物品embedding <font color='red'>$i(t^-)$</font>共同来更新用户embedding <font color='red'>$u(t)$</font>，RNN~I~同理更新物品embedding  <font color='red'>$i(t)$</font>。

+ 当<font color='red'>$u(t)$</font>和<font color='red'>$i(t)$</font>更新完成后，两者又联合时间间隔<font color='red'>$\Delta$</font>来共同预测<font color='red'>$\Delta$</font>时间后用户embedding的变化，产生<font color='red'>$\hat{u}(t+\Delta)$</font>

#### projection演示

![JODIE-projection-illustration](/home/feng/Documents/paper-notes/image/JODIE-projection.png)

对于用户embedding<font color='red'>$u(t)$</font>，projection操作可以预测不同时间间隔（$\Delta_1<\Delta_2< \Delta$）后它的变化情况（轨迹），当下一时刻（$t+\Delta$）的交互发生时，则按照上述update操作对用户embedding进行更新。

#### 静态与动态embedding

静态embedding $\bar{u} \in \mathbb{R}^d \;\forall  u\in \mathcal{U}$  和$\bar{i}\in\mathbb{R}^d\;\forall i\in\mathcal{I}$不随时间变化，选用one-hot编码。

动态embedding：在$t$时刻，$u(t) \in \mathbb{R}^n \;\forall  u\in \mathcal{U}$  和$i(t)\in\mathbb{R}^n\;\forall i\in\mathcal{I}$动态调整，其变化序列被成为其<font color='red'>轨迹</font>。

## 注和引用：

<a name="foot1">  [注1]</a> JODIE： **Jo**int **D**ynamic User-**I**tem **E**mbedding

<a name="ref2">  [2]</a> Y. Zhu, H. Li, Y. Liao, B. Wang, Z. Guan, H. Liu, and D. Cai. **What to do next: modeling user behaviors by time-lstm**. In IJCAI, 2017.

<a name="ref3">  [3]</a>A. Beutel, P. Covington, S. Jain, C. Xu, J. Li, V. Gatto, and E. H. Chi. **Latent cross: Making use of context in recurrent recommender systems**. In WSDM, 2018.

<a name="ref4">  [4]</a> H. Dai, Y. Wang, R. Trivedi, and L. Song. **Deep coevolutionary network: Embed- ding user and item features for recommendation**. arXiv:1609.03675, 2016.

<a name="ref5">  [5]</a> G. H. Nguyen, J. B. Lee, R. A. Rossi, N. K. Ahmed, E. Koh, and S. Kim. **Continuous- time dynamic network embeddings**. In WWWBigNet workshop, 2018.

<a name="ref6">  [6]</a> Y. Zhang, Y. Xiong, X. Kong, and Y. Zhu. **Learning node embeddings in interaction graphs**. In CIKM, 2017.