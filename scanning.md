# paper-note: scanning

## 1. DMPP (deep mixture point processes)

KDD2019: ***Deep Mixture Point Processes: Spatio-temporal Event Prediction with Rich Contextual Information***

基本任务：预测事件发生的时间和地点

特点：考虑了丰富的上下午信息，如，天气，交通，social events，地理位置的特性等，整合异质&高维的图片和文字信息

point process: 由一个个事件组成的随机序列

识别点过程：通常由估计“强度函数”来实现。

问题：给定$T$时间内一系列事件（包含时间和地点信息）$\mathcal{X}=\{x_i=(t_i,s_i)\}_{i=1}^N$，以及$T+\Delta T$一系列特征$\mathcal{D}=A_1,A_2,\cdots,A_K$，其中每个特征$A_i$又由<$time, latitude, longitude, description$>构成，预测：

+ $[T,T+\Delta T]$时间窗口内的事件发生的时间和地点；
+ $[T,T+\Delta T]$时间窗口内任意给定区域内的事件发生总数。

网络结构：文字描述一个CNN网络，图片信息一个CNN网络提取特征，然后各通过一个全连接层，然后拼接后再通过一个全连接层。



## 2. Soccer

KDD2019: ***Actions Speak Louder than Goals: Valuing Player Actions in Soccer***

对足球赛的过程建立描述语言规范，定义价值函数，学习每个动作的权重（对本队进球得分的影响），简单的logistic回归，随机森林，XGBoost等

## 3. DSTNs for CTR prediction

KDD2019: ***Deep Spatio-Temporal Neural Networks for Click-Through Rate Prediction***

对当前展示给特定用户的，需要预测点击率的广告，除了考虑本身特征以外，还考虑该用户之前点击过和没点击过的广告（时间），同时考虑与当前广告一同出现的其他广告（空间）。分别作embedding。

再考虑attention机制。

## 4. EARLIEST: early classification

KDD2019: ***Adaptive-Halting Policy Network for Early Classification***

某些场景下，可以考虑不需要看到全部特征再作分类，在看过一部分特征之后就可以作分类，但是显然会影响准确率。涉及到trade-off。

本文利用RNN-based网络进行分类，同时通过强化学习来学习何时停止观测并且输出分类结果。

## 5. Real-time event detection on Social Data Streams

event detection

event: a list of clusters of trending entites over time.

model: it applies clustering on a large stream with millions of entities per minute and produces a dynamically updated set of events.



## 6. Session-based social recommendation via dynamic graph attention networks

Model both **users' session-based interests** and **their social influences**(their friends' influences on him according to his current session). 

使用LSTM对当前用户会话建模，得到embedding；

对其朋友在当前时刻的前一次会话使用相同LSTM建模，得到朋友短期兴趣embedding；

对其朋友的长期兴趣，使用一个（与时间无关的）矩阵表示，每一列表示一个朋友的长期兴趣；

将以上三者使用一个**dynamic graph attention network**建模，得到用户的最终embedding；

与物品embedding内积，再softmax，得到下一个物品的浏览概率