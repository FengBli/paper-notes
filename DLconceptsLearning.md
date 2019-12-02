# Learn the Deep Learning

## 1. LSTM & GRU

### 1.1 Materials

[Understanding LSTM Networks - Colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[Understanding GRU Networks - Simeon Kostadinov](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)

[LSTM and GRU empirical evaluation](https://arxiv.org/pdf/1412.3555v1.pdf)

[LSTM vs GRU](http://www.deeplearningessentials.science/rnnAdvanced/)

### 1.2 Notes and thoughts

GRU: update gate: $z_t$, reset gate $r_t$:

![GRU](/home/feng/Documents/paper-notes/image/gru.png)

LSTM: forget gate $f_t$, input gate $i_t$, output gate $o_t$ ($\sigma$ from left to right)

![LSTM](/home/feng/Documents/paper-notes/image/LSTM3-chain.png)


$$
\begin{aligned}

f_t  =& \sigma(W_f[h_{t-1},x_t]+b_f)\\

i_t =& \sigma(W_i[h_{t-1},x_t]+b_i) \\

C_t =& \tanh (W_c[h_{t-1},x_t]+b_c) \\

C_t =& f_t\cdot C_{t-1} + i_t\cdot C_t \\

o_t =& \sigma(W_o[h_{t-1}, x_t] + b_o) \\

h_t =& o_t\cdot \tanh(C_t)
\end{aligned}
$$
