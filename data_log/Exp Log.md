# Exp Log

此前的实验发现，似乎dynamic convolution在小训练集上的性能甚至不如普通的卷积

这次训练100epoch，在VD数据集

singlehead:exp_gtcrn_2025-09-03-23h37m

SDR: 18.7468
SISNR: 18.6838
PESQ: 2.8937
ESTOI: 0.8495
STOI: 0.9394

multihead+ln:remote

static:exp_gtcrn_2025-08-31-15h03m

SDR: 18.6681
SISNR: 18.6257
PESQ: 2.8628
ESTOI: 0.8435
STOI: 0.9384

anneal 100 epochs, compare on VD test set

关于因果性的问题 ——> pad left



这次训练200epoch，在dns数据集

**vanilla：remote，2025-09-10-23h45m**	

**multihead+ln+causual_pad：server，exp_gtcrn_2025-09-10-23h28m**



try to reduce hidden channels and delete one dpgrnn : 

tensorboard --logdir_spec=dynamic:exp_gtcrn_2025-09-10-23h28m,vanilla:exp_gtcrn_2025-09-10-23h45m,small:D:/windy/speech-enhancement/SEtrain/exp_gtcrn_2025-09-12-17h07m

TODO：figure out all about time padding:use extra padding left

try to **downsample time** axis, **stride=5**:

server: small:exp_gtcrn_2025-09-13-13h52m ---------> this is **no good**

**stride = 2**:

remote:exp_gtcrn_2025-09-13-16h52m



```
>>> torch.mean(a["model"]['encoder.en_convs.0.conv.weight'].abs(), dim=(0,2,3)) 
tensor([0.3264, 0.3919, 0.4117, 0.0535, 0.0678, 0.0628, 0.0495, 0.0601, 0.0472],
       device='cuda:0')
```

输入的幅度贡献比较大

```
a["model"]['decoder.de_convs.4.bn.weight'] 
tensor([ 0.7491, -0.0060], device='cuda:0')
```

输出的实部贡献比较大

所以进行实验，此外隐藏层改为8（幅值谱，幅值mask）：remote：exp_gtcrn_2025-09-14-18h02m

进行实验，此外隐藏层改为12（幅值谱，幅值mask）：remote：exp_gtcrn_2025-09-16-12h57m



server：exp_gtcrn_2025-09-15-09h41m：尝试time+channel down，channel变为12（但是仍然是复数mask），time下采样一次；200epoch，dns3



server：train_kd, 使用知识蒸馏，每一层kdloss简单相加50epoch，然后正常训练。学生隐藏层改为8（幅值谱，幅值mask）：exp_gtcrn_2025-09-18-17h32m

server：把DPRNN中的intra-RNN换成了多头自注意力exp_gtcrn_2025-09-23-00h00m

fattn（加上feed-forward）:exp_gtcrn_2025-10-02-16h25m

换位rotational pe: exp_gtcrn_2025-10-03-20h59m（不好）

把头数改为2（前面都是4）：exp_gtcrn_2025-10-09-13h19m（不太行，提前结束了）



下一步实验：将inter-RNN换成有限感受野的因果多头注意力

server: exp_gtcrn_2025-09-24-13h45m

感受野都换成200

server：exp_gtcrn_2025-09-30-10h20m

使用retentive network代替inter-RNN：exp_gtcrn_2025-10-15-09h32m（实现错误）

exp_gtcrn_2025-10-17-19h14m



尝试使用RNN代替FFN网络：exp_gtcrn_2025-10-10-21h35m：45.67 MMac 90.762

去掉高维投影的KQ：exp_gtcrn_2025-10-13-11h30m：43.44 MMac 90.218

上面基础上再加上位置编码：exp_gtcrn_2025-10-12-23h40m（效果一般，几乎同上，提前终止）

与其对照试验的channel-20：exp_gtcrn_2025-10-10-15h37m：42.82 MMac 117.241

（也许应该使用非端到端模型）



（尝试使用concat形式的位置编码）

[UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via Network Architecture Search](https://arxiv.org/pdf/2503.00340)（下一步工作）



不同频带/不同时间采用不同参数的RNN：exp_gtcrn_2025-10-21-10h17m

4：exp_gtcrn_2025-10-23-13h55m

2个卷积块也进行上述处理：exp_gtcrn_2025-10-25-16h13m

GTConv也进行上述处理：exp_gtcrn_2025-10-27-13h25m

加入一个表示F坐标的channel（posconv）：exp_gtcrn_2025-10-31-15h36m



加入cTFA，与dyn做对照：exp_gtcrn_2025-10-29-13h25m

clean实现，16channel，g16：experiments/exp_gtcrn_2025-11-03-09h35m
用来进行机器间比较的，结果差不多，没问题

cleaner实现（前后都有mask），16channel，g16：experiments/exp_gtcrn_2025-11-03-10h41m
这个好像不太有效，暂时不管了

per_band_acrean：全部分裂参数，16channel，g16：experiments/exp_gtcrn_2025-11-04-11h15m
机器间比较，结果差不多，甚至稍微差一点，我觉得PESQ最多精确到0.01

very primitive，end2end，16channel，g16：experiments/exp_gtcrn_2025-11-04-17h17m

clean实现，32channel，g16：experiments/exp_gtcrn_2025-11-05-09h04m

per_band_acrean：全部分裂参数，16channel，g16：experiments/exp_gtcrn_2025-11-05-17h31m

fattn, 32channel, g16: experiments/exp_gtcrn_2025-11-06-09h14m

very primitive，end2end, 32channel, g16: experiments/exp_gtcrn_2025-11-06-21h55m

后续可以参考一下LiSenNet对相位的做法，好像不太行

以后也许可以将codec和dpgrnn分开来搞？codec也许可以使用ConvNeXt中的方式进行一些训练，或者使用非对称解码器，或者pixel shuffle 操作

RepVGG和ConvNeXt都看一下(前者似乎有效)

考虑不同层动态卷积选择的相关性，进而考虑动态卷积的静态化，或者折中的方案，比如joint attention

再多加一个path，inter-channel可能会有用，说不定是scaling效果不好的原因