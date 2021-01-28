[TOC]

# Logistic Regression

## 激活函数(Activation Function)

$Logistic$回归常用于处理二分类问题：给定样本输入$x$，让我们判断它是正类还是负类，分别对应标签"1"和"0"。我们引入$sigmoid$函数作为激活函数建立分类模型，它的表达式为：
$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$
图像如图所示：

<img src="./images/sigmoid function.jpg" style="zoom:67%;" />

$sigmoid$函数的取值范围是(0,1)，非常适合用来表示预测目标分类的概率，即$\sigma(w^Tx)$表示该样本是正类的概率，通常我们认为当$\sigma(w^Tx)>0.5$时，该样本为正类；反之，则为负类。



## 损失函数(Loss Function)

我们需要一个损失函数衡量预测值 $\hat{y}$ 与实际值 $y$ 误差的大小，从而评价模型的好坏。在线性回归中我们用平方损失函数$L(\hat{y},y)=\frac{1}{2}(y-\hat{y})^2$，但在逻辑回归中我们使用交叉熵损失函数：
$$
L(\hat{y},y)=-ylog(\hat{y})-(1-y)log(1-\hat{y})
$$
直观一点的理解是，我们希望$\hat{y}$与$y$越接近越好，相应的损失函数也就越小。当y=1时，$L=-log(\hat{y})，\hat{y}$越接近实际值1，损失函数越小；同理，y=0时，$L=-log(1-\hat{y}), \hat{y}$ 越接近0，损失函数就越小。

![](.\images\loss function.jpg)

下面从极大似然角度解释交叉熵损失函数：

样本标签为1的概率为$P(y=1|x)=\hat{y}$，那么样本标签为0的概率为$P(y=0|x)=1-\hat{y}$，我们要做的就是当y=1时最大化 $\hat{y}$，当y=0时最大化$1-\hat{y}$。从极大似然角度将两个概率表达式合并，得到：
$$
P(y|x)=\hat{y}^y(1-\hat{y})^{1-y}
$$
当y取0或1时，该式可以转化为上面两个式子。我们希望不论y取何值，概率P(y|x)越大越好。为了方便求导，等式两边取对数（单调性不变），得到：
$$
logP(y|x)=ylog\hat{y}+(1-y)log(1-\hat{y})
$$
log(P|x)越大，-log(P|x)就越小，因此我们可以把-log(P|x)看作损失函数：
$$
L=-log(P|x)=-ylog\hat{y}-(1-y)log(1-\hat{y})
$$
这就是交叉熵损失函数的简单推导。

**损失函数**(Loss Function)是定义在**单个训练样本**上的，即单个样本预测值与实际值的误差，我们在**整个训练样本**上定义**代价函数**(Cost Fcuntion)，表示所有样本误差和的平均，即损失函数的平均值：
$$
J(\omega,b)=-\frac{1}{m}\sum_{i=1}^{m}{y^{(i)}log\hat{y}^{(i)}+(1-y^{(i)})log(1-\hat{y}^{(i)})}
$$
代价函数用$J(\omega,b)$表示。

## 梯度下降法

我们要求代价函数对每个参数$\omega$的偏导

定义：
$$
z=\omega^T x +b\\
\hat{y}=a=\sigma(z) \\
$$
为了计算方便，我们逐步求偏导：
$$
\frac{\partial L}{\partial a}=\frac{\partial}{\partial a}(-yloga-(1-y)log(1-a))= -\frac{y}{a}+\frac{1-y}{1-a}\\
\frac{\partial a}{\partial z}=\frac{\partial }{\partial z}(\frac{1}{1+e^{-z}})=\frac{1}{1+e^{-z}}(1-\frac{1}{1+e^{-z}})=a(1-a) \\
\frac{\partial z}{\partial \omega}=x \\
$$
根据链式法则，可求得损失函数$L$对$\omega$的偏导
$$
\begin{aligned}
\frac{\partial L}{\partial \omega} &=  
\frac{\partial L}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial \omega} \\
&= (-\frac{y}{a}+\frac{1-y}{1-a})a(1-a)x \\
&=(a-y)x \\
&=(\hat{y}-y)x  \\
\end{aligned}
$$
而代价函数$J(\omega)$对$\omega$的偏导即为所有样本损失函数的平均值
$$
\frac{\partial J}{\partial \omega}=\frac{1}{m}\sum_{i=i}^{m}(\hat{y}^{(i)}-y^{(i)})x^{(i)} \\
$$
我们用$\hat{Y}$和$Y$分别表示所有训练样本的预测值和真实值(m\*1)，$X$表示所有训练样本的特征值(m\*n)，则代价函数对$\omega$的梯度最终可表示为：
$$
\frac{\partial J}{\partial \omega}=\frac{1}{m}(\hat Y-Y)·X
$$
$J$对b的偏导与$\omega$类似，唯一的区别就是$\frac{\partial z}{\partial b}=1$，则有
$$
\frac{\partial J}{\partial b}=\frac{1}{m}(\hat Y-Y)
$$
有了梯度，我们就可以更新参数：
$$
\omega := w-\alpha \frac{\partial J}{\partial \omega} \\
b := b-\alpha \frac{\partial J}{\partial b}
$$
