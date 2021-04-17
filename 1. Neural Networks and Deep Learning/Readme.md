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
<div>
$$
\begin{aligned}
\frac{\partial L}{\partial a} &= \frac{\partial}{\partial a}(-yloga-(1-y)log(1-a)) \\ 
&= -\frac{y}{a}+\frac{1-y}{1-a} \\
\frac{\partial a}{\partial z} &= \frac{\partial }{\partial z}(\frac{1}{1+e^{-z}}) \\ 
&=\frac{1}{1+e^{-z}}(1-\frac{1}{1+e^{-z}}) \\ &=a(1-a) \\
\frac{\partial z}{\partial \omega} &=x \\
\end{aligned}
$$
</div>
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
我们用$\hat{Y}$和$Y$分别表示所有训练样本的预测值和真实值(m,1)，$X$表示所有训练样本的特征值(m,n)，则代价函数对$\omega$的梯度最终可表示为：
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

## 代码解析

```python
# encoding: utf-8

import numpy as np
import h5py
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    """
    初始化权重w和偏置b为0
    :param dim: 向量w的维度(dim, 1)
    :return: w, b
    """
    w = np.zeros((dim, 1))
    b = 0

    return w, b


def propagate(w, b, X, Y):
    """
    计算损失函数及其梯度
    :param w: 权重 （num_px*num_px*3, 1)
    :param b: 偏置，常量
    :param X: 训练集 （num_px*num_px*3, numbers of examples)
    :param Y: 标签 （1, numbers of examples)
    :return: cost -- 代价函数（交叉熵）
             grads -- 代价函数对w和b的梯度，分别用dw，db表示  (dict)
    """
    m = X.shape[1]  # 训练集样本数
    A = sigmoid(np.dot(w.T, X) + b)  # 激活值
    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    cost = np.squeeze(cost)  # 删除维度
    # 反向传播计算梯度
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)
    grads = {"dw": dw,
            "db": db}

    return cost, grads


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    梯度下降法最优化w和b
    :param w: 权重 （num_px*num_px*3, 1)
    :param b: 偏置，常量
    :param X: 训练集 （num_px*num_px*3, numbers of examples)
    :param Y: 标签 （1, numbers of examples)
    :param num_iterations: 迭代次数
    :param learning_rate: 学习率
    :param print_cost: 每100次打印cost
    :return: params -- 存储最优化后权重w和b的字典
             grads -- 存储w和b梯度的字典
             costs -- 存储每百次迭代cost的列表
    """

    costs = []
    for i in range(num_iterations):

        cost, grads = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost:f}")

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    使用逻辑回归预测标签(0/1)
    :param w: 权重 （num_px*num_px*3, 1)
    :param b: 偏置，常量
    :param X: 训练集 （num_px*num_px*3, numbers of examples)
    :return : Y_prediction -- 样本X的预测值
    """
    m = X.shape[1]  # numbers of examples
    w = w.reshape(X.shape[0], 1)
    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X)+b)  # (1, numbers of examples)
    for i in range(A.shape[1]):
        if A[0, 1] >= 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    :param X_train: 训练集
    :param Y_train: 训练集标签
    :param X_test: 测试集
    :param Y_test: 测试集标签
    :param num_iterations: 迭代次数
    :param learning_rate: 学习率
    :param print_cost: 打印cost
    :return: d -- 包含模型信息的字典
    """
    # 初始化参数
    w, b = initialize_with_zeros(X_train.shape[0])
    # 梯度下降法
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]

    train_pred = predict(w, b, X_train)
    test_pred = predict(w, b, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(train_pred - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(test_pred - Y_test)) * 100))
    d = {"costs": costs,
         "train_pred": train_pred,
         "test_pred": test_pred,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.
    return train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, classes


if __name__ == "__main__":
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
    # 示例
    plt.figure(0)
    index = 5
    num_px = 64
    plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    print("y="+str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[int(d["test_pred"][0, index])].decode("utf-8") +  "\" picture.")
    # 绘制学习曲线
    plt.figure(1)
    costs = np.squeeze(d["costs"])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate: " + str(d["learning_rate"]))
    plt.show()

```
