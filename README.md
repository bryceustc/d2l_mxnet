# 动手学习深度学习笔记（Dive into Deep Learning，D2L）

**1. 在MXNet中，NDArray是⼀个类，也是存储和变换数据的主要⼯具。NdArray的一些常见数据操作与运算。**

**答：**
见[1](./Code/Introduction.py)

**2. 内存运算开销**

**答：**
  像Y = X + Y这样的运算，我们也会新开内存，然后将Y指向新内存。所以写成Y+=X 能减少内存开销。
  
**3. 自动求梯度**

**答：**
  ```python
  x = nd.arange(4).reshape((4, 1))
  x.attach_grad()    #调⽤attach_grad函数来申请存储梯度所需要的内存
  with autograd.record():   # 需要调⽤record函数来要求MXNet记录与求梯度有关的计算
      y = 2 * nd.dot(x.T, x)
  # 由于x的形状为（4, 1），y是⼀个标量。接下来我们可以通过调⽤backward函数⾃动求梯度。
  # 如果y不是⼀个标量，MXNet将默认先对y中元素求和得到新的变量，再求该变量有关x的梯度。
  y.backward()
  ```
  ![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/1.jpg)
  
  
**4. 梯度下降优化算法**

**答：**
当模型和损失函数形式较为简单时，上⾯的误差最小化问题的解可以直接⽤公式表达出来。这类解叫作解析解。⼤多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作数值解。

数值解中的优化算法，梯度下降法的三种形式BGD(相当于batch_size=全部数据集),SGD(随机梯度下降，通过每个样本来迭代更新一次，相当于batch_size=1)以及MBGD（小批量随机梯度下降，设置batch_size）[参考](https://zhuanlan.zhihu.com/p/25765735)

minibatch update的好处是，每次运算时候因为量大所以需要发起新计算的次数少，额外开销的损耗也会少，另外batch中训练样本的不同可以帮助相互抵消variance，让训练更稳定不易overfit，后半句的理解：Batch size大的时候一个batch的gradient estimate的variance会变小，并且越大的batch越接近gradient descent时对整个dataset的gradient estimate。这样训练更稳定但同时也更不容易跳出non-convex问题的local minima

**5. batch_size, epochs, iteration三个概念区别**

**答：**

（1）batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；

（2）iteration：1个iteration等于使用batchsize个样本训练一次；

（3）epoch：1个epoch等于使用训练集中的全部样本训练一次，即所有的训练样本完成一次Forword运算以及一次BP运算；

一次epoch 总处理数量 = iterations次数 * batch_size大小
简单一句话说就是，我们有2000个数据，分成4个batch，那么batch size就是500。运行所有的数据进行训练，完成1个epoch，需要进行4次iterations。

**6. 权重和偏置的实际意义**

**答：**

宏观来看，权重在告诉你当前神经元应该更关注来自上一层的哪些单元；或者说权重指示了连接的强弱

偏置则告诉你加权和应该多大才能使神经元变得有意义；或者说当前神经元是否更容易被激活

**7. 批量⼤小和学习率的值是⼈为设定的，并不是通过模型训练学出的，因此叫作超参数（hyperparameter）**


**8. Python中yield函数用法**

**答：**

yield在函数中的功能类似于return，不同的是yield每次返回结果之后函数并没有退出，而是每次遇到yield关键字后返回相应结果，并保留函数当前的运行状态，等待下一次的调用。如果一个函数需要多次循环执行一个动作，并且每次执行的结果都是需要的，这种场景很适合使用yield实现。
包含yield的函数成为一个生成器，生成器同时也是一个迭代器，支持通过next方法获取下一个值。
yield基本使用：
```python
def func():
    for i in range(0,3):
        yield i
# 程序开始执行以后，因为func函数中有yield关键字，所以func函数并不会真的执行，而是先得到一个生成器f(相当于一个对象)
# 直到调用next方法，func函数正式开始执行，遇到yield关键字，然后把yield想想成return,return 了一个0，程序停止所以输出0是执行print(next(f))的结果
# 1 同样是执行print(next(f))的结果，累计执行了两次
f = func()
print(next(f))  # 0
print(next(f))  # 1

```
**9. 为什么squared_loss函数中需要使用reshape函数？**

**答：**
要写成 (y_hat - y.reshape(y_hat.shape)) ，保证形状一样能够做减法。不用广播是因为，广播机制生效有一定的前提，首先是两者在其中一个维度上是相同大小的，其次是那个用来广播的矩阵，另一个维度一定是1。


**10. 如果样本个数不能被批量⼤小整除，data_iter函数的⾏为会有什么变化？**

**答：**
当你的batch_size大于num_expamples的时候，就需要用到min来确定indices[i:min(i+batch_size,num_examples)]的上界了

**11. 深度学习整的一个流程步骤**

**答：**

1). 数据集处理 

2). 读取数据
```python
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
# 随机读取⼩批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
```
3). 定义模型
```python
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1)) #该层输出个数为1
``` 
4). 初始化模型参数
```python
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
``` 
5). 定义损失函数 
```python
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss() # 平⽅损失⼜称L2范数损失
``` 
6). 定义优化算法 
```python
#导⼊Gluon后，我们创建⼀个Trainer实例，并
#指定学习率为0.03的小批量随机梯度下降（sgd）为优化算法。该优化算法将⽤来迭代net实例所
#有通过add函数嵌套的层所包含的全部参数。这些参数可以通过collect_params函数获取。
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```
7). 训练模型 
```python
# 在使⽤Gluon训练模型时，我们通过调⽤Trainer实例的step函数来迭代模型参数。上⼀节中我
# 们提到，由于变量l是⻓度为batch_size的⼀维NDArray，执⾏l.backward()等价于执⾏l.
# sum().backward()。按照小批量随机梯度下降的定义，我们在step函数中指明批量⼤小，从
# 而对批量中样本梯度求平均。
num_epochs = 3
for epoch in range(1, num_epochs + 1):
  for X, y in data_iter:
    with autograd.record():
      l = loss(net(X), y)
    l.backward()
    trainer.step(batch_size)
  l = loss(net(features), labels)
  print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```
**12.trainer.step(batch_size)这个batch_size怎么理解？一次迭代了batch_size步？**

**答：**
相当于把计算出来的梯度除以batch size，因为loss.backward()相当于nd.sum(loss).bacward()，也就是把一个batch的loss都加起来求的梯度，所以通过除以batch size能够弱化batch size在更新参数时候的影响。

**13. 在Gluon中，data模块提供了有关数据处理的⼯具，nn模块定义了⼤量神经⽹络的层，loss模块定义了各种损失函数。**

**14. 如何访问dense.weight的梯度？**

**答：**
``dense.weight.grad()``

**15. Softmax函数与交叉熵**

**答：** 
在进入softmax函数之前，已经有模型输出$C$值，其中$C$是要预测的类别数，模型可以是全连接网络的输出$a$，其输出个数为$C$，即输出为$a_1, a_2, ..., a_C$。

所以对每个样本，它属于类别$i$的概率为：
![2](https://www.zhihu.com/equation?tex=y_%7Bi%7D+%3D+%5Cfrac%7Be%5E%7Ba_i%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BC%7De%5E%7Ba_k%7D%7D+%5C+%5C+%5C+%5Cforall+i+%5Cin+1...C)

通过上式可以保证${\sum_{i=1}^{C} y_{i}}=1$，即属于各个类别的概率和为1。

对softmax函数进行求导，即求：

![3](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7By_%7Bi%7D%7D%7D%7B%5Cpartial%7Ba_%7Bj%7D%7D%7D)

第 $i$ 项的输出对第 $j$ 项输入的偏导。

代入softmax函数表达式，可以得到:

![4](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7By_%7Bi%7D%7D%7D%7B%5Cpartial%7Ba_%7Bj%7D%7D%7D+%3D+%5Cfrac%7B%5Cpartial%7B+%5Cfrac%7Be%5E%7Ba_i%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BC%7De%5E%7Ba_k%7D%7D+%7D%7D%7B%5Cpartial%7Ba_%7Bj%7D%7D%7D)

用我们高中就知道的求导规则：对于

![5](https://www.zhihu.com/equation?tex=f%28x%29+%3D+%5Cfrac%7Bg%28x%29%7D%7Bh%28x%29%7D)

它的导数为

![6](https://www.zhihu.com/equation?tex=f%27%28x%29+%3D+%5Cfrac%7Bg%27%28x%29h%28x%29+-+g%28x%29h%27%28x%29%7D%7B%5Bh%28x%29%5D%5E2%7D)

所以在我们这个例子中，

![7](https://www.zhihu.com/equation?tex=g%28x%29+%3D+e%5E%7Ba_i%7D+%5C%5C+h%28x%29+%3D+%5Csum_%7Bk%3D1%7D%5E%7BC%7De%5E%7Ba_k%7D)

上面两个式子只是代表直接进行替换，而非真的等式。

![8](https://www.zhihu.com/equation?tex=e%5E%7Ba_i%7D)即($g(x)$)对$a_j$进行求导，要分情况讨论:

1). 如果![9](https://www.zhihu.com/equation?tex=i+%3D+j)，则求导结果为![8](https://www.zhihu.com/equation?tex=e%5E%7Ba_i%7D)

2). 如果![10](https://www.zhihu.com/equation?tex=i+%5Cne+j)，则求导结果为0

再来看![11](https://www.zhihu.com/equation?tex=%5Csum_%7Bk%3D1%7D%5E%7BC%7De%5E%7Ba_k%7D)对$a_j$求导，结果为![公式](https://www.zhihu.com/equation?tex=e%5E%7Ba_j%7D)。

所以，当![9](https://www.zhihu.com/equation?tex=i+%3D+j)时：

![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7By_%7Bi%7D%7D%7D%7B%5Cpartial%7Ba_%7Bj%7D%7D%7D+%3D+%5Cfrac%7B%5Cpartial%7B+%5Cfrac%7Be%5E%7Ba_i%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BC%7De%5E%7Ba_k%7D%7D+%7D%7D%7B%5Cpartial%7Ba_%7Bj%7D%7D%7D%3D+%5Cfrac%7B+e%5E%7Ba_i%7D%5CSigma+-+e%5E%7Ba_i%7De%5E%7Ba_j%7D%7D%7B%5CSigma%5E2%7D%3D%5Cfrac%7Be%5E%7Ba_i%7D%7D%7B%5CSigma%7D%5Cfrac%7B%5CSigma+-+e%5E%7Ba_j%7D%7D%7B%5CSigma%7D%3Dy_i%281+-+y_j%29)

当![10](https://www.zhihu.com/equation?tex=i+%5Cne+j)时：

![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7By_%7Bi%7D%7D%7D%7B%5Cpartial%7Ba_%7Bj%7D%7D%7D+%3D+%5Cfrac%7B%5Cpartial%7B+%5Cfrac%7Be%5E%7Ba_i%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BC%7De%5E%7Ba_k%7D%7D+%7D%7D%7B%5Cpartial%7Ba_%7Bj%7D%7D%7D%3D+%5Cfrac%7B+0+-+e%5E%7Ba_i%7De%5E%7Ba_j%7D%7D%7B%5CSigma%5E2%7D%3D-%5Cfrac%7Be%5E%7Ba_i%7D%7D%7B%5CSigma%7D%5Cfrac%7Be%5E%7Ba_j%7D%7D%7B%5CSigma%7D%3D-y_iy_j)

其中，为了方便，令![](https://www.zhihu.com/equation?tex=%5CSigma+%3D+%5Csum_%7Bk%3D1%7D%5E%7BC%7De%5E%7Ba_k%7D)


**16. softmax的计算与数值稳定性**

**答：**
在Python中，softmax函数为：
```python
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum()
    return X_exp / partition
```
传入[1, 2, 3, 4, 5]的向量
```python
>>> softmax([1, 2, 3, 4, 5])
array([ 0.01165623,  0.03168492,  0.08612854,  0.23412166,  0.63640865])
```
但如果输入值较大时：
```python
>>> softmax([1000, 2000, 3000, 4000, 5000])
array([ nan,  nan,  nan,  nan,  nan])
```
这是因为在求exp(x)时候溢出了：

一种简单有效避免该问题的方法就是让exp(x)中的x值不要那么大或那么小，在softmax函数的分式上下分别乘以一个非零常数：

![](https://www.zhihu.com/equation?tex=y_%7Bi%7D+%3D+%5Cfrac%7Be%5E%7Ba_i%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BC%7De%5E%7Ba_k%7D%7D%3D+%5Cfrac%7BEe%5E%7Ba_i%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BC%7DEe%5E%7Ba_k%7D%7D%3D+%5Cfrac%7Be%5E%7Ba_i%2Blog%28E%29%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BC%7De%5E%7Ba_k%2Blog%28E%29%7D%7D%3D+%5Cfrac%7Be%5E%7Ba_i%2BF%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BC%7De%5E%7Ba_k%2BF%7D%7D)

这里![](https://www.zhihu.com/equation?tex=log%28E%29)是个常数，所以可以令它等于![](https://www.zhihu.com/equation?tex=F)。加上常数之后，等式与原来还是相等的，所以我们可以考虑怎么选取常数![](https://www.zhihu.com/equation?tex=F)。我们的想法是让所有的输入在0附近，这样![](https://www.zhihu.com/equation?tex=e%5E%7Ba_i%7D)的值不会太大，所以可以让的![](https://www.zhihu.com/equation?tex=F)值为：

![](https://www.zhihu.com/equation?tex=F+%3D+-max%28a_1%2C+a_2%2C+...%2C+a_C%29)

这样子将所有的输入平移到0附近（当然需要假设所有输入之间的数值上较为接近），同时，除了最大值，其他输入值都被平移成负数，为底的指数函数，越小越接近0，这种方式比得到nan的结果更好。
```python
def softmax(x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)

>>> softmax([1000, 2000, 3000, 4000, 5000])
array([ 0.,  0.,  0.,  0.,  1.])
```
当然这种做法也不是最完美的，因为softmax函数不可能产生0值，但这总比出现nan的结果好，并且真实的结果也是非常接近0的。加了一个常数的softmax对原来的结果影响很小。

**17. 交叉熵损失函数**

**答：**
机器学习里面，对模型的训练都是对Loss function进行优化，在分类问题中，[我们一般使用最大似然估计（Maximum likelihood estimation）](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1)来构造损失函数。对于输入的![](https://www.zhihu.com/equation?tex=x)，其对应的类标签为![](https://www.zhihu.com/equation?tex=t)，我们的目标是找到这样的![](https://www.zhihu.com/equation?tex=%5Ctheta)使得![](https://www.zhihu.com/equation?tex=p%28t%7Cx%29)最大。在二分类的问题中，我们有：

![](https://www.zhihu.com/equation?tex=p%28t%7Cx%29+%3D+%28y%29%5Et%281-y%29%5E%7B1-t%7D)

其中，![公式](https://www.zhihu.com/equation?tex=y+%3D+f%28x%29)是模型预测的概率值，![公式](https://www.zhihu.com/equation?tex=t)是样本对应的类标签。

将问题泛化为更一般的情况，多分类问题：

![](https://www.zhihu.com/equation?tex=p%28t%7Cx%29+%3D+%5Cprod_%7Bi%3D1%7D%5E%7BC%7DP%28t_i%7Cx%29%5E%7Bt_i%7D+%3D+%5Cprod_%7Bi%3D1%7D%5E%7BC%7Dy_i%5E%7Bt_i%7D)

由于连乘可能导致最终结果接近0的问题，一般对似然函数取对数的负数，变成最小化对数似然函数。

![](https://www.zhihu.com/equation?tex=-log%5C+p%28t%7Cx%29+%3D+-log+%5Cprod_%7Bi%3D1%7D%5E%7BC%7Dy_i%5E%7Bt_i%7D+%3D+-%5Csum_%7Bi+%3D+i%7D%5E%7BC%7D+t_%7Bi%7D+log%28y_%7Bi%7D%29)


**交叉熵**

说交叉熵之前先介绍相对熵，相对熵又称为KL散度（Kullback-Leibler Divergence），用来衡量两个分布之间的距离，记为![](https://www.zhihu.com/equation?tex=D_%7BKL%7D%28p%7C%7Cq%29)

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Bsplit%7DD_%7BKL%7D%28p%7C%7Cq%29+%26%3D+%5Csum_%7Bx+%5Cin+X%7D+p%28x%29+log+%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7D+%5C%5C%26+%3D%5Csum_%7Bx+%5Cin+X%7Dp%28x%29log+%5C+p%28x%29+-+%5Csum_%7Bx+%5Cin+X%7Dp%28x%29log+%5C+q%28x%29+%5C%5C%26+%3D-H%28p%29+-+%5Csum_%7Bx+%5Cin+X%7Dp%28x%29log%5C+q%28x%29%5Cend%7Bsplit%7D)

这里![公式](https://www.zhihu.com/equation?tex=H%28p%29)是![公式](https://www.zhihu.com/equation?tex=p)的熵。

假设有两个分布![公式](https://www.zhihu.com/equation?tex=p)和![公式](https://www.zhihu.com/equation?tex=q)，它们在给定样本集上的交叉熵定义为：

![](https://www.zhihu.com/equation?tex=CE%28p%2C+q%29+%3D+-%5Csum_%7Bx+%5Cin+X%7Dp%28x%29log%5C+q%28x%29+%3D+H%28p%29+%2B+D_%7BKL%7D%28p%7C%7Cq%29)

回到我们多分类的问题上，真实的类标签可以看作是分布，对某个样本属于哪个类别可以用One-hot的编码方式，是一个维度为![](https://www.zhihu.com/equation?tex=C)的向量，比如在5个类别的分类中，[0, 1, 0, 0, 0]表示该样本属于第二个类，其概率值为1。我们把真实的类标签分布记为![](https://www.zhihu.com/equation?tex=p)，该分布中![](https://www.zhihu.com/equation?tex=t_i+%3D+1)，当![](https://www.zhihu.com/equation?tex=i)属于它的真实类别![](https://www.zhihu.com/equation?tex=c)。同时，分类模型经过softmax函数之后，也是一个概率分布，因为![](https://www.zhihu.com/equation?tex=%5Csum_%7Bi+%3D+1%7D%5E%7BC%7D%7By_i%7D+%3D+1)，所以我们把模型的输出的分布记为![](https://www.zhihu.com/equation?tex=q)，它也是一个维度为![](https://www.zhihu.com/equation?tex=C)的向量，如[0.1, 0.8, 0.05, 0.05, 0]。对一个样本来说，真实类标签分布与模型预测的类标签分布可以用交叉熵来表示：

![](https://www.zhihu.com/equation?tex=l_%7BCE%7D+%3D+-%5Csum_%7Bi+%3D+1%7D%5E%7BC%7Dt_i+log%28y_i%29)

可以看出，该等式于上面对数似然函数的形式一样！

最终，对所有的样本，我们有以下loss function：

![](https://www.zhihu.com/equation?tex=L+%3D+-%5Csum_%7Bk+%3D+1%7D%5E%7Bn%7D%5Csum_%7Bi+%3D+1%7D%5E%7BC%7Dt_%7Bki%7D+log%28y_%7Bki%7D%29)

其中![公式](https://www.zhihu.com/equation?tex=t_%7Bki%7D)是样本![公式](https://www.zhihu.com/equation?tex=k)属于类别![公式](https://www.zhihu.com/equation?tex=i)的概率，![公式](https://www.zhihu.com/equation?tex=y_%7Bki%7D)是模型对样本![公式](https://www.zhihu.com/equation?tex=k)预测为属于类别![公式](https://www.zhihu.com/equation?tex=i)的概率。

**求导**

对单个样本来说，loss function![公式](https://www.zhihu.com/equation?tex=l_%7BCE%7D)对输入![公式](https://www.zhihu.com/equation?tex=a_j)的导数为：

![](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+l_%7BCE%7D%7D%7B%5Cpartial+a_j%7D+%3D+-%5Csum_%7Bi+%3D+1%7D%5E%7BC%7D%5Cfrac+%7B%5Cpartial+t_i+log%28y_i%29%7D%7B%5Cpartial%7Ba_j%7D%7D+%3D+-%5Csum_%7Bi+%3D+1%7D%5E%7BC%7Dt_i+%5Cfrac+%7B%5Cpartial+log%28y_i%29%7D%7B%5Cpartial%7Ba_j%7D%7D+%3D+-%5Csum_%7Bi+%3D+1%7D%5E%7BC%7Dt_i+%5Cfrac%7B1%7D%7By_i%7D%5Cfrac%7B%5Cpartial+y_i%7D%7B%5Cpartial+a_j%7D)


上面对![公式](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7By_%7Bi%7D%7D%7D%7B%5Cpartial%7Ba_%7Bj%7D%7D%7D)求导结果已经算出：

当![公式](https://www.zhihu.com/equation?tex=i+%3D+j)时：![公式](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7By_%7Bi%7D%7D%7D%7B%5Cpartial%7Ba_%7Bj%7D%7D%7D+%3D+y_i%281+-+y_j%29)

当![公式](https://www.zhihu.com/equation?tex=i+%5Cne+j)时：![公式](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7By_%7Bi%7D%7D%7D%7B%5Cpartial%7Ba_%7Bj%7D%7D%7D+%3D+-y_iy_j)

所以，将求导结果代入上式：

![](https://www.zhihu.com/equation?tex=%5Cbegin%7Bsplit%7D-%5Csum_%7Bi+%3D+1%7D%5E%7BC%7Dt_i+%5Cfrac%7B1%7D%7By_i%7D%5Cfrac%7B%5Cpartial+y_i%7D%7B%5Cpartial+a_j%7D%26%3D+-%5Cfrac%7Bt_i%7D%7By_i%7D%5Cfrac%7B%5Cpartial+y_i%7D%7B%5Cpartial+a_i%7D+-+%5Csum_%7Bi+%5Cne+j%7D%5E%7BC%7D+%5Cfrac%7Bt_i%7D%7By_i%7D%5Cfrac%7B%5Cpartial+y_i%7D%7B%5Cpartial+a_j%7D+%5C%5C%26+%3D+-%5Cfrac%7Bt_j%7D%7By_i%7Dy_i%281+-+y_j%29+-+%5Csum_%7Bi+%5Cne+j%7D%5E%7BC%7D+%5Cfrac%7Bt_i%7D%7By_i%7D%28-y_iy_j%29+%5C%5C%26+%3D+-t_j+%2B+t_jy_j+%2B+%5Csum_%7Bi+%5Cne+j%7D%5E%7BC%7Dt_iy_j+%3D+-t_j+%2B+%5Csum_%7Bi+%3D+1%7D%5E%7BC%7Dt_iy_j+%5C%5C%26+%3D+-t_j+%2B+y_j%5Csum_%7Bi+%3D+1%7D%5E%7BC%7Dt_i+%3D+y_j+-+t_j%5Cend%7Bsplit%7D)

```python
y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = nd.array([0, 2], dtype='int32')
nd.pick(y_hat, y)
# pick函数
#[0.1 0.5]
#<NDArray 2 @cpu(0)>

def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()
```

**18. 直接按照softmax运算的数学定义来实现softmax函数。这可能会造成什么问题？（提⽰：试⼀试计算exp(50)的⼤小。）**

**答：**
会溢出，求exp(x)会溢出了：

一种简单有效避免该问题的方法就是让exp(x)中的x值不要那么大或那么小，在softmax函数的分式上下分别乘以一个非零常数：
**19. cross_entropy函数是按照“softmax回归”⼀节中的交叉熵损失函数的数学定义实现的。这样的实现⽅式可能有什么问题？（提⽰：思考⼀下对数函数的定义域。）**

**答：** 对数函数的定义域是（0.+&），当无限接近于0 .可能导致结果过大为nan。主要是因为log()函数中出现了零值，因此在计算交叉熵时，在log()中加一极小值，如log(x+e-5)

**20. 了解最大似然估计。它与最小化交叉熵损失函数有哪些异曲同工之妙**

**答：**
伯努利分布下的最大似然估计推导出交叉熵损失函数，高斯分布下的最大似然估计推导出均方误差损失函数.[参考](https://blog.csdn.net/zgcr654321/article/details/85204049)
***

**21. 为什么交叉熵损失可以提高具有sigmoid和softmax输出的模型的性能，而使用均方误差损失则会出现很多问题？**

**答：**
交叉熵作为损失函数还有一个好处是使用sigmoid函数在梯度下降时能避免均方误差损失函数学习速率降低的问题(sigmiod函数在z值很大或很小的时候几乎不变，也就是梯度接近零，如果用最小二乘不会解决这个梯度消失问题，故不选择最小二乘损失)。 [参考](https://blog.csdn.net/u014313009/article/details/51043064).

交叉熵函数的形式是−[ylna+(1−y)ln(1−a)]，而不是 −[alny+(1−a)ln(1−y)]，为什么？因为当期望输出的y=0时，lny没有意义；当期望y=1时，ln(1-y)没有意义。而因为a是sigmoid函数的实际输出，永远不会等于0或1，只会无限接近于0或者1，因此不存在这个问题。[参考](https://blog.csdn.net/guoyunfei20/article/details/78247263)

**22. batch_size参数变化的影响**

**答：**
适当增加batch_size的好处：

1). 增加内存使用率，大矩阵乘法的并行化效率提高

2).  跑完一次epoch（全数据）所需迭代次数变少。对相同数据量的数据处理速度变快了，减少训练时间

3).在一定范围内，batch_size越大，其下降方向越准，引起训练震荡越小

盲目增加batch_size的坏处：

1). 内存使用率增加了，内存容量要求提升了

2).  跑完一次epoch（全数据）所需迭代次数变少。参数更新次数变少，修正时间变长

3).会陷入局部最小值，导致模型泛化能力下降，小batch_size更有随机性，小的batchsize带来的噪声有助于逃离局部最小值。

学习率和batchsize的关系 ：通常当我们增加batchsize为原来的N倍时，要保证经过同样的样本后更新的权重相等，按照线性缩放规则，学习率应该增加为原来的N倍[5]。但是如果要保证权重的方差不变，则学习率应该增加为原来的sqrt(N)倍。衰减学习率可以通过增加batchsize来实现类似的效果 。（1）如果增加了学习率，那么batch size最好也跟着增加，这样收敛更稳定。（2)尽量使用大的学习率，因为很多研究都表明更大的学习率有利于提高泛化能力。[参考](https://www.zhihu.com/question/32673260)


**23. 应⽤链式法则，推导出sigmoid函数和tanh函数的导数的数学表达式**

**答：**

**24. 参数初始化的选取，权重初始化选取0左右的随机数，不能取0，偏置可以取0**

**答：**
```python
W1 = nd.random.normal(scale=0.01,shape=(num_inputs,num_hiddens))
b1 = nd.zeros(num_hiddens)
```
通过运行help(nd.random_normal)了解参数的意义可知，scale实际上对应的产生随机数的标准差，即std，我记得NG讲过W的初始化的mean分布在0，std=1/sqrt(n)，n为units的个数，所以本题的W的初始权重在1/16左右是没有问题，因此修改weight_scale=0.1/0.01都是可以运行，当为1时，太大了，导致梯度太大，SGD无法运行。

[参数初始化](https://zhuanlan.zhihu.com/p/21560667?refer=intelligentunit)

**25. 改变超参数num_hiddens的值，看看对实验结果有什么影响**

**答：**

epoch 设为100，num_hiddens 由256改成32 ,loss 会下降比较慢，而 train accuracy 会上升比较慢

![25](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/25.jpeg)

**26. 加⼊⼀个新的隐藏层，看看对实验结果有什么影响**

**答：**

epoch 设为100；增加 一层 num_hiddens =256/128的隐藏层：发现增加一层，迭代100次之后，loss更容易 比不增加一层的loss更容易收敛，train accuracy增加得更多 ；另外 num_hiddens 为256/128 影响并不大。

![26](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/26.jpeg)

增大hidden输出，增加隐藏层，都增加了模型容量，也能观察到训练误差和测试误差相对都有降低。

1、隐藏层的话，设置的每一层的units数量应该逐层递减，比如hidden1 =256,后面可以一次为 128 64 32. 或者直接
hidden 2=64(32)。 这样

2、隐藏层过多的话训练会比较慢，因此epoch次数应该多一点

3、隐藏层太多，导致梯度爆炸或者梯度消失。从而产生nan

**27. K折交叉检验**

**答：**
由于验证数据集不参与模型训练，当训练数据不够⽤时，预留⼤量的验证数据显得太奢侈。⼀种改善的⽅法是K折交叉验证（K-fold cross-validation）。在K折交叉验证中，我们把原始训练数据集分割成K个不重合的⼦数据集，然后我们做K次模型训练和验证。每⼀次，我们使⽤⼀个⼦数据集验证模型，并使⽤其他K − 1个⼦数据集来训练模型。在这K次训练和验证中，每次⽤来验证模型的⼦数据集都不同。最后，我们对这K次训练误差和验证误差分别求平均。

**28. 如果用一个三阶多项式模型来拟合一个线性模型生成的数据，可能会有什么问题？为什么？**

**答：**
可能会发生过拟合。但是要看线性模型生成多少个点，如果点非常少，例如小于等于4，那么3次模型会有可能严重过拟合，在训练集上loss可以降为0，但是在测试集上表现很差。但是如果数据点非常多的话，例如1000个点，3次模型来你和还是不错的，因为高阶项的系数基本都是趋近于0的。因此在测试集上表现也不会很差的

**29. 如果用一个三阶多项式模型来拟合一个线性模型生成的数据，可能会有什么问题？为什么？**

**答：**
没有可能。除非这1000个样本中只有小于等于4个点不共线，这种情况才会使得loss为0，因为3次多项式最多可以完全你和4个不共线的点。

**30. 应对过拟合的常用方法**

**答：**
    -  正则化
