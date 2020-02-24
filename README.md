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

一种简单有效避免该问题的方法就是让exp(x)中的x值不要那么大或那么小，在softmax函数的分式上下分别乘以一个非零常数

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

**29. 的三阶多项式拟合问题⾥，有没有可能把100个样本的训练误差的期望降到0，为什么？（提⽰：考虑噪声项的存在。）**

**答：**
没有可能。除非这1000个样本中只有小于等于4个点不共线，这种情况才会使得loss为0，因为3次多项式最多可以完全你和4个不共线的点。

**30. 回顾⼀下训练误差和泛化误差的关系。除了权重衰减（正则化）、增⼤训练量以及使⽤复杂度合适的模型，你还能想到哪些办法来应对过拟合？**

**答：**
    -  正则化


**31. 贝叶斯估计、最大似然估计(MLE)、最大后验概率估计(MAP)概念理解**

**答：**
概率是已知模型和参数，推数据。统计是已知数据，推模型和参数。频率学派的代表是最大似然估计；贝叶斯学派的代表是最大后验概率估计。

![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_1.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_2.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_3.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_4.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_5.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_6.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_7.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_8.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_9.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_10.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_11.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_12.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_13.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_14.jpg)
![1](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/31_15.jpg)

[参考](https://www.jianshu.com/p/9c153d82ba2d)

**32. 如果你了解⻉叶斯统计，你觉得权重衰减对应⻉叶斯统计⾥的哪个重要概念？**

**答：**

**先验（prior）**

从贝叶斯的角度来看，正则化等价于对模型参数引入 先验分布 。L2正则化对应于参数是服从高斯分布的先验假设,L1对应拉普拉斯分布。

![32](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/32.png)

**33. drop out防止过拟合的原因**

**答：**
防止参数过分依赖训练数据，增加参数对数据集的泛化能力，因为在实际训练的时候，每个参数都有可能被随机的Drop掉，所以参数不会过分的依赖某一个特征的数据，而且不同参数之间的相互关联性也大大减弱，这些操作都可以增加泛化能力。

CNN训练过程中使用dropout是在每次训练过程中随机将部分神经元的权重置为0，即让一些神经元失效，这样可以缩减参数量，避免过拟合，关于dropout为什么有效，有两种观点：1）每次迭代随机使部分神经元失效使得模型的多样性增强，获得了类似多个模型ensemble的效果，避免过拟合  2）dropout其实也是一个data augmentation的过程，它导致了稀疏性，使得局部数据簇差异性更加明显，这也是其能够防止过拟合的原因。
dropout率的选择： 经过交叉验证，隐含节点dropout率等于0.5的时候效果最好，原因是0.5的时候dropout随机生成的网络结构最多。
dropout也可以被用作一种添加噪声的方法，直接对input进行操作。输入层设为更接近1的数。使得输入变化不会太大（0.8）

[参考](https://zhuanlan.zhihu.com/p/21560667?refer=intelligentunit)

**34.如果把本节中的两个丢弃概率超参数对调，会有什么结果？**

**答：**
效果变差一些，drop_pro1 = 0.2, drop_pro2 = 0.5，第一层是与原始数据相连，drop概率变大，所丢失信息变多，所以效果差一点。

**35.增⼤迭代周期数，⽐较使⽤丢弃法与不使⽤丢弃法的结果**

**答：**
不使用dropout时，训练集的准确度略有上升，而测试集的准确度略有下降。

**36. 以本节中的模型为例，⽐较使⽤丢弃法与权重衰减的效果。如果同时使⽤丢弃法和权重衰减，效果会如何？**

**答：**
使用如下代码做了尝试，发现同时使用丢弃法和权重衰减，效果很差，分类结果近似于在10个类别中随机猜测一种，测试精度约等于1/10。
被丢弃的权重对Loss的导数是0，参数更新之后，这一项权重会变成接近0，退回到类似于刚初始化时的权重。

如果非得同时使用丢弃法和权重衰减，那就需要修改权重衰减的范数惩罚项，计算范数惩罚项时只考虑没有被drop的那些权重。

**37. L1、L2、Batch Normalization、Dropout为什么能够防止过拟合呢？**

**答：**

![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/37_1.jpg)
![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/37_2.jpg)
![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/37_3.jpg)
[参考](https://blog.csdn.net/qq_29462849/article/details/83068421)
**38. 反向传播算法（过程及公式推导）**

**答：**

[反向传播算法（过程及公式推导）](https://blog.csdn.net/u014313009/article/details/51039334)

**39 有⼈说随机初始化模型参数是为了“打破对称性”。这⾥的“对称”应如何理解？**

**答：**

对称性指的是在进行梯度下降更新时更新变化量不能全部每次都相同，达不到训练模型的效果。

**40.是否可以将线性回归或softmax回归中所有的权重参数都初始化为相同值？**

**答：**
回到之前的例子里面尝试了一下，线性回归设相同值对结果影响不大。softmax设相同值，分类准确率一直是0.1

可以这样理解，权重相同，隐藏层的神经元相当于单个神经元，此时神经网络相当于线性分类器。所以线性回归可以设相同值，而softmax是非线性分类器，不能设相同的值。

**41.如果不在 MLP 类的 __init__ 函数里调用父类的 __init__ 函数，会出现什么样的错误信息？(super函数的用法，\*args 和 \*\*kwargs 的区别)**

**答：**

![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/41-2.jpg)

会出现如下错误信息。

![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/41.jpg)

调用MLP(）时，如果不继承父类的__init__函数，则因为初始化没有继承父类初始化过程中的一些参数，而无法正常使用父类的函数，也就无法继续进行向前传播，进而无法训练网络。

**42.如果去掉 FancyMLP 类里面的 asscalar 函数，会有什么问题？**

**答：**

x.norm().asscalar() 返回的是 “True” or “False”，

x.norm() 返回的是 “[1.]” or “[0.]”，

不会出现问题，不使用asscalar函数时判断条件结果为0或者1，使用asscalar函数判断结果为True或者False，然而while和if这两种方式都能正确决断，所以我觉得不会出现问题。

**43.如果将NestMLP类中通过Sequential实例定义的self.net改为self.net = [nn.Dense(64, activation='relu'), nn.Dense(32, activation='relu')]，会有什么问题？**

**答：**
根据题目将代码修改后，会报错，因为修改后会变成list类型，而不是Block， 这样就不会被自动注册到 Block 类的 self.\_children 属性, 导致 initialize 时在 self.\_children 找不到神经元, 无法初始化参数. 知道原因后，修改代码如下，到网络改成下面的结构：（第一张图片是修改后的代码，第二张图片是修改后代码的网网络结构，第三张是原来网络结构）显然可以看出来后面两层没有改变，只有前三层在使用不同类进行构造网络。

![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/42_1.png)
![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/42_2.png)
![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/42_3.png)

**44.如何对每个层使用不同的初始化函数 （整理的方法）**

**答：** 
1.先构建网络，重新为每一层初始化
2.构建网络时，为每一层初始化
https://discuss.gluon.ai/t/topic/987/23


**45.尝试在net.initialize()后、net(X)前访问模型参数，观察模型参数的形状。**

**答：** 模型参数的形状有一个维度为 0。

因为不知道输入数据的维度，无法为参数开辟空间。延迟初始化。

**46.构造⼀个含共享参数层的多层感知机并训练。在训练过程中，观察每⼀层的模型参数和梯度**

**答：**
![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/46_1.jpg)

前向传播会将参数乘两次，所以在反传是也应该分别求出梯度，分别更新参数，但是第二次会覆盖第一次的结果，所以我们只能看到一个值。

**47.手写⼆维卷积运算，二维卷积层 Python代码**

**答：**
```python
from mxnet import nd, autograd
from mxnet.gluon import nn


def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros(shape=(X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum()
    return Y


class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape= kernel_size)
        self.bias = self.params.get('bias', shape = (1,))


    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()

```

卷积层的简单应⽤：检测图像中物体的边缘，即找到像素变化的位置，卷积层可通过重复使⽤卷积核有效地表征局部空间

**48.卷积运算与互相关运算**

**答：**

![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/48.png)
![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/48_2.jpg)
[参考](https://blog.csdn.net/yjinyyzyq/article/details/91867123)

**49.构造⼀个输⼊图像X，令它有⽔平⽅向的边缘。如何设计卷积核K来检测图像中⽔平边缘？如果是对⻆⽅向的边缘呢？**

**答：**

这篇博文介绍了很多边缘检测卷积核：https://blog.csdn.net/zlsjsj/article/details/80057312

水平方向的边缘卷积核：
```python
X = nd.ones((8, 6))
X[2:6, :] = 0
print(X)
K = nd.array([[1], [-1]])
Y = corr2d(X, K)
print(Y)

### 
X:
[[1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1.]
 [0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0.]
 [1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1.]]
<NDArray 8x6 @cpu(0)>

Y：
[[ 0.  0.  0.  0.  0.  0.]
 [ 1.  1.  1.  1.  1.  1.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [-1. -1. -1. -1. -1. -1.]
 [ 0.  0.  0.  0.  0.  0.]]
<NDArray 7x6 @cpu(0)>
```
对角方向的边缘卷积核：
```python
X = nd.eye(6, 6)
print(X)
K = nd.array([[1, -1], [-1, 1]])
Y = corr2d(X, K)
print(Y)

###
X:
[[1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1.]]
<NDArray 6x6 @cpu(0)>

Y:
[[ 2. -1.  0.  0.  0.]
 [-1.  2. -1.  0.  0.]
 [ 0. -1.  2. -1.  0.]
 [ 0.  0. -1.  2. -1.]
 [ 0.  0.  0. -1.  2.]]
<NDArray 5x5 @cpu(0)>

```

**50. 试着对我们⾃⼰构造的Conv2D类进⾏⾃动求梯度，会有什么样的错误信息？在该类的forward函数⾥，将corr2d函数替换成nd.Convolution类使得⾃动求梯度变得可⾏**

**答：**
虽然我们之前构造了Conv2D类，但由于corr2d使用了对单个元素赋值（[i, j]=）的操作会导致无法自动求导，下面我们使用Gluon提供的Conv2D类来实现这个例子。

corr2d因为用了[i,j]=导致自动求导失败，具体错误如下
```
Inplace operations (+=, -=, x[:]=, etc) are not supported when recording with autograd.
```

将corr2d函数替换成nd.Convolution类使得⾃动求梯度变得可⾏

```python
# e.x.2 在 Conv2D 的 forward 函数⾥，将 corr2d 替换成 nd.Convolution 使得其可以求导。
class Conv2D_ex2(nn.Block):
  """
    - **data**: *(batch_size, channel, height, width)*
    - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_height, out_width)*.
  """

  def __init__(self, channels, kernel_size, **kwargs):
    super().__init__(**kwargs)
    self.weight = self.params.get(
        'weight', shape=(channels, 1,) + kernel_size)
    self.bias = self.params.get('bias', shape=(channels, ))
    self.num_filter = channels
    self.kernel_size = kernel_size

  def forward(self, x):
    return nd.Convolution(
        data=x, weight=self.weight.data(), bias=self.bias.data(), num_filter=self.num_filter, kernel=self.kernel_size)

```


**51.  如何通过变化输⼊和核数组将互相关运算表⽰成⼀个矩阵乘法？**

**答：**

为了加速运算啊，传统的卷积核依次滑动的计算方法很难加速。转化为矩阵乘法之后，就可以调用各种线性代数运算库，CUDA里面的矩阵乘法实现。这些矩阵乘法都是极限优化过的，比暴力计算快很多倍。

[二维离散卷积转换为矩阵相乘——卷积与反卷积](https://howardlau.me/machine-learning/convolution-to-matrix-multiplication.html)

![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/51.jpg)
![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/51_2.jpg)
![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/51_3.jpg)

**52.  卷积输出形状**

**答：**

![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/52.jpg)

**53. 1 × 1卷积层被当作保持⾼和宽维度形状不变的全连接层使⽤。于是，我们可以通过调整⽹络层之间的通道数来控制模型复杂度**

**答：**

![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/53.jpg)

**54.假设输⼊形状为ci×h×w，且使⽤形状为co×ci×kh×kw、填充为(ph, pw)、步幅为(sh, sw)的卷积核。那么这个卷积层的前向计算分别需要多少次乘法和加法**

**答：**

![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/54.png)
![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/54_1.png)

**55.⽤矩阵乘法实现卷积计算**

**答：**

[二维离散卷积转换为矩阵相乘——卷积与反卷积](https://howardlau.me/machine-learning/convolution-to-matrix-multiplication.html)

![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/51.jpg)
![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/51_2.jpg)
![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/51_3.jpg)

**56. 池化层**

**答：**
-  最⼤池化和平均池化分别取池化窗口中输⼊元素的最⼤值和平均值作为输出。
-  池化层的⼀个主要作⽤是缓解卷积层对位置的过度敏感性。
-  可以指定池化层的填充和步幅。
-  池化层的输出通道数跟输⼊通道数相同。

**57. 分析池化层的计算复杂度。假设输⼊形状为c × h × w，我们使⽤形状为ph × pw的池化窗口，而且使⽤(ph, pw)填充和(sh, sw)步幅。这个池化层的前向计算复杂度有多⼤？**

**答：**
池化层的前向计算复杂度 c * ( (h - p_h + p_h + s_h) / s_h * (w - p_w + p_w + s_w) / s_w)

**58. 想⼀想，最⼤池化层和平均池化层在作⽤上可能有哪些区别？**

**答：**

最大池化层，增强图片亮度；平均池化层，减少冲击失真，模糊，平滑。

**59. 你觉得最小池化层这个想法有没有意义？**

**答：**
最大池化的意义是在于寻找该区域内最突出的特征；最小池化也就是寻找最不明显的特征，可用于图像去噪，模糊化等应用。

filter卷积的过程是勾勒图片最小单元（特征）的过程。其输出值越大，表示该位置越贴近filter所代表的特征。而MaxPooling的过程实在降低特征在图像位置上的精确程度。表示的是在该片区域存在该特征。如果MinPooling的话，岂不是在该区域全部都是该特征？
那么MinPooling也许在做一些特殊的图像分类时会有奇效？ 比方碧空无云的蓝天；无疾病、疤痕或色素痣的皮肤；无暇美玉；
使用场景大概相对有限吧。

**60. 问题：学习率太大会导致无法收敛。**

**解决：学习率调整技巧：**

优先使用Adam优化算法，此算法会自动调整学习率，以适用模型

当loss值忽大忽小或者保持不变，学习率过大

当loss值在减小但是幅度很小，学习率过小

**61.CUDA之nvidia-smi命令详解**

https://blog.csdn.net/Bruce_0712/article/details/63683787

**62. 参考VGG论⽂⾥的表1来构造VGG其他常⽤模型，如VGG-16和VGG-19**

**答：**

![](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/61.jpg)

```python
from mxnet import gluon, init, nd
from mxnet.gluon import nn


def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk


conv_arch_vgg11 = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
conv_arch_vgg16 = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
conv_arch_vgg19 = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))

def vgg(conv_arch):
    net = nn.Sequential()
    # 卷积层部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # 全连接层部分
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net


net = vgg(conv_arch_vgg19)
net.initialize()
X = nd.random.uniform(shape=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

**63. NiN网络以及1×1卷积层的作用**

**答：**

![63](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/63.jpg)


如果设置kernel_size大小为1x1，则可以实现多个feature map的线性组合，实现跨通道的信息整合的功效，这也是mlpconv的由来，它将传统的conv中特征的“单层”线性升级为非线性的“多层”抽象表达，类比于在conv层中实现从单层感知机到多层感知机的变换。

而恰好设置kernel_size大小为1x1，可以满足这个目的：实现多个feature map的线性组合这一功效。原因如下图：

![63](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/63_2.png)


![63](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/63_3.png)

经过第一层时，只是单纯的生成各自的feature map：f1，f2。第二次后，f1和f2作为input，在kernel1x1的作用下，可以认为f1和f2线性组合成f3，注意这里是线性组合。f4的生成雷同。

1.NIN两大特性：

 -  mlpconv
 -  平均池化层
 
2.由此，引发出1x1卷积核的作用：

 -  实现跨通道的交互和信息整合
 -  进行卷积核通道数的降维和升维，减少网络参数
 
**64.对⽐AlexNet、VGG和NiN、GoogLeNet的模型参数尺⼨。为什么后两个⽹络可以显著减小 模型参数尺⼨？**

**答：** 

VGG与Alexnet相比，具有如下改进几点：

1、去掉了LRN层，作者发现深度网络中LRN的作用并不明显，干脆取消了

2、采用更小的卷积核-3x3，Alexnet中使用了更大的卷积核，比如有7x7的，因此VGG相对于Alexnet而言，参数量更少

3、池化核变小，VGG中的池化核是2x2，stride为2，Alexnet池化核是3x3，步长为2

  ![64](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/64.jpg)

  ![64](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/64_2.jpg)

GoogLeNet借鉴了NIN的思想，大量使用1x1的卷积层，同时也有创新，一个inception同时使用多个不同尺寸的卷积层，以一种结构化的方式来捕捉不同尺寸的信息，很大程度地降低了参数量和计算量。

**65.在模型训练时，批量归一化利用小批量上的均值和标准差，不断调整神经网络的中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。**

BN中使用的均值和方差最好是总体的均值和方差。总体是未知的，所以只能去估计总体的方差。
理论上来说，什么时候用哪个均值、哪个方差是两方面的考虑：1.对总体均值和反差的估计的置信度 2.计算的方便

当然实际上来说，就是哪个效果好用哪个。

至于为什么测试集的时候使用移动平均，教程里也有说明：

 -  不用的话，训练出的模型参数很可能在测试时就不准确了；
 -  用的话，万一测试的数据就只有一个数据实例就不好办了。
 
 **66.为什么对全连接层做BN时候只计算列方向的？**
 
 **答：** 这是数据的原因。当维度为2的时候，维度是(batch, num_features)，每一列都认为是一列特征，不同特征间的分布、相关性都是未知的，所以一般不对所有特征做BN，但是可以认为相同的特征来自相同的分布，所以在一个batch里面每一个feature单独做BN。
当维度为4的时候，（batch, channel, height, weight)，不同的channel数据的分布可能不同，所以不对不同通道的数据做BN，同时对于图像数据，都是像素点，会假设相同通道的像素点取自相同的分布，所以对于四维的数据，同一个batch 里面，每个通道单独做BN。

个人理解：不同的BN，基于不同的数据特征假设。

全连接：每个特征都要在小批量上求平均

卷积：每个通道都要在 小批量x高x宽 上求平均

```python
>>> from mxnet import nd
>>> X = nd.arange(120)
>>> X

[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.
  14.  15.  16.  17.  18.  19.  20.  21.  22.  23.  24.  25.  26.  27.
  28.  29.  30.  31.  32.  33.  34.  35.  36.  37.  38.  39.  40.  41.
  42.  43.  44.  45.  46.  47.  48.  49.  50.  51.  52.  53.  54.  55.
  56.  57.  58.  59.  60.  61.  62.  63.  64.  65.  66.  67.  68.  69.
  70.  71.  72.  73.  74.  75.  76.  77.  78.  79.  80.  81.  82.  83.
  84.  85.  86.  87.  88.  89.  90.  91.  92.  93.  94.  95.  96.  97.
  98.  99. 100. 101. 102. 103. 104. 105. 106. 107. 108. 109. 110. 111.
 112. 113. 114. 115. 116. 117. 118. 119.]

>>> X2 = X.reshape(20, -1)
>>> X2.mean(axis=0)

[57. 58. 59. 60. 61. 62.]
<NDArray 6 @cpu(0)>

>>> X4 = X.reshape(2, 3, 4, 5)
>>> X4

[[[[  0.   1.   2.   3.   4.]
   [  5.   6.   7.   8.   9.]
   [ 10.  11.  12.  13.  14.]
   [ 15.  16.  17.  18.  19.]]

  [[ 20.  21.  22.  23.  24.]
   [ 25.  26.  27.  28.  29.]
   [ 30.  31.  32.  33.  34.]
   [ 35.  36.  37.  38.  39.]]

  [[ 40.  41.  42.  43.  44.]
   [ 45.  46.  47.  48.  49.]
   [ 50.  51.  52.  53.  54.]
   [ 55.  56.  57.  58.  59.]]]


 [[[ 60.  61.  62.  63.  64.]
   [ 65.  66.  67.  68.  69.]
   [ 70.  71.  72.  73.  74.]
   [ 75.  76.  77.  78.  79.]]

  [[ 80.  81.  82.  83.  84.]
   [ 85.  86.  87.  88.  89.]
   [ 90.  91.  92.  93.  94.]
   [ 95.  96.  97.  98.  99.]]

  [[100. 101. 102. 103. 104.]
   [105. 106. 107. 108. 109.]
   [110. 111. 112. 113. 114.]
   [115. 116. 117. 118. 119.]]]]
<NDArray 2x3x4x5 @cpu(0)>

>>> X4.mean(axis=(0, 2, 3), keepdims=True)

[[[[39.5]]

  [[59.5]]

  [[79.5]]]]
<NDArray 1x3x1x1 @cpu(0)>
```

**67. BN作用**

**答：**

BN解决了反向传播过程中的梯度问题（梯度消失和爆炸），同时使得不同scale的w整体更新步调更一致。

{Batch Normalization的作用是通过规范化的手段,将越来越偏的分布拉回到标准化的分布,使得激活函数的输入值落在激活函数对输入比较敏感的区域,从而使梯度变大,加快学习收敛速度,避免梯度消失的问题。}

BN最大的优点为允许网络使用较大的学习速率进行训练，加快网络的训练速度。

**68.能否将批量归⼀化前的全连接层或卷积层中的偏差参数去掉？为什么？（提⽰：回忆批量归⼀化中标准化的定义。）**

**答：**可以，求平均值会减去偏差，可以一开始就去掉

**69.为什么不加激活函数的情况下几层卷积和一层卷积没有区别呢?**

**答：**卷积是线性变换，n个叠加还是线性。通过非线性激活可以打断这个。
