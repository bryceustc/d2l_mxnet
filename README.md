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

**答：**yield在函数中的功能类似于return，不同的是yield每次返回结果之后函数并没有退出，而是每次遇到yield关键字后返回相应结果，并保留函数当前的运行状态，等待下一次的调用。如果一个函数需要多次循环执行一个动作，并且每次执行的结果都是需要的，这种场景很适合使用yield实现。
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

**答：**要写成 (y_hat - y.reshape(y_hat.shape)) ，保证形状一样能够做减法。不用广播是因为，广播机制生效有一定的前提，首先是两者在其中一个维度上是相同大小的，其次是那个用来广播的矩阵，另一个维度一定是1。


**10. 如果样本个数不能被批量⼤小整除，data_iter函数的⾏为会有什么变化？**

**答：**当你的batch_size大于num_expamples的时候，就需要用到min来确定indices[i:min(i+batch_size,num_examples)]的上界了

**11. 深度学习整的一个流程步骤**

**答：**
1、数据集处理 

2、读取数据
```python
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
# 随机读取⼩批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
```
3、定义模型
```python
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1)) #该层输出个数为1
``` 
4、初始化模型参数
```python
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
``` 
5、定义损失函数 
```python
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss() # 平⽅损失⼜称L2范数损失
``` 
6、定义优化算法 
```python
#导⼊Gluon后，我们创建⼀个Trainer实例，并
#指定学习率为0.03的小批量随机梯度下降（sgd）为优化算法。该优化算法将⽤来迭代net实例所
#有通过add函数嵌套的层所包含的全部参数。这些参数可以通过collect_params函数获取。
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```
7、训练模型 
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

**答：**相当于把计算出来的梯度除以batch size，因为loss.backward()相当于nd.sum(loss).bacward()，也就是把一个batch的loss都加起来求的梯度，所以通过除以batch size能够弱化batch size在更新参数时候的影响。

**13. 在Gluon中，data模块提供了有关数据处理的⼯具，nn模块定义了⼤量神经⽹络的层，loss模块定义了各种损失函数。**

**14. 如何访问dense.weight的梯度？**

**答：**``dense.weight.grad()``

**15. Softmax函数与交叉熵**

**答：** 
在进入softmax函数之前，已经有模型输出$C$值，其中$C$是要预测的类别数，模型可以是全连接网络的输出$a$，其输出个数为$C$，即输出为$a_1, a_2, ..., a_C$。

所以对每个样本，它属于类别$i$的概率为：
![2](http://latex.codecogs.com/gif.latex?y_%7Bi%7D%3D%5Cfrac%7Be%5E%7Ba_%7Bi%7D%7D%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BC%7D%20e%5E%7Ba_%7Bk%7D%7D%7D%20%5Cquad%20%5Cforall%20i%20%5Cin%201%20%5Cldots%20C)

**16. 为什么交叉熵损失可以提高具有sigmoid和softmax输出的模型的性能，而使用均方误差损失则会出现很多问题？**

**答：**

**17. 了解最大似然估计。它与最小化交叉熵损失函数有哪些异曲同工之妙**

**答：**
