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

**5. batch_size, epochs, iteration三个概念区别**

**答：**
（1）batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；

（2）iteration：1个iteration等于使用batchsize个样本训练一次；

（3）epoch：1个epoch等于使用训练集中的全部样本训练一次，即所有的训练样本完成一次Forword运算以及一次BP运算；

一次epoch 总处理数量 = iterations次数 * batch_size大小
简单一句话说就是，我们有2000个数据，分成4个batch，那么batch size就是500。运行所有的数据进行训练，完成1个epoch，需要进行4次iterations。

**6. 权重和偏置的实际意义**

**答：**
![2](https://github.com/bryceustc/d2l_mxnet/blob/master/Images/2.png)

**7. 权重和偏置的实际意义**

**答：**
