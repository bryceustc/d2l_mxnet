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
