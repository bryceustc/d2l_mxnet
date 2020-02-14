# 动手学习深度学习笔记（Dive into Deep Learning，D2L）

**1. 在MXNet中，NDArray是⼀个类，也是存储和变换数据的主要⼯具。NdArray的一些常见数据操作与运算。**
**答：**
```python
from mxnet import nd

x = nd.arange(12)  #[ 0. 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11.]
                   # <NDArray 12 @cpu(0)>
x.shape  #(12,)
x.size   # 12
X = x.reshape((3, 4)) # [[ 0. 1. 2. 3.]
                      #  [ 4. 5. 6. 7.]
                      #  [ 8. 9. 10. 11.]]
                      # <NDArray 3x4 @cpu(0)>                      
nd.zeros((2, 3, 4)) # 各元素为0，形状为(2, 3, 4)的张量
nd.ones((3, 4)) #各元素为1的张量
Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]) #通过Python的列表（list）指定需要创建的NDArray中每个元素的值
nd.random.normal(0, 1, shape=(3, 4))  #每个元素都随机采样于均值为0、标准差为1的正态分布
X + Y
X * Y
X / Y
Y.exp()
nd.dot(X, Y.T)

nd.concat(X, Y, dim=0), nd.concat(X, Y, dim=1)  #将多个NDArray连结（concatenate）。下⾯分别在⾏上（维度0，即形状中的最左边元素）
                                                #和列上（维度1，即形状中左起第⼆个元素）连结两个矩阵
                                               
X == Y
X.sum()  #对NDArray中的所有元素求和得到只有⼀个元素的NDArray。非标量注意
X.norm().asscalar() # 通过asscalar函数将结果变换为Python中的标量
#我们也可以把Y.exp()、X.sum()、X.norm()等分别改写为nd.exp(Y)、nd.sum(X)、nd.norm(X)等。
```

**2. 当对两个形状不同的NDArray按元素运算时，可能会触发⼴播（broadcasting）机制：先适当复制元素使这两个NDArray形状相同后再按元素运算。**
**答：**
```python
X = nd.arange(3).reshape((3,1))

# [[0.]
#  [1.]
#  [2.]]
# <NDArray 3x1 @cpu(0)>

Y = nd.arange(2).reshape((1,2))
# [[0. 1.]]
# <NDArray 1x2 @cpu(0)>)

X+Y 
#[[0. 1.]
#[1. 2.]
#[2. 3.]]
#<NDArray 3x2 @cpu(0)>
```
