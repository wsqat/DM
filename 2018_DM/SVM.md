支持向量机是属于原创性、非组合的具有明显直观几何意义的分类算法，具有较高的准确率。

使用SVM算法的思路：

（1）简单情况，线性可分情况，把问题转化为一个凸优化问题，可以用拉格朗日乘子法简化，然后用既有的算法解决；

（2）复杂情况，线性不可分，用核函数将样本投射到高维空间，使其变成线性可分的情形，利用核函数来减少高纬度计算量。

## 一、SVM相关基本概念

###  分割超平面

设C和D为两不相交的凸集，则存在超平面P，P可以将C和D分离。 

![1](http://img.blog.csdn.net/20160505153425587?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

两个集合的距离，定义为两个集合间元素的最短距离。

做集合C和集合D最短线段的垂直平分线。

![2](http://img.blog.csdn.net/20160506224239999?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

但是， 如何定义两个集合的"最优"分割超平面？找到集合“边界”上的若干点，以这些点为“基础”计算超平面的方向，以两个集合边界上的这些点的平均作为超平面的“截距”。这些点被称作支持向量，点是可用向量方式表示。

![3](http://img.blog.csdn.net/20160506225041348?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


###  输入数据

假设给定一个特征空间上的训练数据集

![](http://img.blog.csdn.net/20160505153402992?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

其中，![](http://img.blog.csdn.net/20160505153724682?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center),Xi为第i个实例（若n>1，即x是多维度，具有多个属性特征，此时
Xi为向量）；Yi为Xi的类标记，当Yi为+1时，称Xi为正例，当Yi为-1时，称Xi为负例。


### 线性可分支持向量机

 给定线性可分训练数据集，通过间隔最大化得到的分离超平面为![](http://img.blog.csdn.net/20160505154947326?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)，相应的分类决策函数![](http://img.blog.csdn.net/20160505160050990?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)该决策函数称为线性可分支持向量机。其中，![](http://img.blog.csdn.net/20160505160221694?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)是某个确定的特征空间转换函数，它的作用是将x映射到（更高的）维度，最简单直接的：![](http://img.blog.csdn.net/20160505160457335?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)。事实上，求解分离超平面问题可以等价为求解相应的凸二次规划问题。



### 整理符号

分割平面：![](http://img.blog.csdn.net/20160505154947326?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

训练集：![](http://img.blog.csdn.net/20160505160955196?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

目标值：![](http://img.blog.csdn.net/20160505161300325?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

新数据的分类：![](http://img.blog.csdn.net/20160505161417105?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

## 二、SVM推导过程

推导目标函数

根据题设![](http://img.blog.csdn.net/20160505154947326?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

有：![](http://img.blog.csdn.net/20160505161937756?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

w,b等比例缩放，则t*y的值同样缩放，从而：
![](http://img.blog.csdn.net/20160505162345093?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

最大间隔分离超平面
目标函数：![](http://img.blog.csdn.net/20160505162748782?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)，表示最近点到直线距离尽可能大

![](http://img.blog.csdn.net/20160505163158194?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

函数间隔和几何间隔

分割平面：![](http://img.blog.csdn.net/20160505154947326?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center) (函数间隔)

总可以通过等比例缩放w的方法，使得两类点的函数值都满足![](http://img.blog.csdn.net/20160505163604675?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

![](http://img.blog.csdn.net/20160505163828297?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

### 建立目标函数

1.总可以通过等比例缩放w的方法，使得两类点的函数值都满足

![](http://img.blog.csdn.net/20160505163604675?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

2.约束条件：

![](http://img.blog.csdn.net/20160505164148302?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

3.原目标函数：

![](http://img.blog.csdn.net/20160505162748782?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

4.新目标函数：

![](http://img.blog.csdn.net/20160505164424446?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

![](http://img.blog.csdn.net/20160505164737779?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

5.目标函数变换一下：

![](http://img.blog.csdn.net/20160505164944978?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

6.拉格朗日乘子法

![](http://img.blog.csdn.net/20160505165350427?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

7.原问题是极小极大问题

![](http://img.blog.csdn.net/20160505165622746?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

原问题的对偶问题是极大极小问题

![](http://img.blog.csdn.net/20160505165843090?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


8.将6中的拉格朗日函数分别对w, b 求偏导并令其为0：

![](http://img.blog.csdn.net/20160505170639521?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

9.计算拉格朗日的对偶函数

![](http://img.blog.csdn.net/20160506190920681?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

10.继续求

![](http://img.blog.csdn.net/20160506192022839?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)的极大

![](http://img.blog.csdn.net/20160506192613457?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

11.整理目标函数：添加负号

![](http://img.blog.csdn.net/20160506192830132?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

12.线性可分支持向量机学习算法
计算结果如下

![](http://img.blog.csdn.net/20160506193759067?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


13.分类决策函数

![](http://img.blog.csdn.net/20160506194005443?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


## 三、线性不可分SVM

1.若数据线性不可分，则增加松弛因子，使函数间隔加上松弛变量大于等于1，

则约束条件变成

![](http://img.blog.csdn.net/20160506194650492?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

目标函数：

![](http://img.blog.csdn.net/20160506194943353?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

（这里是为了保证松弛因子不至于过大）


2.此时的凸优化为

![](http://img.blog.csdn.net/20160506195308651?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

3.拉格朗日函数

![](http://img.blog.csdn.net/20160506200646172?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

4.将三式代入L中，得到

![](http://img.blog.csdn.net/20160506201726629?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


5. 整理，得到对偶问题的最优化问题

![](http://img.blog.csdn.net/20160506202059084?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

求得最优解![](http://img.blog.csdn.net/20160506202407408?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

 6.计算

![](http://img.blog.csdn.net/20160506210637634?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)                   

实践中往往取支持向量的所有值取平均，作为b*

7.求得分离超平面

![](http://img.blog.csdn.net/20160506210944401?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

8.分类决策函数为

![](http://img.blog.csdn.net/20160506212736072?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

核函数：可以使用核函数，将原始输入空间映射到新的特征空间，从而使得原本线性不可分的样本可在核空间可分。

有多项式核函数

![](http://img.blog.csdn.net/20160506213350049?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

高斯核函数RBF 

![](http://img.blog.csdn.net/20160506213619208?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

字符串核函数

在实际应用中，往往依赖先验领域知识或交叉验证等方案才能选择有效的核函数。没有更多先验信息，则使用高斯核函数。


核函数映射：



![](http://img.blog.csdn.net/20160506214126560?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


![](http://img.blog.csdn.net/20160506214145210?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)


高斯核

![](http://img.blog.csdn.net/20160506214359688?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

 粗线是分割超“平面”，其他线是y(x)的等高线，绿色圈点是支持向量点。

高斯核是无穷维的，因为



注：SVM和Logistic回归的比较：
- （1）经典的SVM，直接输出类别，不给出后验概率；
- （2）Logistic回归，会给出属于哪一个类别的后验概率；
- （3）比较重点是二者目标函数的异同。

