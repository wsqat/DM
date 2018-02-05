# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family']='sans-serif'
# plt.rcParams['axes.unicode_minus'] = False

iris=load_iris()
iris_data=iris.data
iris_target=iris.target
import pandas as pd
iris_target1=pd.get_dummies(iris_target).values
print(iris_data.shape)

pca=PCA(n_components=2)

X=pca.fit_transform(iris_data)
print(X.shape)
f=plt.figure()
ax=f.add_subplot(111)
ax.plot(X[:,0][iris_target==0],X[:,1][iris_target==0],'bo')
ax.scatter(X[:,0][iris_target==1],X[:,1][iris_target==1],c='r')
ax.scatter(X[:,0][iris_target==2],X[:,1][iris_target==2],c='y')
ax.set_title(u'数据分布图')
plt.show()


x=tf.placeholder(dtype=tf.float32,shape=[None,2],name="input")
y=tf.placeholder(dtype=tf.float32,shape=[None,3],name="output")

w=tf.get_variable("weight",shape=[2,3],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
bais=tf.get_variable("bais",shape=[3],dtype=tf.float32,initializer=tf.constant_initializer(0))
y_1=tf.nn.bias_add(tf.matmul(x,w),bais)

loss=tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_1))
x0min,x0max=X[:,0].min(),X[:,0].max()
x1min,x1max=X[:,1].min(),X[:,1].max()

with tf.Session() as sess:
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y,1),tf.arg_max(y_1,1)),tf.float32))
    train_step=tf.train.AdamOptimizer().minimize(loss)
    my=tf.arg_max( y_1,1)
    sess.run(tf.global_variables_initializer())
    for i in range(3001):
        sess.run(train_step,feed_dict={x:X,y:iris_target1})
        if i%500==0:
            accuracy_print=sess.run(accuracy,feed_dict={x:X,y:iris_target1})
            print(accuracy_print)

    h=0.05
    xx,yy=np.meshgrid(np.arange(x0min-1,x0max+1,h),np.arange(x1min-1,x1max+1,h))
    x_=xx.reshape([xx.shape[0]*xx.shape[1],1])
    y_=yy.reshape([yy.shape[0]*yy.shape[1],1])
    test_x=np.c_[x_,y_]
    my_p=sess.run(my,feed_dict={x:test_x})
    coef=w.eval()
    intercept=bais.eval()
z=my_p.reshape(xx.shape)
f=plt.figure()
plt.contourf(xx,yy,z, cmap=plt.cm.Paired)
plt.axis('tight')
colors='bry'
for i,color in zip([0,1,2],colors):
    idx=np.where(iris_target==i)
    plt.scatter(X[idx,0],X[idx,1],c=color,cmap=plt.cm.Paired)

xmin,xmax=plt.xlim()

print(intercept)
print(coef)

plt.show()