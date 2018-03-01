# 数据挖掘
2017_DM: 2017年数据挖掘课程大实验的代码
- dm：机器学习相关模型
- nn：神经网络相关模型

## 一、机器学习

>词云

![cloud.jpg](http://upload-images.jianshu.io/upload_images/688387-0cce52a19efb97c5.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 0、机器学习流程图

![流程图.png](http://upload-images.jianshu.io/upload_images/688387-edcc7aa78c9d64fd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 1、定义问题：挖掘每个站点与所属区域的关系

### 2、特征选择：特征数据集的选取
- 一卡通数据集
- 地铁站点特征集
时间特征集
节假日特征集
交通故障特征集

### 3、特征提取
> 提取特征数据集中的特征

![特征提取.png](http://upload-images.jianshu.io/upload_images/688387-f021235973c2faf0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 4、标签选择
> 根据每个站点的区域性质不同，划分区域。
- #1 代表远郊住宅区（外环以外）
- #2 代表市区住宅区（外环以内）
- #3 代表公司企业
- #4 代表购物商业
- #5 代表交通枢纽

### 5、模型选择
![模型选择.png](http://upload-images.jianshu.io/upload_images/688387-397ed6ba006e621a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![模型选择2.png](http://upload-images.jianshu.io/upload_images/688387-9f312c9c89241b0d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 6、调参优化
> 解决样本不均衡问题、区域划分标准不统一、网格搜索优化（RF中叶子节点太低会过拟合……）

在 sklearn中，随机森林的函数模型是：	
RandomForestClassifier(bootstrap=True, class_weight=None, criterion=‘gini’, max_depth=None, max_features=‘auto’, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

- A. max_features
增加max_features一般能提高模型的性能，但会降低算法的速度。 因此，需要适当的平衡和选择最佳max_features。
- B. n_estimators
较多的子树可以让模型有更好的性能，但同时让你的代码变慢。 应该选择尽可能高的值，只要处理器能够承受的住，因为这使预测更好更稳定。
- C. min_sample_leaf
叶是决策树的末端节点。 较小的叶子使模型更容易捕捉训练数据中的噪声。应该尽量尝试多种叶子大小种类，以找到最优的那个。
对上面提到的三个参数，进行调优，首先参数A，由于在我们的这个数据中，数据段总共只有七八个，所以我们就简单的选取所有的特征，所以我们只需要对剩下的两个变量进行调优。


- leaf_size: 1 n_estimators_size: 81   precision: 0.824239464438
- leaf_size: 1 n_estimators_size: 91   precision: 0.824334544889
- leaf_size: 1 n_estimators_size: 101 precision: 0.824423142582
- leaf_size: 1 n_estimators_size: 111 precision: 0.824055786294
- leaf_size: 1 n_estimators_size: 121 precision: 0.823984475955
- ……
- model = RandomForestClassifier(min_samples_leaf=1,n_estimators=100)
precision: 82.91%, recall: 82.45% accuracy: 82.45 %

### 7、交叉验证

 交叉验证 | precision | recall | accuracy
---|---|---|---
KFold | 61.23% | 60.63% | 61.16%
StratifiedKFold | 57.59% | 55.41% | 58.27%

### 8、特征工程
> 使用feature_importances_对数据的特征进行排序。

![特征工程.png](http://upload-images.jianshu.io/upload_images/688387-897c02bdc2eed607.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 9、模型融合
![模型融合.png](http://upload-images.jianshu.io/upload_images/688387-151d741ec0e8ae79.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> 模型融合：
- Stacking   cv : 5
- Accuracy: 0.65 (+/- 0.06) [KNN]
- Accuracy: 0.59 (+/- 0.05) [Random Forest]
- Accuracy: 0.60 (+/- 0.09) [GBDT]
- Accuracy: 0.66 (+/- 0.11) [Naive Bayes]
- Accuracy: 0.72 (+/- 0.02) [StackingClassifier]



## 二、深度学习
### 1、Tensorflow  Simple NN
- Input -> hidden -> Output
- batch_size: 100 , num_epochs: 1000
- ('Batch', 300, 'J = ', 5.1246061, 'test accuracy =', 0.87)

### 2、Tensorflow DNNClassifier
- hidden_units=[10, 20, 10], n_classes=5
- nTest Accuracy: 0.946000 n
