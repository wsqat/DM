# coding=gbk

from pyspark import SparkConf, SparkContext

# def main(args: Array[String]): Unit = {
#   System.setProperty("hadoop.home.dir","D:\\hadoop\\hadoop-2.5.2");
#   System.setProperty("spark.sql.warehouse.dir","F:\\spark培训\\spark-2.0.0-bin-hadoop2.6");
#   Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
#   Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
#
#   val conf = new SparkConf().setAppName("NaiveBayesExample").setMaster("local[2]")
#   val sc = new SparkContext(conf)

# local 时URL，本地计算机(从族的概念)
# My App 应用程序名字
from pyspark.sql import SQLContext
conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)
# 此时的sc就是一个SparkContext,了SparkContext的实例化对象，即刻就可以创建RDD了。
sqlContext = SQLContext(sc)
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('/Users/shiqingwang/PycharmProjects/dm/data/20150401_train.csv')
# Displays the content of the DataFrame to stdout
df.show()


from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Species", outputCol="labelindex")
indexed = indexer.fit(df).transform(df)
indexed.show()

from pyspark.sql import Row
from pyspark.mllib.linalg import Vectors
from pyspark.ml.classification import NaiveBayes

# Load and parse the data
def parseRow(row):
    return Row(label=row["labelindex"],
               features=Vectors.dense([row["Sepal.Length"],
                   row["Sepal.Width"],
                   row["Petal.Length"],
                   row["Petal.Width"]]))

## Must convert to dataframe after mapping
parsedData = indexed.map(parseRow).toDF()
labeled = StringIndexer(inputCol="label", outputCol="labelpoint")
data = labeled.fit(parsedData).transform(parsedData)
data.show()

## 训练模型
#Naive Bayes
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model_NB = nb.fit(data)
predict_data= model_NB.transform(data)
traing_err = predict_data.filter(predict_data['label'] != predict_data['prediction']).count()
total = predict_data.count()
nb_scores = float(traing_err)/total
print traing_err, total, nb_scores
#7 150 0.0466666666667


#Logistic Regression###########################################################
# Logistic regression. Currently, this class only supports binary classification.
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=5, regParam=0.01)
model_lr = lr.fit(data)
predict_data= model_lr.transform(data)
traing_err = predict_data.filter(predict_data['label'] != predict_data['prediction']).count()
total = predict_data.count()
lr_scores  = float(traing_err)/total
print traing_err, total, float(traing_err)/total


#Decision Tree
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(maxDepth=2,labelCol = 'labelpoint')
model_DT= dt.fit(data)
predict_data= model_DT.transform(data)
traing_err = predict_data.filter(predict_data['label'] != predict_data['prediction']).count()
total = predict_data.count()
dt_scores = float(traing_err)/total
print traing_err, total, float(traing_err)/total


#GBT###########################################################
## GBT. Currently, this class only supports binary classification.
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=5, maxDepth=2,labelCol="labelpoint")
model_gbt = gbt.fit(data)
predict_data= model_gbt.transform(data)
traing_err = predict_data.filter(predict_data['label'] != predict_data['prediction']).count()
total = predict_data.count()
dt_scores = float(traing_err)/total
print traing_err, total, float(traing_err)/total


#Random Forest
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="labelpoint", seed=42)
model_rf= rf.fit(data)
predict_data= model_rf.transform(data)
traing_err = predict_data.filter(predict_data['label'] != predict_data['prediction']).count()
total = predict_data.count()
dt_scores = float(traing_err)/total
print traing_err, total, float(traing_err)/total

#MultilayerPerceptronClassifier###########################################################
# Classifier trainer based on the Multilayer Perceptron. Each layer has sigmoid activation function, output layer has softmax.
# Number of inputs has to be equal to the size of feature vectors. Number of outputs has to be equal to the total number of labels.
from pyspark.ml.classification import MultilayerPerceptronClassifier
mlp = MultilayerPerceptronClassifier(maxIter=100, layers=[150, 5, 150], blockSize=1, seed=11)
model_mlp= mlp.fit(parsedData)
predict_data= model_mlp.transform(parsedData)
traing_err = predict_data.filter(predict_data['label'] != predict_data['prediction']).count()
total = predict_data.count()
dt_scores = float(traing_err)/total
print traing_err, total, float(traing_err)/total