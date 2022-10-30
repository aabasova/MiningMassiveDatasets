from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from itertools import chain
import pandas as pd


spark = SparkSession.builder.appName('Predict Adult Salary').getOrCreate()

schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("workclass", StringType(), True),
    StructField("fnlwgt", IntegerType(), True),
    StructField("education", StringType(), True),
    StructField("education-num", IntegerType(), True),
    StructField("marital-status", StringType(), True),
    StructField("occupation", StringType(), True),
    StructField("relationship", StringType(), True),
    StructField("race", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("capital-gain", IntegerType(), True),
    StructField("capital-loss", IntegerType(), True),
    StructField("hours-per-week", IntegerType(), True),
    StructField("native-country", StringType(), True),
    StructField("salary", StringType(), True)
])

path = "/Users/angelinabasova/Library/CloudStorage/OneDrive-bwedu/MSc 3. Semester/MMD/MMD-Problem Sets/Datasets/adult/"
train_df = spark.read.csv(path+'train.csv', header=False, schema=schema)
test_df = spark.read.csv(path+'test.csv', header=False, schema=schema)

print(train_df.head(5))

categorical_variables = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
indexers = [StringIndexer(inputCol=column, outputCol=column+"-index") for column in categorical_variables]
encoder = OneHotEncoder(
    inputCols=[indexer.getOutputCol() for indexer in indexers],
    outputCols=["{0}-encoded".format(indexer.getOutputCol()) for indexer in indexers]
)
assembler = VectorAssembler(
    inputCols=encoder.getOutputCols(),
    outputCol="categorical-features"
)
pipeline = Pipeline(stages=indexers + [encoder, assembler])
train_df = pipeline.fit(train_df).transform(train_df)
test_df = pipeline.fit(test_df).transform(test_df)

print(train_df.printSchema())

continuous_variables = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
assembler = VectorAssembler(
    inputCols=['categorical-features', *continuous_variables],
    outputCol='features'
)
train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)


indexer = StringIndexer(inputCol='salary', outputCol='label')
train_df = indexer.fit(train_df).transform(train_df)
test_df = indexer.fit(test_df).transform(test_df)
train_df.limit(10).toPandas()['label']

### Regression
lr = LogisticRegression(featuresCol='features', labelCol='label')
model = lr.fit(train_df)

### Predict
pred = model.transform(test_df)
pred.limit(10).toPandas()[['label', 'prediction']]

#save prediction in csv
pred.toPandas()[['label', 'prediction']].to_csv(path+'pred.csv')

#b) save model to disk
print("b) save model to disk:")
model.write().overwrite().save(path + "lr_model")



#a) print table with feature  names
print("solution a)")

transformed = model.transform(train_df)
attrs = sorted((attr['idx'], attr['name']) for attr in (chain(*transformed.schema['features'].metadata['ml_attr']['attrs'].values())))
gbCvFeatureImportance = pd.DataFrame([(name, transformed.featureImportances[idx]) for idx, name in attrs],columns=['feature_name','feature_importance'])

print(gbCvFeatureImportance.sort_values(by=['feature_importance'],ascending =False))