from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
import pyspark.ml

spark = SparkSession.builder.appName('Predict Adult Salary').getOrCreate()

path = "/Users/angelinabasova/Library/CloudStorage/OneDrive-bwedu/MSc 3. Semester/MMD/MMD-Problem Sets/Datasets/adult/"
model = LogisticRegressionModel.load(path + "lr_model")

print("Model loaded")