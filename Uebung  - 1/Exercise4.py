from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, DoubleType
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import min, max
from pyspark.sql.functions import when

######################
# 4a) Import adult dataset into a dataframe by reading-in RDDs first, and then converting RDDs to dataframes
# with suitable column names. If necessary, clean up this
# data so that you can execute the following queries on the dataframe.
print("Exercise 4a)")
######################

sparkSession = SparkSession.builder.appName('A1E4').getOrCreate()

rdd_datafile = sparkSession.sparkContext.textFile('./adult.data')
print(rdd_datafile.first())

rddFlat = rdd_datafile.map(lambda x: x.split(","))

adultRDD = rddFlat.map(lambda p: (
    p[0], p[1].strip(), p[2].strip(), p[3].strip(), int(p[4].strip()), p[5].strip(), p[6].strip(), p[7].strip(),
    p[8].strip(),
    p[9].strip(), p[10].strip(), p[11].strip(), int(p[12].strip()), p[13].strip(), p[14].strip()))

print(adultRDD.first())

schema = StructType([
    StructField("age", StringType(), True),
    StructField("workclass", StringType(), True),
    StructField("fnlwgt", StringType(), True),
    StructField("education", StringType(), True),
    StructField("education_num", IntegerType(), True),
    StructField("marital_status", StringType(), True),
    StructField("occupation", StringType(), True),
    StructField("relationship", StringType(), True),
    StructField("race", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("capital_gain", StringType(), True),
    StructField("capital_loss", StringType(), True),
    StructField("hours_per_week", IntegerType(), True),
    StructField("native_country", StringType(), True),
    StructField("income", StringType(), True)
])

dataframeAdult = sparkSession.createDataFrame(adultRDD, schema)
dataframeAdult.printSchema()
dataframeAdult.show(truncate=False)

######################
# 4b) Compute the ratio of males for each type of marital_status. Please also consider the types of
# marital_status with 0 males.
print("Exercise 4b)")
######################

dataframeAdult.groupBy("marital_status", "sex") \
    .count() \
    .withColumn('total', F.sum('count').over(Window.partitionBy("marital_status"))) \
    .withColumn('percentage', F.col('count') / F.col('total') * 100) \
    .filter(dataframeAdult["sex"] == "Male") \
    .show()

######################
# 4c) Compute the average hours_per_week of females who have income greater than 50K
# for each native_country
print("Exercise 4c)")
######################

dataframeAdult \
    .filter(dataframeAdult["income"] == ">50K") \
    .filter(dataframeAdult["sex"] == "Female") \
    .groupby("native_country") \
    .avg("hours_per_week") \
    .show()

######################
# 4d) Get the highest and lowest level of education for each group of income.
# The highest level of education is the level with highest value of education_num.
# To simplify, you can use Python dictionary to translate from
# education_num to education for displaying results (derive values from data).
# The result should be displayed similarly to the table below:
print("Exercise 4d)")
######################

df2 = dataframeAdult.groupBy("income") \
    .agg(max("education_num").alias("highest_education"), min("education_num").alias("lowest_education"))

df2.show()

df2.withColumn('highest_education',
               when(df2.highest_education == 16, 'Doctorate') \
               # ...
               .otherwise(df2.highest_education)) \
    .withColumn('lowest_education',
                when(df2.lowest_education == 2, '1st-4th') \
                .when(df2.lowest_education == 1, 'Preschool') \
                # ...
                .otherwise(df2.lowest_education)) \
    .show(truncate=False)
