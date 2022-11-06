import numpy as np
import pandas
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, IntegerType
from pyspark.sql.functions import when, sum

######################
# 5a)
print("Exercise 5a)")
######################

sparkSession = SparkSession.builder.appName('A2E5').getOrCreate()

print("# --------------------------------")
print("# Create DataFrame DF1 user_artist")
print("# --------------------------------")
rdd_datafile = sparkSession.sparkContext.textFile('./user_artist_data_small.txt')
print(rdd_datafile.first())

rdd_user_artist = rdd_datafile.map(lambda x: x.split(" "))

musicRDD = rdd_user_artist.map(lambda p: (
    int(p[0]), int(p[1].strip()), int(p[2].strip())))

print(musicRDD.first())

schema = StructType([
    StructField("userid", IntegerType(), True),
    StructField("artistid", IntegerType(), True),
    StructField("playcount", IntegerType(), True)
])

df1 = sparkSession.createDataFrame(musicRDD, schema)
df1.printSchema()
df1.show(truncate=False)

print("# ---------------------------------")
print("# Create DataFrame DF2 artist_alias")
print("# ---------------------------------")
rdd_datafile_alias = sparkSession.sparkContext.textFile('./artist_alias_small.txt')
print(rdd_datafile_alias.first())

rdd_datafile_alias = rdd_datafile_alias.map(lambda x: x.split("\t"))

musicRDD_alias = rdd_datafile_alias.map(lambda p: (
    int(p[0]), int(p[1].strip())))

schema_alias = StructType([
    StructField("badid", IntegerType(), True),
    StructField("goodid", IntegerType(), True)
])

df2 = sparkSession.createDataFrame(musicRDD_alias, schema_alias)
df2.printSchema()
df2.show(truncate=False)

print("# ---------------------------------")
print("# Create DataFrame dfjoin")
print("# Join rdd_datafile rdd_datafile_alias")
print("# ---------------------------------")

dfjoin = df1.join(df2, df1.artistid == df2.badid, "leftouter")

dfjoin.show()

print("Ein Guter:")
print(dfjoin.filter("badid is null").first())

print("Ein Schlechter:")
print(dfjoin.filter("badid is not null").first())

print("")
print("# ---------------------------------")
print("# Create DataFrame korrigiert")
print("# ---------------------------------")

# Die nicht gefüllten (NULL, None) Felder mit -1 ersetzen.
targetDf = dfjoin.na.fill(-1)

# dem Dataframe eine neue Column mit der korrigierten artistidKorr hinzufügen
targetDf = targetDf.withColumn("artistidKorr",
                               when(targetDf["goodid"] == -1, targetDf["artistid"]).otherwise(targetDf["goodid"]))

# die Spalten renamen
targetDf = targetDf.withColumnRenamed("userid", "user") \
    .withColumnRenamed("artistidKorr", "artist") \
    .withColumnRenamed("playcount", "listeningAmount")

# Achtung: Duplikatfehler!!
# +-------+--------+---------------+-------+-------+-------+
# |   user|artistid|listeningAmount|  badid| goodid| artist|
# +-------+--------+---------------+-------+-------+-------+
# |1017610| 1249239|             29|     -1|     -1|1249239|
# |1017610| 1017610|              1|1017610|1249239|1249239|
# +-------+--------+---------------+-------+-------+-------+

# deshalb jetzt aggregieren:
# erstmal die Spalten löschen, die wir nicht brauchen:
targetDf = targetDf.drop("badid").drop("goodid").drop("artistid")
# +-------+---------------+-------+
# |   user|listeningAmount| artist|
# +-------+---------------+-------+
# |1017610|             29|1249239|
# |1017610|              1|1249239|
# +-------+---------------+-------+

targetDf.printSchema()
targetDf = targetDf.groupBy("artist", "user") \
    .agg(sum("listeningAmount").alias("listeningAmount"))
# +-------+-------+---------------+
# |   user| artist|listeningAmount|
# +-------+-------+---------------+
# |1017610|1249239|             30|
# +-------+-------+---------------+


print("# ---------------------------------")
print("# Utility-Matrix erstellen")
print("# ---------------------------------")

pandaDataFrameMusic = targetDf.toPandas()

utilityMatrix = pandaDataFrameMusic.set_index(['artist', 'user'])['listeningAmount'].unstack()

print(utilityMatrix)

utilityMatrix.to_csv("./utilityMatrix.csv")

######################
# 5b)
print("Exercise 5b)")
######################

print("# ---------------------------------")
print("# Pearson-Matrix erstellen")
print("# ---------------------------------")

#NaN-Werte = "0"
utilityMatrix = utilityMatrix.replace(np.nan, 0)

#Funktion für Pearson ist dataframe.corr()
pearsonMatrix = utilityMatrix.corr(method='pearson', min_periods=1, numeric_only=float)

print(pearsonMatrix)

pearsonMatrix.to_csv("./Pearson.csv")

######################
# 5c)
print("Exercise 5c)")
######################

def highestSimilarityForUser(eingabe_User, pearsonMatrix, eingabe_K):
    if eingabe_User in pearsonMatrix:
        print("# ---------------------------------")
        print("# Highest Similarity Matrix For User:")
        print("# ---------------------------------")
        pearsonMatrixForUser = pearsonMatrix[eingabe_User]
        highestSimilarityMatrixForUser = pearsonMatrixForUser.sort_values(ascending=False).head(eingabe_K)
        return highestSimilarityMatrixForUser
    else:
        print("UserID not given in PearsonMatrix! Try again")

#UserID Input
eingabe_User = input("UserID: ")

#K-neighborhood Input
eingabe_K = input("K-neighborhood: ")

user = int(eingabe_User)
k = int(eingabe_K)

#Funktionsaufruf mit Inputeingaben highestSimilarityForUser()
highestSimilarityMatrixForUser = highestSimilarityForUser(user, pearsonMatrix, k)
print(highestSimilarityMatrixForUser)