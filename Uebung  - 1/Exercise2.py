#####
# 2b) #
#####
# Give three own programming examples of your choice for transformations (but not
# for map() or filter()) and three examples for actions (again, of your choice).
# Write executable code and test its correctness (either single program or several
# ones). To generate initial RDDs you can use code from lecture one or from Spark
# documentation. Submit as solution the source code and results of program runs.
# TEST Lea

from pyspark.context import SparkContext
sc = SparkContext('local', 'test')

# ####################################
print("transformations, Join")
# ####################################

# create 4 subject rating pairs
rdd_rating1 = sc.parallelize([('linux', 1), ('C#', 2), ('javascript', 4), ('python', 5)])

# create 2 subject rating pairs
rdd_rating2 = sc.parallelize([('linux', 4), ('java', 2)])

# perform inner join
print(rdd_rating1.join(rdd_rating2).collect())


# #####################################
print("transformations, flatMap")
# #####################################

data = ["Project Gutenberg’s",
        "Alice’s Adventures in Wonderland",
        "Project Gutenberg’s",
        "Adventures in Wonderland",
        "Project Gutenberg’s"]

rdd1 = sc.parallelize(data)

for element in rdd1.collect():
    print(element)


print("###################Flatmap:")
rddFlatmap = rdd1.flatMap(lambda x: x.split(" "))
for element in rddFlatmap.collect():
    print(element)


# #####################################
print("transformations, union:")
# #####################################

dataForUnion1 = ["Hund", "Katze", "Maus"]
dataForUnion2 = ["Schildkröte", "Elefant", "Stachelschwein"]

rddForUnion1 = sc.parallelize(dataForUnion1)

rddForUnion2 = sc.parallelize(dataForUnion2)

rddUnion = rddForUnion1.union(rddForUnion2)

print("###################Union:")
for element in rddUnion.collect():
    print(element)


#data for actions
dataForActions = [10, 12, 11, 100, 90, 67]

rddForActions = sc.parallelize(dataForActions)

# #####################################
print("actions, count:")
# #####################################

#count the entries and print the amount
rddCount = rddForActions.count()
print(rddCount)

# #####################################
print("actions, take(n):")
# #####################################

#get 3 entries
rddTake = rddForActions.take(3)
print(rddTake)

# #####################################
print("actions, top(n):")
# #####################################

#get highest 2 entries
rddTop = rddForActions.top(2)
print(rddTop)