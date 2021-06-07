from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
# from pyspark.sql.functions import asc, desc

from user_definition import *


ss = SparkSession.builder.getOrCreate()

sc = ss.sparkContext

whole_files = sc.wholeTextFiles(files, minPartitions=16)

# Step 1

num_files = whole_files.count()

print(num_files)

print()

# Step 2


def get_values(val):
    names = val[0].split('/')[-1].split('.txt')[0].split('_')[1:]
    values = val[1].split(',')[1:]
    return names+values


wisdm_rdd = whole_files.map(lambda x: (x[0], x[1].split(';\n')[:-1]))\
    .flatMapValues(lambda x: x)\
    .map(get_values)\
    .map(lambda x: (int(x[0]), x[1], x[2], x[3], int(x[4]),
                    float(x[5]), float(x[6]), float(x[7])))

print(wisdm_rdd.filter(lambda x: x is not None).count())

print()

# Step 3

schema = StructType([StructField("subject_id", IntegerType(), False),
                     StructField("sensor", StringType(), False),
                     StructField("device", StringType(), False),
                     StructField("activity_code", StringType(), False),
                     StructField("timestamp", LongType(), False),
                     StructField("x", FloatType(), False),
                     StructField("y", FloatType(), False),
                     StructField("z", FloatType(), False)
                     ])

wisdm = ss.createDataFrame(wisdm_rdd, schema=schema)

wisdm.printSchema()

# Step 4

wisdm.select("subject_id").distinct().sort("subject_id").show(100)

print()

# Step 5

wisdm.select("sensor").distinct().sort("sensor").show()

print()

# Step 6

wisdm.select("activity_code").distinct().sort("activity_code").show()

# Step 7

wisdm.where("subject_id == " + str(subject_id) +
            " and activity_code == '" + activity_code + "'")\
     .orderBy(["timestamp", "sensor"], ascending=[True, False]).show(n)

# Step 8

wisdm.where("subject_id == " + str(subject_id) +
            " and activity_code == '" + activity_code + "'")\
     .orderBy(["timestamp", "sensor"], ascending=[True, False])\
     .withColumn('x_positive', wisdm['x'] >= 0)\
     .withColumn('y_positive', wisdm['y'] >= 0)\
     .withColumn('z_positive', wisdm['z'] >= 0)\
     .drop("x", "y", "z").show(n)

ss.stop()
