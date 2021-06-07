from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *

from user_definition import *

# Do not add any other libraies/packages

ss = SparkSession.builder.getOrCreate()

activity_code_df = ss.read.jdbc(
    url=url, table=table, properties=properties).cache()

# Step 1

print(activity_code_df.count())

print()

# Step 2

activity_code_df.orderBy('activity', ascending=False).show(truncate=False)

# Step 3


def eating(activity):
    return bool([ele for ele in eating_strings if(ele in activity.lower())])


eating_udf = udf(eating, BooleanType())

activity_eating = activity_code_df.withColumn('eating', eating_udf('activity'))

activity_eating.printSchema()

activity_eating.orderBy('eating', 'code', ascending=[False, True]).show()


# Step 4

sc = ss.sparkContext

whole_files = sc.wholeTextFiles(files, minPartitions=16)


def get_values(val):
    names = val[0].split('/')[-1].split('.txt')[0].split('_')[1:]
    values = val[1].split(',')[1:]
    return names+values


wisdm_rdd = whole_files.map(lambda x: (x[0], x[1].split(';\n')[:-1]))\
    .flatMapValues(lambda x: x)\
    .map(get_values)\
    .map(lambda x: (int(x[0]), x[1], x[2], x[3], int(x[4]),
                    float(x[5]), float(x[6]), float(x[7])))

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

wisdm.select('subject_id', 'sensor', 'device', 'activity_code')\
    .distinct()\
    .groupBy('subject_id', 'sensor', 'device')\
    .agg(count('activity_code').alias('count'))\
    .orderBy('subject_id', 'device', 'sensor')\
    .show(250)

# Step 5

wisdm_cache = wisdm.join(activity_code_df,
                         wisdm.activity_code == activity_code_df.code)\
                   .repartition(16)\
                   .cache()


wisdm_cache.groupBy('subject_id', 'activity',
                    'device', 'sensor')\
           .agg(min('x').alias('x_min'), min('y').alias('y_min'),
                min('z').alias('z_min'),
                avg('x').alias('x_avg'), avg('y').alias(
                'y_avg'), avg('z').alias('z_avg'),
                max('x').alias('x_max'), max('y').alias(
                'y_max'), max('z').alias('z_max'),
                expr('percentile(x, array(0.05))')[0].alias('x_05%'),
                expr('percentile(y, array(0.05))')[0].alias('y_05%'),
                expr('percentile(z, array(0.05))')[0].alias('z_05%'),
                expr('percentile(x, array(0.25))')[0].alias('x_25%'),
                expr('percentile(y, array(0.25))')[0].alias('y_25%'),
                expr('percentile(z, array(0.25))')[0].alias('z_25%'),
                expr('percentile(x, array(0.5))')[0].alias('x_50%'),
                expr('percentile(y, array(0.5))')[0].alias('y_50%'),
                expr('percentile(z, array(0.5))')[0].alias('z_50%'),
                expr('percentile(x, array(0.75))')[0].alias('x_75%'),
                expr('percentile(y, array(0.75))')[0].alias('y_75%'),
                expr('percentile(z, array(0.75))')[0].alias('z_75%'),
                expr('percentile(x, array(0.95))')[0].alias('x_95%'),
                expr('percentile(y, array(0.95))')[0].alias('y_95%'),
                expr('percentile(z, array(0.95))')[0].alias('z_95%'),
                stddev('x').alias("x_std"), stddev("y").alias("y_std"),
                stddev("z").alias("z_std"))\
           .orderBy('activity', 'subject_id', 'device', 'sensor')\
           .show(n)

# Step 6


def check_act(activity):
    if activity_string in activity.lower():
        return True
    else:
        return False


check_act_udf = udf(check_act, BooleanType())


wisdm_cache.select('activity', 'timestamp', 'device', 'sensor',
                   'x', 'y', 'z',
                   check_act_udf('activity').alias('check_act'))\
           .where("subject_id == " + str(subject_id) +
                  " and check_act == true")\
           .orderBy('timestamp', 'device', 'sensor')\
           .drop('check_act')\
           .show(n)

# Step 7

code = activity_code_df.where(check_act_udf('activity'))\
    .collect()[0]['code']

wisdm_accel = wisdm_cache.filter("subject_id == " + str(subject_id) +
                                 " and sensor == 'accel'") \
    .filter("activity_code == '" + code + "'") \
    .withColumnRenamed('x', 'accel_x') \
    .withColumnRenamed('y', 'accel_y') \
    .withColumnRenamed('z', 'accel_z')
wisdm_gyro = wisdm_cache.filter("subject_id == " + str(subject_id) +
                                "  and sensor == 'gyro'") \
    .filter("activity_code == '" + code + "'") \
    .withColumnRenamed('x', 'gyro_x') \
    .withColumnRenamed('y', 'gyro_y') \
    .withColumnRenamed('z', 'gyro_z')

joined = wisdm_accel.join(
    wisdm_gyro, ['activity_code', 'device', 'timestamp'], 'inner')
joined.sort('timestamp', ascending=True).drop(
    'subject_id', 'sensor', 'activity', 'code').show(n)

ss.stop()
