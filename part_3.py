from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder


from user_definition import *


ss = SparkSession.builder.getOrCreate()

# Step 1

activity_code_df = ss.read.jdbc(
    url=url, table=table, properties=properties).cache()

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

# Step 2


def eating(activity):

    for ele in eating_strings:
        if ele in activity.lower():
            return 1
    return 0


eating_udf = udf(eating, IntegerType())

activity_code_df.withColumn('eating', eating_udf('activity'))\
                .where("eating == 1")\
                .select('code')\
                .distinct().orderBy('code').show()

# Step 3

wisdm_cache = wisdm.join(activity_code_df,
                         wisdm.activity_code == activity_code_df.code)\
                   .withColumn('eating', eating_udf('activity'))\
                   .repartition(16)\
                   .cache()

wisdm_cache.orderBy('subject_id', 'timestamp', 'device', 'sensor')\
           .drop('activity', 'code').show(n)

# Step 4

wisdm_accel = wisdm_cache.filter("sensor == 'accel'") \
    .withColumnRenamed('x', 'accel_x') \
    .withColumnRenamed('y', 'accel_y') \
    .withColumnRenamed('z', 'accel_z')
wisdm_gyro = wisdm_cache.filter("sensor == 'gyro'") \
    .withColumnRenamed('x', 'gyro_x') \
    .withColumnRenamed('y', 'gyro_y') \
    .withColumnRenamed('z', 'gyro_z')\
    .withColumnRenamed('eating', 'gyro_eating')\
    .withColumnRenamed('activity', 'gyro_activity')\
    .withColumnRenamed('subject_id', 'gyro_subject')\
    .withColumnRenamed('sensor', 'gyro_sensor')

wisdm_df = wisdm_accel.join(wisdm_gyro,
                            ['activity_code', 'device', 'timestamp'], 'inner')\
                      .cache()

wisdm_device = wisdm_df.select('activity', 'activity_code', 'subject_id',
                               'timestamp', 'device', 'eating',
                               'accel_x', 'accel_y',
                               'accel_z', 'gyro_x', 'gyro_y', 'gyro_z')\
                       .distinct()

print(wisdm_device.count())

print()

# Step 5

wisdm_device_5 = wisdm_device

windowSpec = Window.partitionBy('subject_id', 'activity_code', 'device')\
    .orderBy('timestamp')

for i in range(1, window_size+1):
    wisdm_device_5 = wisdm_device_5.withColumn(
        f"lead_{i}_accel_x", lead('accel_x', i).over(windowSpec))
    wisdm_device_5 = wisdm_device_5.withColumn(
        f"lead_{i}_accel_y", lead('accel_y', i).over(windowSpec))
    wisdm_device_5 = wisdm_device_5.withColumn(
        f"lead_{i}_accel_z", lead('accel_z', i).over(windowSpec))
    wisdm_device_5 = wisdm_device_5.withColumn(
        f"lead_{i}_gyro_x", lead('gyro_x', i).over(windowSpec))
    wisdm_device_5 = wisdm_device_5.withColumn(
        f"lead_{i}_gyro_y", lead('gyro_y', i).over(windowSpec))
    wisdm_device_5 = wisdm_device_5.withColumn(
        f"lead_{i}_gyro_z", lead('gyro_z', i).over(windowSpec))

wisdm_device_5.drop('activity', 'activity_code')\
              .orderBy("subject_id", "activity_code", "device", "timestamp")\
              .show(n)

# Step 6


def indexStringColumns(df, cols):
    # variable newdf will be updated several times
    newdf = df
    for c in cols:
        si = StringIndexer(inputCol=c, outputCol=c+"-num")
        sm = si.fit(newdf)
        newdf = sm.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-num", c)
    return newdf


dfnumeric = indexStringColumns(wisdm_device_5, ["device"])


def oneHotEncodeColumns(df, cols):
    newdf = df
    for c in cols:
        ohe = OneHotEncoder(inputCol=c, outputCol=c+"-onehot", dropLast=False)
        ohe_model = ohe.fit(newdf)
        newdf = ohe_model.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-onehot", c)
    return newdf


wisdm_onehot = oneHotEncodeColumns(dfnumeric, ["device"])

cols = wisdm_onehot.columns

wisdm_onehot.select('subject_id', 'timestamp', 'device', *cols[4:-1])\
            .orderBy("subject_id", "timestamp",
                     "device", ascending=[True, True, True])\
            .drop('activity_code', 'activity', 'eating')\
            .show(n)

# Step 7

input_col = []
for col in wisdm_onehot.schema.names:
    if 'accel' in col:
        input_col.append(col)
    elif 'gyro' in col:
        input_col.append(col)

va = VectorAssembler(outputCol="features",
                     inputCols=input_col, handleInvalid="skip")

wisdm_va = va.transform(wisdm_onehot)


def std_scaler(input_df):
    df = input_df

    scaler = StandardScaler(inputCol="features",
                            outputCol="features_Scaled",
                            withStd=True, withMean=True)

    mm = scaler.fit(df)

    df = mm.transform(df).drop("features")
    df = df.withColumnRenamed("features_Scaled", "features")
    return df


wisdm_scaled = std_scaler(wisdm_va)

wisdm_scaled = wisdm_scaled.select('eating', 'device', 'features')\
                .orderBy('subject_id', 'activity_code', 'device', 'timestamp')

wisdm_scaled.show(n)

# Step 8

new_va = VectorAssembler(outputCol="new_features",
                         inputCols=["features", "device"],
                         handleInvalid="skip")

new_wisdm_va = new_va.transform(wisdm_scaled)

wisdm_label = new_wisdm_va.withColumnRenamed('eating', 'label')\
                     .withColumnRenamed('features', 'old_features')\
                     .withColumnRenamed('new_features', 'features')\
                     .select('features', 'label')

# Step 9

splits = wisdm_label.randomSplit([0.8, 0.2], 1)

train = splits[0].cache()
valid = splits[1].cache()

train.show(n)

valid.show(n)

# Step 10

lr = LogisticRegression(fitIntercept=True)

lrmodel = lr.fit(train)

bceval = BinaryClassificationEvaluator()

cv = CrossValidator().setEstimator(lr).setEvaluator(bceval).setNumFolds(n_fold)

paramGrid = ParamGridBuilder().addGrid(lr.maxIter, max_iter)\
    .addGrid(lr.regParam, reg_params).build()

cv.setEstimatorParamMaps(paramGrid)

cvmodel = cv.fit(train)

print(cvmodel.bestModel.coefficients)
print()

print(cvmodel.bestModel.intercept)
print()

print(cvmodel.bestModel.getMaxIter())
print()

print(cvmodel.bestModel.getRegParam())
print()

validpredicts = cvmodel.bestModel.transform(valid)

# Step 11

print(bceval.evaluate(validpredicts))

ss.stop()
