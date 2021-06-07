import h2o
from h2o.automl import H2OAutoML

from pyspark.sql import SparkSession
from pysparkling import *

from user_definition import *

ss = SparkSession.builder.config('spark.ext.h2o.log.level', 'FATAL')\
                         .getOrCreate()

ss.sparkContext.setLogLevel('OFF')

hc = H2OContext.getOrCreate()

# Step 1

train = ss.read.parquet(train_folder).cache()

valid = ss.read.parquet(valid_folder).cache()

train_h2o = hc.asH2OFrame(train, "train")

valid_h2o = hc.asH2OFrame(valid, "valid")

train_h2o[response] = train_h2o[response].asfactor()

valid_h2o[response] = valid_h2o[response].asfactor()

h2o.no_progress()

# Step 2

predictors = valid_h2o.names[:]

predictors.remove(response)

# Step 3

model_automl = H2OAutoML(max_runtime_secs=max_runtime_secs,
                         seed=seed, nfolds=n_fold)
model_automl.train(x=predictors,
                   y=response,
                   training_frame=train_h2o,
                   validation_frame=valid_h2o)

print(model_automl.leaderboard)

print()

print(round(model_automl.leader.auc(valid=True), n_digits))

print()

print(model_automl.leader.confusion_matrix(valid=True))

print()

ss.stop()
