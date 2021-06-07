from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from pyspark.sql import SparkSession
from pysparkling import H2OContext

from user_definition import *

ss = SparkSession.builder.config('spark.ext.h2o.log.level', 'FATAL')\
                         .getOrCreate()

hc = H2OContext.getOrCreate()

ss.sparkContext.setLogLevel('OFF')

# Step 1

train = ss.read.parquet(train_folder).cache()

valid = ss.read.parquet(valid_folder).cache()

train_h2o = hc.asH2OFrame(train, "train")

train_h2o["label"] = train_h2o["label"].asfactor()

valid_h2o = hc.asH2OFrame(valid, "valid")

valid_h2o["label"] = valid_h2o["label"].asfactor()

for k, v in train_h2o.types.items():
    print(k+" - "+v)

print()

# Step 2

predictors = valid_h2o.names[:]

response = "label"

predictors.remove(response)

for val in predictors:
    print(val)

print()

# Step 3

model_xg = H2OXGBoostEstimator(
    nfolds=n_fold, max_runtime_secs=max_runtime_secs, seed=seed)
model_xg.train(x=predictors,
               y=response,
               training_frame=train_h2o,
               validation_frame=valid_h2o)

print(round(model_xg.auc(valid=True), n_digits))
print()
print(model_xg.confusion_matrix(valid=True))
print()

# Step 4

model_dl = H2ODeepLearningEstimator(nfolds=n_fold,
                                    max_runtime_secs=max_runtime_secs,
                                    seed=seed, variable_importances=True,
                                    loss="Automatic")

model_dl.train(x=predictors,
               y="label",
               training_frame=train_h2o,
               validation_frame=valid_h2o)

print(round(model_dl.auc(valid=True), n_digits))
print()
print(model_dl.confusion_matrix(valid=True))
print()

ss.stop()
