from pyspark.sql import *
from pyspark.ml import *

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline

from builtins import round

from user_definition import *

ss = SparkSession.builder.getOrCreate()

sc = ss.sparkContext

# Step 1

train = ss.read.parquet(train_folder).cache()

valid = ss.read.parquet(valid_folder).cache()

print(train.count())

print()

print(valid.count())

print()

# Step 2

print("RandomForestClassifier")

rf = RandomForestClassifier()

bceval = BinaryClassificationEvaluator()

cv = CrossValidator().setEstimator(rf).setEvaluator(bceval).setNumFolds(n_fold)

paramGrid = ParamGridBuilder().addGrid(rf.numTrees, num_trees).build()

cv.setEstimatorParamMaps(paramGrid)

cvmodel = cv.fit(train)

print(cvmodel.bestModel.getNumTrees)

validpredicts = cvmodel.bestModel.transform(valid)

print(round(bceval.evaluate(validpredicts), n_digits))

print()

# Step 3

print("GBTClassifier")

gb = GBTClassifier()

bceval = BinaryClassificationEvaluator()

cv = CrossValidator().setEstimator(gb).setEvaluator(bceval).setNumFolds(n_fold)

paramGrid = ParamGridBuilder().addGrid(gb.maxDepth, max_depth).build()

cv.setEstimatorParamMaps(paramGrid)

cvmodel = cv.fit(train)

print(cvmodel.bestModel.getMaxDepth())

validpredicts = cvmodel.bestModel.transform(valid)

print(round(bceval.evaluate(validpredicts), n_digits))

ss.stop()
