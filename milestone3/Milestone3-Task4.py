from pyspark.ml import Pipeline
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler, UnivariateFeatureSelector
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor as sparkRFR
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd

## Depending on the permissions that you provided to your bucket you might need to provide your aws credentials
## to read from the bucket, if so provide with your credentials and pass as storage_options=aws_credentials
aws_credentials = {
    "key": "ASIAVAPDWVQZAMHS6TPW",
    "secret": "M8CYYGCnrO2bPb6Oe48zrER4NpxyJYqayhdtbxt6",
    "token":"FwoGZXIvYXdzEPn//////////wEaDJnq3tykV1TAQvEvkCLKAV6b90or2eVd4wNgBXiqeHs/YoSkPAo32lN/320eLujAZKf14+Aj4lkXnFmw1JsVjarb6fTq4huOzX4dAOvsBH0qYjEIIZjM6Yu8forpoI9927FulvG0tkZGwI2/ugr1LGXgxuWJp2qvSh+Dxgzk2EkznbRsrA7dImkzRvoIDv5CUtACJiXb/ewL/D9kxKZAlk+uwZPV8R9I/8HEvs96bm4qLQP7IV7b8+Zp8sIznDCz3c4IaWfBC6oPPDjyxkVIZYgagxUPcpv0eWIon6XmkgYyLasupgGhKbbSTgHaJXvxReyP0JZ74f1svDVy/mZviXzOUAaxh124APQF1HFeKw=="
}
## here 100 data points for testing the code

#pandas_df = pd.read_csv(
#    "s3://mds-s3-30/output/wrangled.csv", 
#    parse_dates=True, 
#    storage_options=aws_credentials).iloc[:100].dropna().rename({"rain (mm/day)": "label"}, axis=1)

pandas_df = pd.read_csv(
    "s3://mds-s3-30/output/wrangled.csv", 
    parse_dates=True, 
    storage_options=aws_credentials).dropna().rename({"rain (mm/day)": "label"}, axis=1)

feature_cols = list(pandas_df.drop(columns="label").columns)

# Load dataframe and coerce features into a single column called "Features"
# This is a requirement of MLlib
# Here we are converting your pandas dataframe to a spark dataframe, 
# Here "spark" is a spark session I will discuss this in our Wed class. 
# It is automatically created for you in this notebook.
# read more  here https://blog.knoldus.com/spark-createdataframe-vs-todf/
training = spark.createDataFrame(pandas_df)
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
training = assembler.transform(training).select("features", "label")

##Once you finish testing the model on 100 data points, then load entire dataset and run , this could take ~15 min.
## write code here.

rf = sparkRFR()
grid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [10, 50,100])
    .addGrid(rf.maxDepth, [5, 10])
    .addGrid(rf.bootstrap, [False, True])
    .build())
evaluator = RegressionEvaluator(labelCol="label")
cv = CrossValidator(
    estimator=rf, 
    estimatorParamMaps=grid, 
    evaluator=evaluator)
cvModel = cv.fit(training)

# Print run info
print("\nBest model")
print("==========")
print(f"\nCV Score: {min(cvModel.avgMetrics):.2f}")
print(f"numTrees: {cvModel.bestModel.getNumTrees}")
print(f"maxDepth: {cvModel.bestModel.getMaxDepth()}")


