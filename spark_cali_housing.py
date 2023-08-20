from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize SparkSession with custom configurations
spark = SparkSession.builder \
    .appName("CaliforniaHousing") \
    .getOrCreate()

# Load data from AWS S3
data = spark.read.csv("s3://awsbucket1-stevensonch/cali_housing.csv", header=True, inferSchema=True)

# Pre-process the data
clean_data = data.dropna()

# One-hot encode the ocean_proximity column
string_indexer = StringIndexer(inputCol="ocean_proximity", outputCol="ocean_proximity_index")
one_hot_encoder = OneHotEncoder(inputCols=["ocean_proximity_index"], outputCols=["ocean_proximity_vec"])

# Assemble features
vector_assembler = VectorAssembler(
    inputCols=["longitude", "latitude", "median_house_value", "total_rooms", "total_bedrooms", "households", "ocean_proximity_vec"],
    outputCol="features"
)

# Set up the pipeline
pipeline = Pipeline(stages=[string_indexer, one_hot_encoder, vector_assembler])
pipeline_model = pipeline.fit(clean_data)
processed_data = pipeline_model.transform(clean_data).select("features", "median_income")

# Split the data
train_data, test_data = processed_data.randomSplit([0.8, 0.2])

# Train a Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="median_income")
lr_model = lr.fit(train_data)
test_results = lr_model.evaluate(test_data)

# Train a Decision Tree Regression model
dt = DecisionTreeRegressor(featuresCol="features", labelCol="median_income")
dt_model = dt.fit(train_data)
dt_test_results = dt_model.transform(test_data)

# Evaluate the Decision Tree model
dt_evaluator = RegressionEvaluator(labelCol="median_income", predictionCol="prediction", metricName="rmse")
dt_rmse = dt_evaluator.evaluate(dt_test_results)
dt_r2 = dt_evaluator.evaluate(dt_test_results, {dt_evaluator.metricName: "r2"})

# Train a Random Forest Regression model
rf = RandomForestRegressor(featuresCol="features", labelCol="median_income")
rf_model = rf.fit(train_data)
rf_test_results = rf_model.transform(test_data)

# Evaluate the Random Forest model
rf_evaluator = RegressionEvaluator(labelCol="median_income", predictionCol="prediction", metricName="rmse")
rf_rmse = rf_evaluator.evaluate(rf_test_results)
rf_r2 = rf_evaluator.evaluate(rf_test_results, {rf_evaluator.metricName: "r2"})

# Print metrics
print("---------- LINEAR REGRESSION RESULTS ----------")
print("Root Mean Squared Error (RMSE):", test_results.rootMeanSquaredError)
print("R2:", test_results.r2)
print("-----------------------------------------------\n")

print("---------- DECISION TREE RESULTS ----------")
print("Root Mean Squared Error (RMSE):", dt_rmse)
print("R2:", dt_r2)
print("-------------------------------------------\n")

print("---------- RANDOM FOREST RESULTS ----------")
print("Root Mean Squared Error (RMSE):", rf_rmse)
print("R2:", rf_r2)
print("------------------------------------------")

# End the Spark session
spark.stop()


# from pyspark.sql import SparkSession
# from pyspark.ml import Pipeline
# from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
# from pyspark.ml.classification import NaiveBayes
# from pyspark.ml.feature import VectorAssembler, StringIndexer, Bucketizer
# from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator

# # Initialize SparkSession
# spark = SparkSession.builder \
#     .appName("CaliforniaHousing") \
#     .getOrCreate()

# # Load data from CSV
# data = spark.read.csv("s3://awsbucket1-stevensonch/cali_housing.csv", header=True, inferSchema=True)

# data.printSchema()
# data.show()

# # Data Preprocessing
# # Handle missing values
# data = data.na.fill({'total_bedrooms': data.agg({'total_bedrooms': 'mean'}).collect()[0][0]})

# # Encoding categorical column
# indexer = StringIndexer(inputCol="ocean_proximity", outputCol="oceanProximityIndex")
# data = indexer.fit(data).transform(data)

# # Feature engineering
# data = data.withColumn("rooms_per_household", data["total_rooms"] / data["households"]) \
#            .withColumn("population_per_household", data["population"] / data["households"])

# # VectorAssembler
# feature_columns = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
#                    "population", "households", "median_income", "ocean_proximity_index",
#                    "rooms_per_household", "population_per_household"]
# vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
# final_data = vector_assembler.transform(data)

# # Split data
# train_data, test_data = final_data.randomSplit([0.8, 0.2])

# # Models
# # Linear Regression
# lr = LinearRegression(featuresCol="features", labelCol="median_house_value")
# lr_model = lr.fit(train_data)
# lr_predictions = lr_model.transform(test_data)

# # Decision Tree
# dt = DecisionTreeRegressor(featuresCol="features", labelCol="median_house_value")
# dt_model = dt.fit(train_data)
# dt_predictions = dt_model.transform(test_data)

# # Binning data for Naive Bayes
# splits = [-float("inf"), 100000, 200000, 300000, float("inf")]
# bucketizer = Bucketizer(splits=splits, inputCol="median_house_value", outputCol="labels")
# train_binned = bucketizer.transform(train_data)
# test_binned = bucketizer.transform(test_data)

# nb = NaiveBayes(featuresCol="features", labelCol="labels", modelType="multinomial")
# nb_model = nb.fit(train_binned)
# nb_predictions = nb_model.transform(test_binned)

# # Evaluation
# reg_evaluator = RegressionEvaluator(labelCol="median_house_value")
# print("Linear Regression RMSE:", reg_evaluator.evaluate(lr_predictions))
# print("Decision Tree RMSE:", reg_evaluator.evaluate(dt_predictions))

# class_evaluator = MulticlassClassificationEvaluator(labelCol="labels", metricName="accuracy")
# print("Naive Bayes Accuracy:", class_evaluator.evaluate(nb_predictions))

# # Stop Spark session
# spark.stop()
