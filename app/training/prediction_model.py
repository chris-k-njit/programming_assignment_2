from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # First, start the Spark Session
    spark = SparkSession.builder \
        .appName("WineQualityPrediction") \
        .getOrCreate()

    # Next, load the pre-trained model
    model = LogisticRegressionModel.load("trained_model")

    # Load the validation data
    data = spark.read.csv("data/ValidationDataset.csv", header=True, inferSchema=True, sep=";")

    # Prepare the features (Assume all of the columns except 'quality' are features)
    feature_columns = [col for col in data.columns if col != 'quality']

    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    data_prepared = assembler.transform(data)

    # Run the Predictions
    predictions = model.transform(data_prepared)

    # Evaluate the F1 Score as needed
    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality",
        predictionCol="prediction",
        metricName="f1"
    )

    f1_score = evaluator.evaluate(predictions)

    print(f"\n✅ Prediction Complete — F1 Score: {f1_score:.4f}\n")

    spark.stop()

if __name__ == "__main__":
    main()
