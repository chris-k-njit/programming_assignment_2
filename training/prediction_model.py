from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def sanitize_column_names(df):
    for col in df.columns:
        clean_col = col.replace('"', '').strip()
        if clean_col != col:
            df = df.withColumnRenamed(col, clean_col)
    return df

def main():
    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

    # Load and sanitize validation dataset
    data = spark.read.option("header", True) \
                     .option("delimiter", ";") \
                     .option("quote", '"') \
                     .option("escape", '"') \
                     .option("inferSchema", True) \
                     .csv("app/data/ValidationDataset.csv")

    data = sanitize_column_names(data)
    data.printSchema()

    # Prepare features
    features = [col for col in data.columns if col != 'quality']
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    data_prepared = assembler.transform(data)

    # Load trained model
    model = LogisticRegressionModel.load("training/trained_model")

    # Run prediction and evaluate
    predictions = model.transform(data_prepared)
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)

    print(f"F1 Score: {f1_score:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()
