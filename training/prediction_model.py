from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import re

def sanitize_column_names(df):
    renamed_df = df
    for col in df.columns:
        clean_col = re.sub(r'["“”;]', '', col).strip()  # Remove quotes, semicolons, etc.
        renamed_df = renamed_df.withColumnRenamed(col, clean_col)
    return renamed_df

def main():
    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

    data = spark.read.option("header", True) \
                     .option("delimiter", ";") \
                     .option("inferSchema", True) \
                     .csv("data/ValidationDataset.csv")

    # Sanitize column names BEFORE feature selection
    data = sanitize_column_names(data)
    data.printSchema()
    data.show(5)

    features = [col for col in data.columns if col != "quality"]
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    data_prepared = assembler.transform(data)

    model = LogisticRegressionModel.load("training/trained_model")
    predictions = model.transform(data_prepared)

    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    print(f"F1 Score: {f1_score:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()
