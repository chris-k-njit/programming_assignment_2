from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel

import re
import subprocess
import os

def clean_csv_header(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found at: {input_path}")
    
    print(f"Cleaning CSV header from {input_path} -> {output_path}")
    command = f"""awk 'NR==1{{gsub(/"/, ""); print; next}}1' "{input_path}" > "{output_path}" """
    subprocess.run(command, shell=True, check=True)

def load_data(spark, path):
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("delimiter", ";")
        .option("quote", '"')
        .option("escape", '"')
        .csv(path)
    )
    return df


def sanitize_column_names(df):
    """
    Sanitizes column names that are incorrectly parsed as a single string.
    """
    if len(df.columns) == 1:
        # Looks like entire header is one column; attempt to split
        split_columns = df.columns[0].split(";")
        for i, col in enumerate(split_columns):
            df = df.withColumnRenamed(df.columns[i], col.strip().replace('"', ''))
    else:
        import re
        renamed_df = df
        for col in df.columns:
            clean_col = re.sub(r'["“”;]', '', col).strip()
            renamed_df = renamed_df.withColumnRenamed(col, clean_col)
        df = renamed_df
    return df


def preprocess_data(df):
    """
    Converts features into a single vector and keeps the label column.
    """
    features = [col for col in df.columns if col != 'quality']
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    return assembler.transform(df).select("features", "quality")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

    # Clean header of CSV file before Spark reads it
    clean_csv_header("data/ValidationDataset.csv", "data/ValidationDataset_cleaned.csv")

    # Load cleaned data
    val_df = load_data(spark, "data/ValidationDataset_cleaned.csv")
    val_data = preprocess_data(val_df)

    # Load model and evaluate
    model = LogisticRegressionModel.load("training/trained_model")
    predictions = model.transform(val_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(predictions)
    print(f"F1 Score: {f1:.4f}")

    spark.stop()
