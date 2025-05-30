from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

def load_data(spark, path):
    df = spark.read.option("header", True) \
                   .option("delimiter", ";") \
                   .option("inferSchema", True) \
                   .csv("data/ValidationDataset.csv") = path
    df = sanitize_column_names(df)
    df.printSchema()
    df.show(5)
    return df

def sanitize_column_names(df):
    import re
    renamed_df = df
    for col in df.columns:
        # Remove all quotes and leading/trailing whitespace
        clean_col = re.sub(r'["“”]', '', col).strip()
        renamed_df = renamed_df.withColumnRenamed(col, clean_col)
    return renamed_df

def preprocess_data(df):
    features = [col for col in df.columns if col != 'quality']
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    return assembler.transform(df).select("features", "quality")

def train_model(df):
    lr = LogisticRegression(labelCol="quality", featuresCol="features", maxIter=100)
    model = lr.fit(df)
    return model

def evaluate_model(model, df):
    predictions = model.transform(df)
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(predictions)
    print(f"F1 Score: {f1:.4f}")
    return f1

if __name__ == "__main__":
    spark = SparkSession.builder.appName("WineQualityTrainer").getOrCreate()

    train_df = load_data(spark, "data/TrainingDataset.csv")
    val_df = load_data(spark, "data/ValidationDataset.csv")

    train_data = preprocess_data(train_df)
    val_data = preprocess_data(val_df)

    model = train_model(train_data)
    f1 = evaluate_model(model, val_data)

    model.save("training/trained_model")

    spark.stop()
