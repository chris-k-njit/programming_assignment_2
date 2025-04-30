from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

def load_data(spark, path):
    df = spark.read.option("header", True) \
                   .option("inferSchema", True) \
                   .option("delimiter", ";") \
                   .option("quote", '"') \
                   .option("escape", '"') \
                   .csv(path)
    df = sanitize_column_names(df)
    df.printSchema()
    df.show(5)
    return df

def sanitize_column_names(df):
    # Remove all double quotes and strip surrounding whitespace
    for col in df.columns:
        clean_col = col.replace('"', '').strip()
        if clean_col != col:
            df = df.withColumnRenamed(col, clean_col)
    return df

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
