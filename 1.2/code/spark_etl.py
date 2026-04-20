# spark_etl.py

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DataLakeETL") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .getOrCreate()

# Read RAW CSV from Data Lake
df = spark.read.csv(
    "s3a://datalake/raw/patient_raw.csv",
    header=True,
    inferSchema=True
)

# Basic cleaning
df_clean = df.dropna()

# Write structured Parquet (Warehouse layer)
df_clean.write.mode("overwrite").parquet(
    "s3a://curated/patient_visits/"
)

print("ETL complete — Parquet stored in curated layer")

