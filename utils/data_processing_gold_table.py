
import os
import glob
import pyspark
import pyspark.sql.functions as F

from tqdm import tqdm

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, MapType, NumericType, ArrayType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler, StandardScaler

def read_silver_table(table, silver_db, spark):
    """
    Helper function to read all partitions of a silver table
    """
    folder_path = os.path.join(silver_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    return df

############################
# Label Store
############################
def build_label_store(mob, dpd, silver_db, gold_db, partitions_list, spark, train_ratio=0.8, seed=42):
    """
    Functiion to build label store
    """
    label_table = 'lms' # hard coded

    # Connect to silver table, read all partitions
    df = read_silver_table(label_table, silver_db, spark)

    ####################
    # Create labels
    ####################

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "customer_id", "label", "label_def", "snapshot_date")

    #####################
    # Do train test split
    #####################
    label_counts = df.groupBy("label").count().collect()
    fractions = {row["label"]: train_ratio for row in label_counts}

    df_train = df.stat.sampleBy("label", fractions=fractions, seed=42)
    df_test = df.join(df_train, on=df.columns, how="left_anti")


    ######################
    # save gold table - IRL connect to database to write
    ###################### 

    # Partition and save train data
    for date_str in tqdm(partitions_list, total=len(partitions_list), desc="Saving train labels"):
        partition_name = date_str.replace('-','_') + '.parquet'
        train_filepath = os.path.join(gold_db, 'label_store', 'train', partition_name)
        df_train.filter(col('snapshot_date')==date_str).write.mode('overwrite').parquet(train_filepath)

    
    # Partition and save test data
    for date_str in tqdm(partitions_list, total=len(partitions_list), desc="Saving test labels"):
        partition_name = date_str.replace('-','_') + '.parquet'
        test_filepath = os.path.join(gold_db, 'label_store', 'test', partition_name)
        df_test.filter(col('snapshot_date')==date_str).write.mode('overwrite').parquet(test_filepath)
    
    return df_train, df_test

############################
# Feature Store
############################
def one_hot_encoder(df, category_col):
    """
    Utility function for one hot encoding
    """
    # Get label encoding
    indexer = StringIndexer(inputCol=category_col, outputCol=f"{category_col}_index", handleInvalid="keep")
    indexer_model = indexer.fit(df)
    df = indexer_model.transform(df)

    # Transform into one hot encoding
    encoder = OneHotEncoder(inputCol=f"{category_col}_index", outputCol=f"{category_col}_ohe", dropLast=False)
    df = encoder.fit(df).transform(df)
    vector_to_array_udf = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
    df = df.withColumn(f"{category_col}_array", vector_to_array_udf(f"{category_col}_ohe"))

    # Split into columns
    categories = [cat.lower() for cat in indexer_model.labels]

    for i, cat in enumerate(categories):
        df = df.withColumn(f"{category_col}_{cat}", df[f"{category_col}_array"][i])
        df = df.withColumn(f"{category_col}_{cat}", col(f"{category_col}_{cat}").cast(IntegerType()))

    # Optional: drop intermediate columns
    df = df.drop(category_col, f"{category_col}_index", f"{category_col}_ohe", f"{category_col}_array")
    return df

def build_feature_store(df_attributes, df_financials, df_loan_type, df_clickstream, df_label):
    """
    Function to build feature store based on the subset (train/test)
    """
    #############
    # Join attributes and financials into a single matrix
    #############
    df_joined = df_attributes.join(df_financials, on=["customer_id", "snapshot_date"], how="inner")
    df_joined = df_joined.join(df_loan_type, on=["customer_id", "snapshot_date"], how="inner")
    df_joined = df_joined.drop("name", "ssn", "type_of_loan", "credit_history_age", "type_of_loan") # drop identifiers and duplicated columns
    df_joined = df_joined.join(df_label.select("customer_id"), on="customer_id", how="left_semi") # filter by user IDs that have labels

    print("1. Joined dataframes")

    #############
    # Impute mean into null numeric variables
    #############
    numeric_columns = [column for column in df_joined.columns if isinstance(df_joined.schema[column].dataType, NumericType)]
    imputer = Imputer(inputCols=numeric_columns, outputCols=numeric_columns)
    df_joined = imputer.fit(df_joined).transform(df_joined)

    print("2. Imputed mean into numeric variables")

    #############
    # Turn categorical variables into one hot encoded columns
    #############
    df_joined = one_hot_encoder(df_joined, "occupation")
    df_joined = one_hot_encoder(df_joined, "payment_of_min_amount")
    df_joined = one_hot_encoder(df_joined, "credit_mix")
    df_joined = one_hot_encoder(df_joined, "payment_behaviour_spent")
    df_joined = one_hot_encoder(df_joined, "payment_behaviour_value")

    print("3. Performed one-hot encoding")

    #############
    # Aggregate mean clickstream data for each user
    #############

    # Filter clickstream data
    df_label_renamed = df_label.withColumnRenamed("snapshot_date", "mob_date")
    df_label_renamed = df_label_renamed.select("customer_id", "mob_date")  
    df_clickstream_filtered = df_clickstream.join(df_label_renamed, on="customer_id", how="inner") # filter by user IDs that have labels
    df_clickstream_filtered = df_clickstream_filtered.filter(col("snapshot_date") < col("mob_date")) # get clickstream data before the label date

    # Do mean aggregation
    agg_exprs = [F.avg(f'fe_{i}').alias(f"avg_fe_{i}") for i in range(1, 21)]
    df_clickstream_filtered = df_clickstream_filtered.groupBy("customer_id").agg(*agg_exprs)

    print("4. Processed clickstream data")

    #############
    # Join clickstream data with attributes and financials
    #############
    df_joined = df_joined.join(df_clickstream_filtered, on=["customer_id"], how="left")

    print("5. Joined clickstream data with the rest of the features")

    #############
    # Normalize data
    #############
    numeric_columns = numeric_columns + [f"avg_fe_{i}" for i in range(1, 21)]

    # Create vector column
    assembler = VectorAssembler(
        inputCols=numeric_columns,
        outputCol="features"
    )
    df_vector = assembler.transform(df_joined)

    # Scale the vector
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True
    )
    scaler_model = scaler.fit(df_vector)
    df_joined = scaler_model.transform(df_vector)

    # Convert vector column to array
    vector_to_array_udf = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
    df_joined = df_joined.withColumn("scaled_features_array", vector_to_array_udf("scaled_features"))

    # Override the columns
    for i, col_name in enumerate(numeric_columns):
        df_joined = df_joined.withColumn(col_name, col("scaled_features_array")[i])

    df_joined = df_joined.drop("features", "scaled_features", "scaled_features_array")

    print("6. Normalized data")

    return df_joined

############################
# Pipeline
############################

def process_gold_table(silver_db, gold_db, partitions_list, spark):
    """
    Wrapper function to build all gold tables
    """
    # Build label store
    print("Build label store:")
    df_train_label, df_test_label = build_label_store(6, 30, silver_db, gold_db, partitions_list, spark)
 
    # Read silver tables
    df_attributes = read_silver_table('attributes', silver_db, spark)
    df_clickstream = read_silver_table('clickstream', silver_db, spark)
    df_financials = read_silver_table('financials', silver_db, spark)
    df_loan_type = read_silver_table('loan_type', silver_db, spark)

    # Build train features
    print("Build train features:")
    df_train_features = build_feature_store(df_attributes, df_financials, df_loan_type, df_clickstream, df_train_label)

    # Build test features
    print("Build test features:")
    df_test_features = build_feature_store(df_attributes, df_financials, df_loan_type, df_clickstream, df_test_label)

    # Partition and save train features
    for date_str in tqdm(partitions_list, total=len(partitions_list), desc="Saving train features"):
        partition_name = date_str.replace('-','_') + '.parquet'
        train_filepath = os.path.join(gold_db, 'feature_store', 'train', partition_name)
        df_train_features.filter(col('snapshot_date')==date_str).write.mode('overwrite').parquet(train_filepath)

    # Partition and save test data
    for date_str in tqdm(partitions_list, total=len(partitions_list), desc="Saving test features"):
        partition_name = date_str.replace('-','_') + '.parquet'
        test_filepath = os.path.join(gold_db, 'feature_store', 'test', partition_name)
        df_test_features.filter(col('snapshot_date')==date_str).write.mode('overwrite').parquet(test_filepath)

    return df_train_features, df_train_label, df_test_features, df_test_label