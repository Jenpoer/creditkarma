
import os
import pyspark
import pyspark.sql.functions as F

from collections import Counter

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, MapType

############################
# Attributes
############################
def process_df_attributes(df):
    """
    Function to process attributes table
    """
    numeric_regex = r'([-+]?\d*\.?\d+)'
    
    # Extract numeric part from string in 'Age' column
    df = df.withColumn("age", F.regexp_extract(col("age"), numeric_regex, 1))

    # Define column data types
    columns = {
        'customer_id': StringType(),
        'name': StringType(),
        'age': IntegerType(),
        'ssn': StringType(),
        'occupation': StringType(),
        'snapshot_date': DateType()
    }

    # Cast columns to the proper data type
    for column, new_type in columns.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Enforce valid age constraints
    # The oldest person in the world is a little less than 120 years old, so make everything above that invalid
    # Minimum is 0 because some banks allow opening joint accounts for children
    df = df.withColumn(
        "age",
        F.when((col("age") >= 0) & (col("age") <= 120), col("age"))  # keep valid
        .otherwise(None)  # redact invalid
    ) 

    # Enforce valid SSN
    df = df.withColumn(
        "ssn",
        F.regexp_extract(col("ssn"), r'^(\d{3}-\d{2}-\d{4})$', 1)
    )
    df = df.withColumn(
        "ssn",
        F.when(col("ssn") == "", None).otherwise(col("ssn"))
    )

    # Null empty occupation
    df = df.withColumn(
        "occupation",
        F.when(col("occupation") == "_______", None).otherwise(col("occupation"))
    )
    return df

############################
# Clickstream
############################
def process_df_clickstream(df):
    """
    Function to process clickstream table
    """
    # Define column data types
    columns = {
        **{f'fe_{i}': IntegerType() for i in range(1, 21)},
        'customer_id': StringType(),
        'snapshot_date': DateType()
    }

    # Cast columns to the proper data type
    for column, new_type in columns.items():
        df = df.withColumn(column, col(column).cast(new_type))
    return df

############################
# Financials
############################
def split_loan_type(loan_type):
    """
    Utility function to split loan type into frequency table
    """
    if not isinstance(loan_type, str):
        return {}
    
    loans_list = loan_type.replace(' and ', ',').split(',')

    cleaned = [item.strip().replace(' ', '_').lower() for item in loans_list if item.strip() != '']

    return dict(Counter(cleaned))

def process_df_financials(df, silver_db, snapshot_date_str):
    """
    Function to process financials table
    """
    numeric_regex = r'([-+]?\d*\.?\d+)'
    
    columns = {
        'annual_income': FloatType(),
        'monthly_inhand_salary': FloatType(),
        'num_bank_accounts': IntegerType(),
        'num_credit_card': IntegerType(),
        'interest_rate': IntegerType(),
        'num_of_loan': IntegerType(),
        'delay_from_due_date': IntegerType(),
        'num_of_delayed_payment': IntegerType(),
        'changed_credit_limit': FloatType(),
        'num_credit_inquiries': FloatType(),
        'outstanding_debt': FloatType(),
        'credit_utilization_ratio': FloatType(),
        'total_emi_per_month': FloatType(),
        'amount_invested_monthly': FloatType(),
        'monthly_balance': FloatType()
    }

    # Cast columns to the proper data type
    for col_name, dtype in columns.items():
        df = df.withColumn(col_name, F.regexp_extract(col(col_name), numeric_regex, 1))
        df = df.withColumn(col_name, col(col_name).cast(dtype))

    # Split credit history age
    df = df.withColumn("credit_history_age_year",
                        F.regexp_extract(col('credit_history_age'), r'(\d+)\s+Year', 1))
    df = df.withColumn("credit_history_age_year", col("credit_history_age_year").cast(IntegerType()))
    df = df.withColumn("credit_history_age_month",
                        F.regexp_extract(col('credit_history_age'), r'(\d+)\s+Month', 1))
    df = df.withColumn("credit_history_age_month", col("credit_history_age_month").cast(IntegerType()))

    # Remove negative values from columns that should not have it
    for column_name in ['num_of_loan', 'delay_from_due_date', 'num_of_delayed_payment']:
        df = df.withColumn(
            column_name,
            F.when(col(column_name) >= 0, col(column_name))  # keep valid
            .otherwise(None)  # redact invalid
        ) 
    
    # Clip outliers to 90th percentile
    for column_name in ['num_bank_accounts', 'num_credit_card', 'interest_rate', 'num_of_loan', 'num_of_delayed_payment']:
        percentile_value = df.approxQuantile(column_name, [0.97], 0.01)[0]
        df = df.withColumn(
            column_name,
            F.when(col(column_name) > percentile_value, percentile_value)
            .otherwise(col(column_name))
        )

    # Split payment behaviour
    payment_behaviour_regex = r'(Low|High)_spent_(Small|Medium|Large)_value'
    df = df.withColumn(
        'payment_behaviour_spent',
        F.regexp_extract(col('payment_behaviour'), payment_behaviour_regex, 1)
    )
    df = df.withColumn(
        'payment_behaviour_spent',
        F.when(col('payment_behaviour_spent') != '', col('payment_behaviour_spent'))
        .otherwise(None)
    )
    df = df.withColumn(
        'payment_behaviour_value',
        F.regexp_extract(col('payment_behaviour'), payment_behaviour_regex, 2)
    )
    df = df.withColumn(
        'payment_behaviour_value',
        F.when(col('payment_behaviour_value') != '', col('payment_behaviour_value'))
        .otherwise(None)
    )

    # Null empty credit_mix
    df = df.withColumn(
        "credit_mix",
        F.when(col("credit_mix") == "_", None).otherwise(col("credit_mix"))
    )
    
    ######################################
    # Split loan type into its own table
    ######################################
    df_loan_type = df.select('customer_id', 'snapshot_date', 'type_of_loan')

    # Register helper function as a udf
    split_loan_type_udf = F.udf(split_loan_type, MapType(StringType(), IntegerType()))

    # Apply UDF to column
    df_loan_type = df_loan_type.withColumn("loan_type_counts", split_loan_type_udf(col("Type_of_Loan")))
    all_keys = (
        df_loan_type.select("loan_type_counts")
        .rdd.flatMap(lambda row: row["loan_type_counts"].keys() if row["loan_type_counts"] else [])
        .distinct()
        .collect()
    )

    # Create individual columns for each loan type
    for key in all_keys:
        df_loan_type = df_loan_type.withColumn(
            key,
            F.coalesce(col("loan_type_counts").getItem(key), F.lit(0))
        )

    # Drop intermedate columns
    df_loan_type = df_loan_type.drop("loan_type_counts")
    
    # Save new table
    partition_name = snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_db, 'loan_type', partition_name)
    df_loan_type.write.mode("overwrite").parquet(filepath)

    return df.drop('payment_behaviour', 'type_of_loan')

############################
# LMS
############################
def process_df_lms(df):
    """
    Function to process LMS table
    """
    column_type_map = {
        "loan_id": StringType(),
        "customer_id": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    # Cast columns to proper data type
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Add "month on book" column
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # Add "days past due" column
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
    
    return df

############################
# Pipeline
############################
def process_silver_table(table_name, bronze_db, silver_db, snapshot_date_str, spark):
    """
    Wrapper function to build silver table
    """
    # connect to bronze table
    partition_name = snapshot_date_str.replace('-','_') + '.csv'
    filepath = os.path.join(bronze_db, table_name, partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)

    # Change all column names to be lowercase
    df = df.toDF(*[col_name.lower() for col_name in df.columns])

    if table_name == "attributes":
        df = process_df_attributes(df)
    elif table_name == 'clickstream':
        df = process_df_clickstream(df)
    elif table_name == "financials":
        df = process_df_financials(df, silver_db, snapshot_date_str)   
    elif table_name == "lms":
        df = process_df_lms(df)
    else:
        raise ValueError("Table does not exist!")

    # Save silver table
    partition_name = snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_db, table_name, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    return df