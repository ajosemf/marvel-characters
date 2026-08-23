# Databricks notebook source
# MAGIC %pip install marvel_characters-1.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import time
import os
import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from mlflow import mlflow
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

from marvel_characters.config import ProjectConfig
from marvel_characters.serving.model_serving import ModelServing
from marvel_characters.utils import is_databricks


# COMMAND ----------
# spark session
spark = SparkSession.builder.getOrCreate()

w = WorkspaceClient()

os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Initialize model serving
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.marvel_character_model_custom", endpoint_name="marvel-character-model-serving"
)

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------
# Create a sample request body
required_columns = [
    "Height",
    "Weight",
    "Universe",
    "Identity",
    "Gender",
    "Marital_Status",
    "Teams",
    "Origin",
    "Magic",
    "Mutant"]


# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample records from the training set
sampled_records = test_set[required_columns].sample(n=18000, replace=True)

# Replace NaN values with None (which will be serialized as null in JSON)
import numpy as np
sampled_records = sampled_records.replace({np.nan: None}).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[{'Height': 1.75,
  'Weight': 70.0,
  'Universe': 'Earth-616',
  'Identity': 'Public',
  'Gender': 'Male',
  'Marital_Status': 'Single',
  'Teams': 'Avengers',
  'Origin': 'Human',
  'Magic': 1,
  'Mutant': 1}]
"""

def call_endpoint(record):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/marvel-character-model-serving/invocations"    
    print(f"Calling endpoint: {serving_endpoint}")
    
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Load test
# for i in range(len(dataframe_records)):
#     status_code, response_text = call_endpoint(dataframe_records[i])
#     print(f"Response Status: {status_code}")
#     print(f"Response Text: {response_text}")
#     time.sleep(0.2) 
# COMMAND ----------
# Load test
from datetime import datetime
from pyspark.sql import functions as F


responses = []

for i in range(len(dataframe_records)):
    record = dataframe_records[i]
    request_timestamp = datetime.utcnow()
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    responses.append({
        "record_id": i,
        "status_code": status_code,
        "response_text": response_text,
        "inference_timestamp": request_timestamp
    })
    time.sleep(0.2)

# COMMAND ----------
import json
import pandas as pd

# 1. Achatar e preparar a lista de inputs
# Como cada item em dataframe_records é uma lista de 1 elemento [[{...}], [{...}]], extraímos a posição [0]
flat_inputs = [record[0] for record in dataframe_records]

# 2. Processar a lista de respostas para extrair a predição limpa
processed_responses = []
for resp in responses:
    # Copia o dicionário original da resposta
    resp_data = resp.copy()
    
    # Extrai o valor da predição a partir da string JSON do 'response_text'
    try:
        parsed_json = json.loads(resp['response_text'])
        # Pega a primeira predição (ex: "alive")
        prediction = parsed_json['predictions']['Survival prediction'][0]
    except Exception:
        prediction = None
        
    resp_data['prediction'] = prediction
    processed_responses.append(resp_data)

# 3. Converter para DataFrames do Pandas adicionando a chave de junção (record_id)
df_inputs_pd = pd.DataFrame(flat_inputs)
df_inputs_pd['record_id'] = df_inputs_pd.index

df_responses_pd = pd.DataFrame(processed_responses)

# 4. Fazer o JOIN entre Input e Output pelo record_id
df_inference_pd = pd.merge(df_inputs_pd, df_responses_pd, on='record_id', how='inner')

# 5. Converter para PySpark DataFrame
spark_df = spark.createDataFrame(df_inference_pd)

# Opcional: Reordenar as colunas para ter um layout limpo de Inference Table
columns_order = [
    'record_id', 
    'inference_timestamp', 
    'status_code', 
    'prediction', 
    'response_text'
] + [col for col in df_inputs_pd.columns if col != 'record_id']

spark_df = spark_df.select(columns_order)
# display(spark_df)

# 6. Gravar na tabela do catálogo (Unity Catalog ou Hive Metastore)
target_table = "mlops_dev.marvel_characters.simulated_inference_table"

(spark_df.write
    .format("delta")
    .mode("append") # Use "overwrite" na primeira execução ou para redefinir a tabela
    .option("mergeSchema", "true")
    .saveAsTable(target_table)
)

print(f"Tabela de inferência atualizada com sucesso: {target_table}")
# COMMAND ----------
