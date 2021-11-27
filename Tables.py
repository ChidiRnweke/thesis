# Databricks notebook source
weather = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load("/mnt/bronze/internship_model_drift/data/weather.csv")
weather.write.saveAsTable('weather')

# COMMAND ----------

weather.createOrReplaceTempView("weather_tb")
weather.write.saveAsTable("weather_tb")

# COMMAND ----------

item = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load("/mnt/bronze/internship_model_drift/data/item.csv")
item.write.saveAsTable("item")

# COMMAND ----------

customer = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load("/mnt/bronze/internship_model_drift/data/customer.csv")
customer.write.saveAsTable('customer')

# COMMAND ----------

promos = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load("/mnt/bronze/internship_model_drift/data/promos.csv")
promos.write.saveAsTable('promos')

# COMMAND ----------

subconcept = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load("/mnt/bronze/internship_model_drift/data/subconcept.csv")
subconcept.write.saveAsTable('subconcept')

# COMMAND ----------

sales = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load("/mnt/bronze/internship_model_drift/data/sales.csv")
sales.write.saveAsTable('sales')
