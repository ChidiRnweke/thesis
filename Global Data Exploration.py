# Databricks notebook source
# import packages
from pyspark.sql.functions import isnan, when, count, col

# COMMAND ----------

# DBTITLE 1,Sales Table Exploration
# MAGIC %md

# COMMAND ----------

# import and display sales table
sales_df = spark.table('sales')
display(sales_df)

# COMMAND ----------

# number of null values 
sales_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in sales_df.columns]).show()

# COMMAND ----------

# stats for for OrderQtyFirst
sales_df.select('orderQtyFirst').summary("count", "min", "25%", "50%", "mean", "75%","max","stddev").show()

# COMMAND ----------

# stats for CalcValueNet3
sales_df.select('CalcValueNet3').summary("count", "min", "25%", "50%", "mean", "75%","max","stddev").show()

# COMMAND ----------

display(sales_df.orderBy('OrderQtyFirst', ascending= False))

# COMMAND ----------

# DBTITLE 1,Item Table Exploration + merging
# MAGIC %md

# COMMAND ----------

# import and display item table
item_df = spark.table('item')
display(item_df)

# COMMAND ----------

# join items and sales tables
sales_item_df = sales_df.join(item_df, on='ItemKey', how = 'left')


# COMMAND ----------

# select columns of interest
sales_item_df = sales_item_df.select('ItemKey','LoadDate','GroupKey','AgencyKey','OrderQtyFirst','ItemDesc')
display(sales_item_df.orderBy('OrderQtyFirst', ascending= False))

# COMMAND ----------

# number of null values for sales x item table 
sales_item_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in sales_item_df.columns]).show()

# COMMAND ----------

# import and display customer table
customer_df = spark.table('customer')
display(customer_df)

# COMMAND ----------

# Explore customer with Agency Key 3024
customer_df.filter('AgencyKey==3024').show()

# COMMAND ----------

# Filter out outliers and explore new stats
#STD is high but we are measuring accross drifferent products that each have different order sizes
sales_item_no_outliers = sales_item_df.filter('OrderQtyFirst<=50000')
sales_item_no_outliers.select('orderQtyFirst').summary("count", "min", "25%", "50%", "mean", "75%","max","stddev").show()

# COMMAND ----------

#How much data did we filter out? Answer: 280
no_of_outliers = sales_item_df.count() - sales_item_no_outliers.count()

# COMMAND ----------

# DBTITLE 1,Weather Table Exploration
# MAGIC %md

# COMMAND ----------

# import and display weather table
weather_df = spark.table('weather')
display(weather_df)

# COMMAND ----------

# What is the date range? Answer: 12/2015 --> 12/2021
weather_df.describe(["location", "Date_time"]).show()

# COMMAND ----------

# is the location always brussels? Answer: yes
weather_df.groupBy('Location').count().show()

# COMMAND ----------

# number of null values 
weather_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in weather_df.columns]).show()
