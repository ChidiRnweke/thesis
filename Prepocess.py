# Databricks notebook source
# imports
from pyspark.sql.types import DateType
from pyspark.sql.functions import col, concat, sum, to_date
import plotly.express as px
import pyspark.pandas as ps

# COMMAND ----------

sales = table("sales")
display(sales)

# COMMAND ----------

item_group =  sales.groupBy('ItemKey','GroupKey').count().orderBy("count", ascending=False)


# COMMAND ----------

# select first five products
first_5 = item_group.limit(5)
first_5_data = sales.join(first_5, on= (first_5.ItemKey==sales.ItemKey) & (first_5.GroupKey==sales.GroupKey), how ="semi")
display(first_5_data)

# COMMAND ----------

first_5_data = first_5_data\
            .withColumn('groups', concat(first_5_data.ItemKey,first_5_data.GroupKey))
first_5_data = first_5_data.orderBy('LoadDate')

# COMMAND ----------

# Plot for first 5 item x group combination
#first_5_psdf = first_5_data.to_pandas_on_spark()
first_5_psdf = first_5_data.toPandas()
fig = px.line(first_5_psdf, x='LoadDate', y="OrderQtyFirst", color='groups')
fig.show()

# COMMAND ----------

#first_5_pivot = first_5_data.groupBy("LoadDate").pivot("groups").agg(sum("OrderQtyFirst"))
#first_5_pivot = first_5_pivot\
                #.withColumn("LoadDate", to_date("LoadDate"))

# COMMAND ----------

first_5_data = first_5_data.filter("LoadDate >='2015-01-01'")
first_5_data.write.mode("overwrite").saveAsTable("top5")

# COMMAND ----------


