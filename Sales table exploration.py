# Databricks notebook source
sales_df = spark.table('sales')
display(sales_df)

# COMMAND ----------

sales_df.select('orderQtyFirst').summary("count", "min", "25%", "50%", "mean", "75%","max","stddev").show()

# COMMAND ----------

display(sales_df.orderBy('OrderQtyFirst', ascending= False))

# COMMAND ----------

item_df = spark.table('item')
display(item_df)

# COMMAND ----------

sales_item_df = sales_df.join(item_df, on='ItemKey', how = 'left')


# COMMAND ----------

sales_item_df = sales_item_df.select('ItemKey','LoadDate','GroupKey','AgencyKey','OrderQtyFirst','ItemDesc')
display(sales_item_df.orderBy('OrderQtyFirst', ascending= False))

# COMMAND ----------

customer_df = spark.table('customer')
customer_df.filter('AgencyKey==3024').show()

# COMMAND ----------

#How much data did we filter out
#STD is high but we are measuring accross drifferent products that each have different order sizes
sales_item_no_outliers = sales_item_df.filter('OrderQtyFirst<=50000')
sales_item_no_outliers.select('orderQtyFirst').summary("count", "min", "25%", "50%", "mean", "75%","max","stddev").show()

# COMMAND ----------

sales_item_no_outliers.count()

# COMMAND ----------

sales_item_df.count()
