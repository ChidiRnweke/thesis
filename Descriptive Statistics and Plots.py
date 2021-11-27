# Databricks notebook source
# imports
from pyspark.sql.functions import to_date
from datetime import datetime
from pyspark.sql.functions import col, udf, year, month, dayofmonth, date_format, expr
from pyspark.sql.types import DateType
import plotly.express as px

# COMMAND ----------

# load weather and sales tables
weather = spark.table('weather')
sales = spark.table('sales')

# COMMAND ----------

# load item table
item = spark.table('item')
display(item)

# COMMAND ----------

display(sales)

# COMMAND ----------

# convert sales to datetime
func =  udf(lambda x: datetime.strptime(x, '%Y-%m-%d'), DateType())
sales_date = sales.withColumn('LoadDate', func(col('LoadDate')))

# COMMAND ----------

sales_date = sales_date\
             .withColumn('month', month('LoadDate'))\
             .withColumn('year', year('LoadDate'))

# COMMAND ----------

# group by product per month
products_per_month = sales_date.select('ItemKey', 'year','month').distinct().groupBy('year','month').count()
display(products_per_month)

# COMMAND ----------

# Product Count per Year Stacked Plot
df = products_per_month.toPandas()
df = df.sort_values(by="month")
fig = px.bar(df, x='year', y="count", hover_name='month', color='month', title='Count of Products Per Year Stacked Per Month')
fig.update_xaxes(categoryorder='category ascending')
fig.show()

# COMMAND ----------

# Combine month and date and change to datetime
products_per_month_adjusted = products_per_month.withColumn("Date", date_format(expr("make_date(year, month, 1)"), "yyyy-MM-dd"))
products_per_month_adjusted = products_per_month_adjusted.withColumn('date', func(col('Date')))
display(products_per_month_adjusted)

# COMMAND ----------

# Product Count per Month
df_1 = products_per_month_adjusted.toPandas()
fig = px.bar(df_1, x='date', y="count", title='Product Count Per Month')
fig.show()
