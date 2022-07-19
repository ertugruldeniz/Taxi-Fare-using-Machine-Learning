# Databricks notebook source
# MAGIC %md #Veriyi dosya sisteminden okuma işlemi

# COMMAND ----------

import pandas as pd
import numpy as np
from numpy import array
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
## Pyspark kütüphanelerini projeye dahil etme 
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark import sql
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import *

## Fititng K means Model 
from sklearn.cluster import KMeans

from pyspark.ml.feature import VectorAssembler
### Converting lat long to float values
from pyspark.sql.types import DoubleType

# Importing ML libraries
from pyspark.ml.regression import LinearRegression  
from pyspark.ml.feature import VectorAssembler  
from pyspark.ml.feature import StandardScaler  
from pyspark.ml import Pipeline  
from pyspark.sql.functions import *  

# Dosya tipi ve lokasyonu
file_location = "/FileStore/tables/yellow_tripdata_2016_02*.csv"
file_type = "csv"

# CSV seçenekleri
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)


print("Toplam Veri Sayısı:",df.count())

temp_table_name = "yellow_taxi"
df.createOrReplaceTempView(temp_table_name)


display(df)

# COMMAND ----------

"""
Independent Variables
id — a unique identifier for each trip
vendor_id — a code indicating the provider associated with the trip record
pickup_datetime — date and time when the meter was engaged
dropoff_datetime — date and time when the meter was disengaged
passenger_count — the number of passengers in the vehicle (driver entered value)
pickup_longitude — the longitude where the meter was engaged
pickup_latitude — the latitude where the meter was engaged
dropoff_longitude — the longitude where the meter was disengaged
dropoff_latitude — the latitude where the meter was disengaged
store_and_fwd_flag — This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server — Y=store and forward; N=not a store and forward trip.
 

Target Variable
trip_duration — duration of the trip in seconds
"""


# COMMAND ----------

# MAGIC %md #Veride bulunan aykırı değerleri temizleme

# COMMAND ----------

def clean_and_filter(df):
    ## Converting to appropriate datatypes
    df = df.withColumn('pickup_longitude', df['pickup_longitude'].cast('float'))
    df = df.withColumn('pickup_latitude', df['pickup_latitude'].cast('float'))
    df = df.withColumn('dropoff_longitude', df['dropoff_longitude'].cast('float'))
    df = df.withColumn('dropoff_latitude', df['dropoff_latitude'].cast('float'))
      ## Converting to appropriate datatype
    df = df.withColumn('trip_distance', df['trip_distance'].cast('float'))
    
    #NYC nin koordinatları dışındakileri sil
    df = df.filter("pickup_latitude>=40.53 and pickup_latitude<=40.88")
    df = df.filter("pickup_longitude>=-74.09 and pickup_longitude<=-73.72")
    #31 milden fazla yolculukları sil
    df = df.filter("trip_distance>=0.25 and trip_distance<31")
    
    #6 dan fazla yolcu olamaz
    df = df.filter("passenger_count>0 and passenger_count<7")
    #Ücret en az 2.5 dolar en fazla 500 dolar olabilir
    df = df.filter("total_amount>2.5 and total_amount<500")
      
    #Hız 0 dan düşükse sil
    df = df.filter("Speed_mph>0")

    ## Boş satırları sil
    df = df.na.drop()
    df.show(3)
    return df



# COMMAND ----------

# MAGIC %md #Özellik Oluşturma
# MAGIC 
# MAGIC Pickup_datetime ve dropoff_datetime'ın nesnelerinden yararlanarak istersek,  yeni özellikler oluşturabiliriz.
# MAGIC 
# MAGIC Hız, Yolculuk süresi, yolcunun bırakıldığı saat, ayı veri setimize ekleme işlemleri yapılmıştır.

# COMMAND ----------

## Hızı hesaplama
format = "yyyy-MM-dd HH:mm:ss"
timeDiff = (F.unix_timestamp('tpep_dropoff_datetime', format)
            - F.unix_timestamp('tpep_pickup_datetime', format))/60

df = df.withColumn("tpep_pickup_datetime", from_unixtime(unix_timestamp(df.tpep_pickup_datetime, "yyyy-MM-dd HH:mm:ss")))
df = df.withColumn("pickup_hr",hour(df.tpep_pickup_datetime))
df = df.withColumn("Duration_in_mins", timeDiff)
df = df.withColumn("Speed_mph",df.trip_distance/ ((df.Duration_in_mins)/60))
df = df.withColumn("pickup_month",month(df.tpep_pickup_datetime))

import datetime as dt
#Haftanın Gününü hesaplama
def get_weekday(date):
    import datetime
    import calendar
    date = date.split(' ')[0]
    year,month,day = (int(x) for x in date.split('-'))    
    weekday = datetime.date(year, month, day)
    return calendar.day_name[weekday.weekday()]


weekday_udf = udf(get_weekday)
df = df.withColumn('pickup_day', weekday_udf(df.tpep_pickup_datetime))


df = df.withColumn('pickup_longitude', df['pickup_longitude'].cast('float'))
df = df.withColumn('pickup_latitude', df['pickup_latitude'].cast('float'))
df = df.withColumn('dropoff_longitude', df['dropoff_longitude'].cast('float'))
df = df.withColumn('dropoff_latitude', df['dropoff_latitude'].cast('float'))


#Yeni verilerimi ekledikten sonra aykırı değerleri temizle
df= clean_and_filter(df)
print("Aykırı değerler temizlendikten sonra veri Sayısı:",df.count())

# COMMAND ----------

#df.describe()

# COMMAND ----------

#Birden çok sütunu bir vektör sütununda birleştiren bir özellik dönüştürücü.
vecAssembler = VectorAssembler(inputCols=["dropoff_latitude", "dropoff_longitude"], outputCol="features")
new_df = vecAssembler.transform(df)
display(new_df)
from pyspark.ml.clustering import KMeans

#K-means kümeleme işlemi
kmeans = KMeans(k=15, seed=1) 
model = kmeans.fit(new_df.select('features'))


vecAssembler = VectorAssembler(inputCols=["pickup_latitude", "pickup_longitude"], outputCol="features")
new_df = vecAssembler.transform(df)
df = model.transform(new_df)

## Pickup kümesine tahmin atama işlemi ve column ismini değiştirme
df = df.withColumnRenamed('prediction', 'pickup_cluster')
df = df.drop('features')

vecAssembler = VectorAssembler(inputCols=["dropoff_latitude", "dropoff_longitude"], outputCol="features")
new_df = vecAssembler.transform(df)
df = model.transform(new_df)

df = df.withColumnRenamed('prediction', 'dropoff_cluster')
df = df.drop('features')

# COMMAND ----------

# MAGIC %md #Seaborn kullanarak verimizi görselleştirme

# COMMAND ----------

import seaborn as sns
### Visualizing the clusters
pd_df = df.toPandas()
pd_df = pd_df.sample(frac= 0.1)
sns.set_style("whitegrid")
sns.lmplot(x="pickup_latitude", y="pickup_longitude",data = pd_df[pd_df['pickup_latitude']!=0.0],fit_reg=False,hue='pickup_cluster',size=10,scatter_kws={"s":100})


# COMMAND ----------

sns.catplot(x="pickup_day",y="Duration_in_mins",kind="bar",data=pd_df,height=6,aspect=1)
plt.title('Haftanın günlerine göre yolculuk süreleri')

# COMMAND ----------

#Modelimizdeki hedef değişken ‘Duration_in_mins’ olduğu için onun dağılımına bakıyoruz ve veri setinin geri kalanından ayırıyoruz.
plt.figure(figsize=(4,4))
X = pd_df.drop(['Duration_in_mins'], axis=1)
y = pd_df['Duration_in_mins']
sns.distplot(y)
plt.show()

# COMMAND ----------

# Yolcu_sayısı verileri için veri dağılımını kontrol etme

fig, ax = plt.subplots(figsize = (10,5))
sns.countplot(pd_df.passenger_count, ax = ax)
ax.set_title('Yolcu Sayısı Analizi', size = 16)
ax.set_xlabel('Yolcu Sayısı', size = 12)
ax.set_ylabel('Count', size = 12)
ax.grid(axis='y')
for p in ax.patches:
    ax.annotate('{:.1f}%'.format( (p.get_height() / pd_df.shape[0]) * 100 ), (p.get_x()+0.2, p.get_height()+55))
plt.show()


# COMMAND ----------

figure,(ax9)=plt.subplots(ncols=1,figsize=(10,5))
ax9.set_title('Saat Başına Yolculuk')
ax=sns.countplot(x="pickup_hr",data=pd_df,ax=ax9)

# COMMAND ----------

sns.relplot(y=pd_df.trip_distance,x='Duration_in_mins',data=pd_df)


# COMMAND ----------

df = df.withColumn('VendorID', df['VendorID'].cast('double'))
df = df.withColumn('passenger_count', df['passenger_count'].cast('double'))
df = df.withColumn('trip_distance', df['trip_distance'].cast('double'))
df = df.withColumn('RatecodeID', df['RatecodeID'].cast('double'))
df = df.withColumn('store_and_fwd_flag', df['store_and_fwd_flag'].cast('double'))
df = df.withColumn('payment_type', df['payment_type'].cast('double'))
df = df.withColumn('fare_amount', df['fare_amount'].cast('double'))
df = df.withColumn('extra', df['extra'].cast('double'))
df = df.withColumn('mta_tax', df['mta_tax'].cast('double'))
df = df.withColumn('tip_amount', df['tip_amount'].cast('double'))
df = df.withColumn('tolls_amount', df['tolls_amount'].cast('double'))
df = df.withColumn('improvement_surcharge', df['improvement_surcharge'].cast('double'))
df = df.withColumn('total_amount', df['total_amount'].cast('double'))


#Özelliklerimiz
features = ['passenger_count', 'trip_distance', \
            'RatecodeID', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',\
            'improvement_surcharge', 'pickup_cluster', 'dropoff_cluster', 'pickup_hr', 'Speed_mph']
lr_data = df.select(col("Duration_in_mins").alias("label"), *features) 

## Null değer var mı kontrol et ve sil
for f in features:
    print (f)
    #print (lr_data.where(col(f).isNull()).count())
    
lr_data = lr_data.dropna()

# Train/Test olarak verimizi bölüyoruz.
(training, test) = lr_data.randomSplit([.2, .8])

#Linear regresyon uygula
vectorAssembler = VectorAssembler(inputCols=features, outputCol="unscaled_features")  
# Ölçeklendirme işlemi
standardScaler = StandardScaler(inputCol="unscaled_features", outputCol="features")  
lr = LinearRegression(maxIter=10, regParam=.01)

#Pipeline Uygulama
stages = [vectorAssembler, standardScaler, lr]  
pipeline = Pipeline(stages=stages) 

model = pipeline.fit(training)  
prediction = model.transform(test)

from pyspark.ml.evaluation import RegressionEvaluator  
eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

eval

# COMMAND ----------

lr_data

# COMMAND ----------

# Root Mean Square Error
rmse = eval.evaluate(prediction)  
print("RMSE: %.3f" % rmse)

# Mean Square Error
mse = eval.evaluate(prediction, {eval.metricName: "mse"})  
print("MSE: %.3f" % mse)

# Mean Absolute Error
mae = eval.evaluate(prediction, {eval.metricName: "mae"})  
print("MAE: %.3f" % mae)

# r2 - coefficient of determination
r2 = eval.evaluate(prediction, {eval.metricName: "r2"})  
print("r2: %.3f" %r2)  
