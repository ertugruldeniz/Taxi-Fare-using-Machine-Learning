# Databricks notebook source
# MAGIC %md
# MAGIC ###Verisetini Okuma işlemi

# COMMAND ----------

import pandas as pd
import numpy as np
from numpy import array
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
## Setting up Pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark import sql
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler

# File location and type
file_location = "/FileStore/tables/yellow_tripdata_2016_02.csv"
file_type = "csv"

# CSV options
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

# MAGIC %md ###4- Yüklenen veri kümesinin şemasını yazdırılması

# COMMAND ----------

#df.printSchema()


NUMERICAL_FEATURES = ["pickup_longitude", 
                      "pickup_latitude",
                      "dropoff_longitude",
                      "dropoff_latitude",
                      "trip_distance",
                      "passenger_count"] 
                      
TARGET_VARIABLE = "fare_amount"

print("{:d} Numerical features = [{:s}]".format(len(NUMERICAL_FEATURES), ", ".join(["`{:s}`".format(nf) for nf in NUMERICAL_FEATURES])))
print("1 Target variable = `{:s}`".format(TARGET_VARIABLE))


# MAGIC %md ### 5- Veri kümesinin ilk 5 satırını görüntülenmesi

# COMMAND ----------

display(df)
    

# COMMAND ----------

# MAGIC %md ###6- Boş(Null) değerlerin kontrol edilmesi

# COMMAND ----------

#for c in df.columns:
   #print("`{:s}` satırındaki null değer sayısı = {:d}".format(c, df.where(col(c).isNull()).count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Özelliklerimiz
# MAGIC Sütunların veri türleri, flooat değişken türü olarak değiştirilmelidir.
# MAGIC 
# MAGIC Yolculuk Bilgileri
# MAGIC 
# MAGIC pickup_longitude : sayacın devreye girdiği tarih ve saat
# MAGIC 
# MAGIC pickup_latitude : sayacın devre dışı bırakıldığı tarih ve saat
# MAGIC 
# MAGIC dropoff_longitude : Taksimetrenin devre dışı bırakıldığı boylam
# MAGIC 
# MAGIC dropoff_latitude : Taksimetrenin devre dışı bırakıldığı enlem
# MAGIC 
# MAGIC trip_duration : (hedef) saniye cinsinden yolculuk süresi
# MAGIC 
# MAGIC trip_distance: Taksimetre tarafından rapor edilen mil cinsinden geçen yolculuk mesafesi.
# MAGIC 
# MAGIC passenger_count: Yolcu sayısı
# MAGIC 
# MAGIC fare_amount= Toplam ödenen ücret

# COMMAND ----------

# MAGIC %md
# MAGIC ###Data Cleaning-Aykırı Değerlerin Kaldırılması

# COMMAND ----------

df = df.withColumn('pickup_longitude', df['pickup_longitude'].cast('float'))
df = df.withColumn('pickup_latitude', df['pickup_latitude'].cast('float'))
df = df.withColumn('dropoff_longitude', df['dropoff_longitude'].cast('float'))
df = df.withColumn('dropoff_latitude', df['dropoff_latitude'].cast('float'))
df = df.withColumn('trip_distance', df['trip_distance'].cast('float'))
df = df.withColumn('passenger_count', df['passenger_count'].cast('float'))
df = df.withColumn('total_amount', df['total_amount'].cast('float'))

#1 dakikadan az yolculukları sil
#trip_distancce Taksimetre tarafından rapor edilen mil cinsinden geçen yolculuk mesafesi.
#Yolcu sayısı 1 ve 6 dan büyükleri temizle
# Dropping boş satırları sil
df = df.na.drop()

df = df.filter("pickup_latitude>=40.53 and pickup_latitude<=40.88")
df = df.filter("trip_distance>=0.25 and trip_distance<31")
df = df.filter("pickup_longitude>=-74.09 and pickup_longitude<=-73.72")
df = df.filter("passenger_count>0 and passenger_count<7")
df = df.filter("total_amount>2.5 and total_amount<500")

#display(df)
print("Data Cleaning sonrası Toplam Veri Sayısı:",df.count())

# COMMAND ----------

# COMMAND ----------

df.describe().toPandas()  #transpose işlemi yaparak tabloyu daha rahat okuyabiliyoruz.

# COMMAND ----------

# Grafik kütüphanelerini kullanabilmek için önce PySpark DataFrame'imizi Pandas DataFrame'e dönüştürmemiz gerekiyor
pdf = df.toPandas()

#pdf['trip_duration'] = (pdf['tpep_dropoff_datetime'] - pdf['tpep_pickup_datetime']).dt.seconds

#pdf.info()

# COMMAND ----------

#Veri kümesindeki sayısal özelliklerle ilgili temel istatistikler

#pdf.describe()

# COMMAND ----------

#Kullanıllmayacak Sutunları temizle
pdf=pdf.drop(["improvement_surcharge"], axis='columns')
pdf=pdf.drop(["VendorID"], axis='columns')
pdf=pdf.drop(["RatecodeID"], axis='columns')
pdf=pdf.drop(["store_and_fwd_flag"], axis='columns')
pdf=pdf.drop(["extra"], axis='columns')
pdf=pdf.drop(["mta_tax"], axis='columns')
#pdf=pdf.drop(["tip_amount"], axis='columns')
pdf=pdf.drop(["tolls_amount"], axis='columns')
pdf=pdf.drop(["tpep_pickup_datetime"], axis='columns')
pdf=pdf.drop(["tpep_dropoff_datetime"], axis='columns')
pdf=pdf.drop(["payment_type"], axis='columns')
pdf=pdf.drop(["total_amount"], axis='columns')
pdf


# COMMAND ----------

# MAGIC %md
# MAGIC ###Veri Görselleştirmesi

# COMMAND ----------

#Modelimizdeki hedef değişken ‘fare_amount’ olduğu için onun dağılımına bakıyoruz ve veri setinin geri kalanından ayırıyoruz.
plt.figure(figsize=(12,8))
X = pdf.drop(['fare_amount'], axis=1)
y = pdf['fare_amount']
sns.distplot(y)
plt.show()

# COMMAND ----------

# COMMAND ----------

# MAGIC %md ##8- Veri Dağılımlarının Analizi: Sayısal Özellikler

# COMMAND ----------

# MAGIC %md ###Sayısal özelliklerin dağılımları

# COMMAND ----------

# Her sütunun değerlerinin yoğunluk dağılımının çizilmesi

n_rows = 6
n_cols = 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7,20))

for i,f in enumerate(NUMERICAL_FEATURES):
    _ = sns.distplot(pdf[f],
                    kde_kws={"color": "#ca0020", "lw": 1}, 
                    hist_kws={"histtype": "bar", "edgecolor": "k", "linewidth": 1,"alpha": 0.8, "color": "#92c5de"},
                    ax=axes[i]
                    )

fig.tight_layout(pad=1.5)

# COMMAND ----------


#rename columns
#rename specific column names
#pdf.rename(columns = {'pickup_longitude':'pickup_longitude', 'pickup_latitude':'pickup_latitude'}, inplace = True)
pdf

# COMMAND ----------

# Yolcu_sayısı verileri için veri dağılımını kontrol etme

fig, ax = plt.subplots(figsize = (10,5))
sns.countplot(pdf.passenger_count, ax = ax)
ax.set_title('Yolcu Sayısı Analizi', size = 16)
ax.set_xlabel('Yolcu Sayısı', size = 12)
ax.set_ylabel('Count', size = 12)
ax.grid(axis='y')
for p in ax.patches:
    ax.annotate('{:.1f}%'.format( (p.get_height() / pdf.shape[0]) * 100 ), (p.get_x()+0.2, p.get_height()+55))
plt.show()


# COMMAND ----------

#Rezervasyonun yaklaşık %70'i tek yolcu için yapılmaktadır. Beş ve altı numaralı yolcular, büyük koltuk kapasiteli kabinleri rezerve etmiş olabilir, bu nedenle yüksek ücretli olduğu açıktır. Bunu mesafeyi hesapladıktan sonra kanıtlayabiliriz çünkü bu bizim hipotez test durumlarımızdan biridir.

# COMMAND ----------

# Finding average fare amount with respect to passenger count

pdf.groupby('passenger_count')['fare_amount'].mean().plot(kind='bar')
plt.title("Ortalama Ücret Tutarı VS Yolcu Sayısı")
plt.xlabel("Yolcu Sayısı")
plt.ylabel("Ortalama Ücret")
plt.show()

# COMMAND ----------

#Varsayım
#Hipotez: Yolcu sayısına göre ücret artacaktır.
#Ancak yolcu sayısı 6 nın ortalama ücret miktarı diğerlerine göre yüksektir. Bu oldukça belirgindir, ancak diğer yandan ortalama ücret tutarı, yolcu sayısı 3,4 ve 5 için #karşılaştırmalı olarak o kadar yüksek olmayacaktır.

# COMMAND ----------

import seaborn as sb

f,ax = plt.subplots(figsize=(15, 15))
sb.heatmap(pdf.corr(), annot=True, linewidths=.9, fmt= '.2f',ax=ax)
plt.show()

# COMMAND ----------

plt.figure(figsize=(10,10))
plt.scatter(pdf["trip_distance"],pdf["fare_amount"])
plt.xlabel("Yolculuk Mesafesi")
plt.ylabel("Yolculuk Ücreti")

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Doğrusal Regresyon -Linear Regression
# MAGIC Doğrusal Regresyon ile tahmin modeli oluşturmak için scikitlearn kütüphanesinden train_test_split ile tüm verinin %20'si test olacak şekilde rastgele olarak veri dağıtma işlemi yapıldı.
# MAGIC Şimdi Linear Regression modeli seçilir ve bu yöntemle makine öğrenmesi gerçekleştirilir.

# COMMAND ----------

# MAGIC %md
# MAGIC Nitelikler Arasındaki Korelasyon
# MAGIC Aşağıdaki gösterildiği gibi, şaşırtıcı olmayan bir şekilde, “fare amount,” “tip amount,” “total amount,” “distance,” ve “trip minute” gibi nitelikler arasındaki korelasyon katsayısı, bu nitelikler arasında pozitif bir ilişki olduğunu göstermektedir.

# COMMAND ----------

#Veri setimizde 7 bağımsız özellik ve 1 Bağımlı özellik olmak üzere 8 özelliğimiz var.
#Bağımsız değişkenlerimiz
#x=pdf[["trip_distance","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","tip_amount","passenger_count"]]
x=pdf.drop(["fare_amount"], axis='columns')
#x
#Bağımlı değişkenimiz/ prediction
y=pdf[["fare_amount"]]
y

# COMMAND ----------


#Model Building
from sklearn.model_selection import train_test_split

#Artık temiz bir veri setimiz olduğuna göre, taksi ücretini tahmin etmek için bir model eğitmeye hazırız. Bunu yapmadan önce, veri setini bir trene (%80) ve teste (%20) böleceğiz.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=42)

#Learning
x_train
y_train
#Controling
x_test
y_test

from sklearn.linear_model import LinearRegression 
lr=LinearRegression()
lr.fit(x_train,y_train)

#prediction
tahmin=lr.predict(x_test)
tahmin

#real data
y_test

#coefficient β0 katsayısını çağırmak için:
print ('Coefficients: \n',lr.intercept_)

#β1 katsayısını çağırmak için:
print ('Coefficients: \n',lr.coef_)

#R2 değeri
df=lr.score(x_test,y_test)
print(df)
#
dff=lr.score(x_train,y_train)
print(dff)

np.mean(tahmin-y_test)

# COMMAND ----------

#Doğrusal Regresyon modeli için RMSE
from sklearn.metrics import mean_squared_error
y_pred=np.round(lr.predict(x_test),2)
lr_mse=mean_squared_error(y_pred, y_test)
lr_rmse=np.sqrt(mean_squared_error(y_pred, y_test))
lr_train_rmse=np.sqrt(mean_squared_error(lr.predict(x_train), y_train))


print("Tahmin ",y_pred)
print("Test MSE for Linear Regression is ",lr_mse)
print("Test RMSE for Linear Regression is ",lr_rmse)
print("Train RMSE for Linear Regression is ",lr_train_rmse)


# COMMAND ----------

g= sns.regplot(pdf["trip_distance"], pdf["fare_amount"],ci=None,scatter_kws={'color':'r','s':20})
g.set_title("Yolculuk Mesafesine Göre Tahmin - Regresyon Model")
g.set_ylabel("Yolculuk Ücreti")
g.set_xlabel("Yolculuk Mesafesi")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Random Forest Regression

# COMMAND ----------

#Random forest
from sklearn.ensemble import RandomForestRegressor
# create regressor object
rf = RandomForestRegressor(n_estimators = 100, random_state = 42,n_jobs=-1)
rf.fit(x_train,y_train)
rf_pred= rf.predict(x_test)
rf_rmse=np.sqrt(mean_squared_error(rf_pred, y_test))
rf_mse=mean_squared_error(rf_pred, y_test)
print("Prediction",rf_pred)
print("Test MSE for Random Forest is",rf_mse)
print("RMSE for Random Forest is ",rf_rmse)

#R2 değeri
df=rf.score(x_test,y_test)
print(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Gradient Boost Regression

# COMMAND ----------

#Gradient Boost Regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
# Create Training and Test Split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.1)
#
# Standardize the dataset
#
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
#
# Hyperparameters for GradientBoostingRegressor
#
gbr_params = {'n_estimators': 1000,
          'max_depth': 3,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}
#
# Create an instance of gradient boosting regressor
#
gbr = GradientBoostingRegressor(**gbr_params)
#
# Fit the model
#
gbr.fit(X_train_std, y_train)
#
# Print Coefficient of determination R^2
#
print("Model Accuracy: %.3f" % gbr.score(X_test_std, y_test))
#
# Create the mean squared error
#
mse = mean_squared_error(y_test, gbr.predict(X_test_std))
rmse=np.sqrt(mean_squared_error(y_test, gbr.predict(X_test_std)))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("Gradient Boost Regression (RMSE) on test set: {:.4f}".format(rmse))

# COMMAND ----------

###Regression Toward the Mean - ortalamaya doğru regresyon

# COMMAND ----------

import numpy as np

class MeanRegressor():
    def fit(self, inputs, targets):
        self.mean = targets.mean()

    def predict(self, inputs):
        return np.full(inputs.shape[0], self.mean)


mean_model = MeanRegressor()

mean_model.fit(x, y)

mean_model.mean

train_preds = mean_model.predict(x)

# COMMAND ----------

val_preds = mean_model.predict(y)
val_preds[0]
