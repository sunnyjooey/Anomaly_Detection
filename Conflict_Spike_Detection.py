# Databricks notebook source
# MAGIC %md
# MAGIC # Next Steps
# MAGIC - Evaluate other period setups (e.g. agging data to weekly or biweekly and calculating outliers there)
# MAGIC - Evaluate admin-specific appraoch (calc outliers for each admin / time period combo individually)
# MAGIC - Create dict of KP IDs for final dataframe so as to easily link outliers to event descriptions (or some other method but playing dictionaries is fun)
# MAGIC - Evaluate other methods including Arima based band construction and rolling window-based clustering methods / also check out z-scores (or filtering data for last two years and clustering) - covid makes it tough as there is undoubtedly a structural shift. See attempts in master branch of git repository.

# COMMAND ----------

# do NOT use another version of statsmodel!
!pip install statsmodels==0.12.0
!pip install adtk

# COMMAND ----------

#Basic
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.core.reshape.merge import merge_asof
import datetime as dt
import matplotlib.pyplot as plt

#Packates
import adtk
from adtk.detector import PersistAD
from adtk.visualization import plot
from sklearn.ensemble import IsolationForest

# COMMAND ----------

from keys import keys

# ACLED data
database_host = keys["database_host"]
database_port = keys["database_port"]
database_name = keys["database_name"]
user = keys["user"]
password = keys["password"]

table = "dbo.CRD_ACLED"
url = f"jdbc:sqlserver://{database_host}:{database_port};databaseName={database_name};"

df1 = (spark.read
  .format("com.microsoft.sqlserver.jdbc.spark")
  .option("url", url)
  .option("dbtable", table)
  .option("user", user)
  .option("password", password)
  .load()
)

df1 = df1.filter(df1.CountryFK==201)
df1 = df1.toPandas()

# COMMAND ----------

#### Functions ###
# Convert ACLED Dates to pd
def convert_dt(value):
    valstr = str(value)
    date_clean = dt.datetime(year=int(valstr[0:4]), month=int(valstr[4:6]), day=int(valstr[6:8]))
    return date_clean

df1['TimeFK_Event_Date'] = df1['TimeFK_Event_Date'].apply(lambda x: convert_dt(x))

# COMMAND ----------

class AnomalyEvent:
    def __init__(self, df, date_col):
        if is_datetime(df1[date_col]):
            self.df = df
            self.date_col = date_col
        else:
            raise Exception("The 'date_col' must be a datetime column")
        self.processed_df = None

    
    def process_df(self, sum_count, df_col_dict, date_filter={}, admin_filter={}):
        ##### filter
        # filter to subset of data by admin
        if len(admin_filter) != 0:
            df = self.df
            df = df.loc[df[admin_filter['admin_col']] == admin_filter['col_val'], :]
        else:
            df = self.df

        # filter to subset of data by date
        if len(date_filter) != 0:
            df = df.loc[(df[self.date_col] >= date_filter['start_date']) & (df[self.date_col] <= date_filter['end_date']), :]

        # filter to subset of data by column value
        if 'col_val' in df_col_dict.keys():
            df = df.loc[df[df_col_dict['df_col']] == df_col_dict['col_val'], :]
        
        #### sum or count
        # sum (like fatalities) or count (where each row is an event) 
        if sum_count == 'sum':
            process_df = df.groupby([self.date_col]).agg({df_col_dict['df_col']:'sum'})
        elif sum_count == 'count':
            process_df = df[[self.date_col, df_col_dict['df_col']]].groupby([self.date_col]).count()
        else:
            raise Exception("sum_count must be 'sum' or 'count'")
        process_df.rename(columns={df_col_dict['df_col']: 'num'}, inplace=True)
        
        ##### set attribute
        self.processed_df = process_df
    
    
    def get_anomaly(self, time_intvl, anom, anom_dict, graph=True):
        if self.processed_df is None:
            raise Exception("'process_df' first!")
        
        ##### date intervals
        processed_df = self.processed_df.resample(time_intvl).sum().fillna(0)
        
        ##### rolling window
        if anom == 'rw':
            persist_ad = PersistAD(**anom_dict)
            processed_df['anomaly'] = persist_ad.fit_detect(processed_df)
        elif anom == 'iso':
            IForest = IsolationForest(**anom_dict)
            iso_anom = IForest.fit_predict(np.array(processed_df['num']).reshape(-1,1))
            processed_df['anomaly'] = [x == -1 for x in iso_anom]
        else:
            raise Exception("'anom' must be 'rw' or 'iso'")
        
        if graph:
            plot(processed_df, processed_df['anomaly'], ts_linewidth=1, ts_markersize=3, anomaly_color='red', figsize=(20,10), anomaly_tag="marker", anomaly_markersize=5)
    
        ##### save params
        anom_dict.update({'algos': anom})
        processed_df['params'] = [anom_dict] * processed_df.shape[0]
        
        return processed_df

# COMMAND ----------

ae = AnomalyEvent(df1, 'TimeFK_Event_Date')
ae.process_df('count', df_col_dict={'df_col':'ACLED_Event_Type', 'col_val':'Protests'}, date_filter={'start_date':dt.datetime(2021,1,1), 'end_date':dt.datetime(2023,1,31)})

# COMMAND ----------

iso = ae.get_anomaly('D', 'iso', {'contamination':.5})
iso

# COMMAND ----------

rw = ae.get_anomaly('D', 'rw', {'window': 30, 'c': 1.5, 'side':'positive'})
rw

# COMMAND ----------


