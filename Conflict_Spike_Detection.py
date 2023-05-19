# Databricks notebook source
# MAGIC %md
# MAGIC # Next Steps
# MAGIC - Evaluate other period setups (e.g. agging data to weekly or biweekly and calculating outliers there)
# MAGIC - Evaluate admin-specific appraoch (calc outliers for each admin / time period combo individually)
# MAGIC - Create dict of KP IDs for final dataframe so as to easily link outliers to event descriptions (or some other method but playing dictionaries is fun)
# MAGIC - Evaluate other methods including Arima based band construction and rolling window-based clustering methods / also check out z-scores (or filtering data for last two years and clustering) - covid makes it tough as there is undoubtedly a structural shift. See attempts in master branch of git repository.

# COMMAND ----------

!pip install statsmodels
!pip install adtk

# COMMAND ----------

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.core.reshape.merge import merge_asof
import datetime as dt
import matplotlib.pyplot as plt

import adtk
from adtk.detector import PersistAD
from adtk.visualization import plot
from sklearn.ensemble import IsolationForest

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

database_host = dbutils.secrets.get(scope='warehouse_scope', key='database_host')
database_port = dbutils.secrets.get(scope='warehouse_scope', key='database_port')
user = dbutils.secrets.get(scope='warehouse_scope', key='user')
password = dbutils.secrets.get(scope='warehouse_scope', key='password')

database_name = "UNDP_DW_CRD"
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
df = df1.toPandas()

# COMMAND ----------

#### Functions ###
# Convert ACLED Dates to pd
def convert_dt(value):
    valstr = str(value)
    date_clean = dt.datetime(year=int(valstr[0:4]), month=int(valstr[4:6]), day=int(valstr[6:8]))
    return date_clean

df['TimeFK_Event_Date'] = df['TimeFK_Event_Date'].apply(lambda x: convert_dt(x))

# COMMAND ----------

class AnomalyEvent:
    def __init__(self, df, date_col):
        if is_datetime(df[date_col]):
            self.df = df
            self.date_col = date_col
        else:
            raise Exception("The 'date_col' must be a datetime column")
        self.processed_df = None
        self.process_params = None

    
    def process_df(self, target_dict, time_intvl, filter_dict={}, date_dict={}):
        ##### filter
        # filter to subset of data by date
        df = self.df.copy()
        if len(date_dict) != 0:
            df = df.loc[(df[self.date_col] >= date_dict['start_date']) & (df[self.date_col] <= date_dict['end_date']), :]

        # filter to subset of data by column values
        for col, val_lst in filter_dict.items():
            df = df.loc[df[col].isin(val_lst), :]
        
        ##### agg 
        df_col = target_dict['tgt_col']
        sum_count = target_dict['agg_typ']
        if sum_count == 'sum':
            process_df = df.groupby([self.date_col]).agg({df_col:'sum'})
        elif sum_count == 'count':
            process_df = df[[self.date_col, df_col]].groupby([self.date_col]).count()
        else:
            raise Exception("'agg_typ' in 'target_dict' must be 'sum' or 'count'")
        process_df.rename(columns={df_col: 'num'}, inplace=True)
        
        ##### time intervals
        process_df = process_df.resample(time_intvl).sum().fillna(0)
        
        ##### save params
        tm = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        idx = f'{df_col}_{sum_count}_{time_intvl}_{tm}'
        target_dict.update({'time_intvl': time_intvl, 'id': idx})
        target_dict.update(filter_dict)
        target_dict.update(date_dict)
        
        ##### set attribute
        self.processed_df = process_df
        self.process_params = target_dict
    
    def check_zeros(self):
        if self.processed_df is None:
            raise Exception("'process_df' first!")
        else:
            processed_df = self.processed_df.copy()
            #plot density of data 
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
            ax.hist(self.processed_df, density=True, bins=30, alpha=0.5)
            ax.set_title('Density Plot')
            ax.axvline(self.processed['num'].mean(), color='red', linestyle='--')
            ax.text(self.processed_df['num'].mean(), 0.025, f'Mean:{self.processed_df["num"].mean():.2f}', rotation=90)    
            plt.show()
            plt.close()
    
    def get_anomaly(self, anom, anom_dict, graph=True):
        if self.processed_df is None:
            raise Exception("'process_df' first!")
        else:
            processed_df = self.processed_df.copy()
        
        ##### rolling window
        if anom == 'rw':
            persist_ad = PersistAD(**anom_dict)
            processed_df['anomaly'] = persist_ad.fit_detect(processed_df['num'])
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
        processed_df['process_params'] = [self.process_params] * processed_df.shape[0]
        processed_df['model_params'] = [anom_dict] * processed_df.shape[0]
        
        return processed_df
    
    def zero_negbin(self, lag=1, min_obs=30, plot=False)
        if self.processed_df is not None:
        processed_df = self.pprocessed_df.copy()
        if len(processed_df) > min_obs:
            if processed_df['num'].nunique()>1:
                #store lag cols and formula 
                expr = """num ~ """
                lag_cols=[] #list of column names 
                for i in range(1, lag+1):
                    colname=f"num_lag{i}"
                    processed_df[colname]=processed_df["num"].shift(i)
                    lag_cols.append(colname)
                lag_col_expr = "+".join(lag_cols)
                expr += lag_col_expr 

                y,X = dmatrices(expr, processed_df, return_type='dataframe')

                #catch-all try statement 
                try:
                    zinb_res= ZINB(y,X).fit(maxiter=500)
                    #test pred on test and calc RMSE
                    print(zinb_res.summary())
                
                return processed_df 

# COMMAND ----------

# instantiate
ae = AnomalyEvent(df, 'TimeFK_Event_Date')
# process
ae.process_df({'tgt_col':'ACLED_PK', 'agg_typ':'count'}, 'W', filter_dict={'ACLED_Event_Type':['Protests']}, date_dict={'start_date':dt.datetime(2021,1,1), 'end_date':dt.datetime(2023,1,31)})

# COMMAND ----------

# iso
iso = ae.get_anomaly('iso', {'contamination':.5})
iso

# COMMAND ----------

# rolling window
rw = ae.get_anomaly('rw', {'window': 30, 'c': 1.5, 'side':'positive'})
rw

# COMMAND ----------


