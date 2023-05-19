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
!pip install openpyxl

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
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP as ZINB
from statsmodels.discrete.count_model import ZeroInflatedPoisson as ZINP
from patsy import dmatrices

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

database_host = dbutils.secrets.get(scope='warehouse_scope', key='database_host') #host
database_port = dbutils.secrets.get(scope='warehouse_scope', key='database_port')
user = dbutils.secrets.get(scope='warehouse_scope', key='user')
password = dbutils.secrets.get(scope='warehouse_scope', key='password')

database_name = "UNDP_DW_CRD"
table = "dbo.CRD_ACLED"
url = f"jdbc:sqlserver://{database_host}:{database_port};databaseName={database_name};"


# COMMAND ----------

df1 = (spark.read
  .format("com.microsoft.sqlserver")
  .option("host", "hostName")
  .option("port", "port") # optional, can use default port 1433 if omitted
  .option("user", "username")
  .option("password", "password")
  .option("database", "databaseName")
  .option("dbtable", "schemaName.tableName") # (if schemaName not provided, default to "dbo")
  .load()
)

# COMMAND ----------

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
def convert_dt(value):
    valstr = str(value)
    date_clean = dt.datetime(year=int(valstr[0:4]), month=int(valstr[4:6]), day=int(valstr[6:8]))
    return date_clean

df['TimeFK_Event_Date'] = df['TimeFK_Event_Date'].apply(lambda x: convert_dt(x))

# COMMAND ----------

#use undss data as acled datawarehouse is down 
df = pd.read_excel('/dbfs/FileStore/df/undss/data/sahel_incident_data.xlsx')
# change date column to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
df1 = df[df['Country'] == 'NIGER']

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
           # ax.axvline(self.processed['num'].mean(), color='red', linestyle='--')
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
    
    def zero_negbin(self, lag=1, min_obs=30):
        if self.processed_df is not None:
            processed_df = self.processed_df.copy()
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
                        pred_values = zinb_res.predict()
                        resid = zinb_res.resid
                        print(resid)
                       # window_size = 10
                        #threshold_factor = 2 
                       # moving_avg = resid.rolling(window=window_size).mean()
                        #threshold = moving_avg + threshold_factor * moving_avg.std()
                        threshold = resid.std()*2 #look at other thresholds 

                        processed_df.reset_index(drop=True, inplace=True)
                        resid.reset_index(drop=True, inplace=True)


                        #subset
                        anomaly_df = pd.DataFrame({'Anomaly': (resid.abs() > threshold).astype(int)})
                        print(anomaly_df)

                        processed_df = pd.concat([processed_df, anomaly_df], axis=1)
                        print(processed_df)

                        plt.figure(figsize=(10, 6))
                        plt.plot(processed_df.index, processed_df['num'], label='Original')
                        plt.scatter(processed_df[processed_df['Anomaly'] == 1].index, processed_df[processed_df['Anomaly'] == 1]['num'], color='red', label='Anomalies', marker='o')
                        plt.xlabel('Time')
                        plt.ylabel('Count')
                        plt.title('Time Series with Anomalies')
                        plt.legend()
                        plt.show() 

                    except:
                        print("Error occurred during model fitting ")    

                else:
                    print("not enough unique values") 
                    return processed_df
                
                    
            else:
                print('Not enough values')  
                return processed_df  
            
                
        else:
            raise Exception("Use 'process_df' to process the data first")


    def zero_poisson(self, lag=1, min_obs=30):
        if self.processed_df is not None:
            processed_df = self.processed_df.copy()
            if len(processed_df) > min_obs:
                if processed_df['num'].nunique() > 1:
                    #store lag cols and formula
                    expr= """ num ~ """
                    lag_cols= []
                    for i in range(1, lag+1):
                        colname=f"num_lag{i}"
                        processed_df[colname] = processed_df["num"].shift(i)
                        lag_cols.append(colname)
                    lag_col_expr = "+".join(lag_cols)
                    expr += lag_col_expr 

                    y,X = dmatrices(expr, processed_df, return_type='dataframe')

                    try:
                        zinb_res= ZINP(y,X).fit(maxiter=500)
                        #test pred on test and calc RMSE
                        pred_values = zinb_res.predict()
                        resid = zinb_res.resid
                        threshold = resid.std()*2
                        #anomalies = processed_df[resid.abs() > threshold]
                        #print(anomalies)

                        # Plot the time series with anomalies highlighted
                        #plt.figure(figsize=(10, 6))
                        #plt.plot(processed_df.index, processed_df['num'], label='Original')
                        #plt.scatter(anomalies.index, anomalies['num'], color='red', label='Anomalies')
                        #plt.xlabel('Time')
                        #plt.ylabel('Count')
                        #plt.title('Time Series with Anomalies')
                        #plt.legend()
                        #plt.show()
            
                    except:
                        print("Error occurred during model fitting ")        
            else:
                print("not enough unique values") 
                return processed_df     
        else: 
            print("Not enough values")
            return processed_df
        

# COMMAND ----------

ae = AnomalyEvent(df1, 'Date')
ae.process_df({'tgt_col':'RecordID', 'agg_typ':'count'}, 'W',filter_dict={'STA': ['Crime', 'Terrorism', 'Armed Conflict']}, date_dict={'start_date':dt.datetime(2018,1,1), 'end_date':dt.datetime(2023,1,31)})
ae.check_zeros()
ae.zero_negbin()

# COMMAND ----------

# instantiate
ae = AnomalyEvent(df, 'TimeFK_Event_Date')
# process
ae.process_df({'tgt_col':'ACLED_PK', 'agg_typ':'count'}, 'W', filter_dict={'ACLED_Event_Type':['Protests']}, date_dict={'start_date':dt.datetime(2021,1,1), 'end_date':dt.datetime(2023,1,31)})
ae.check_zeros()
ae.zero_negbin()

# COMMAND ----------

# iso
iso = ae.get_anomaly('iso', {'contamination':.5})
iso

# COMMAND ----------

# rolling window
rw = ae.get_anomaly('rw', {'window': 30, 'c': 1.5, 'side':'positive'})
rw

# COMMAND ----------


