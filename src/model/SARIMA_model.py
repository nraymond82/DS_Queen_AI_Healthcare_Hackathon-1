#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime
import time,os,re,csv,sys,xlrd,yaml,glob
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings("ignore")

import joblib
import math
from itertools import product 
import statsmodels.api as sm

def ingest_processed_dataset():

    # Function to load yaml configuration file
    def load_config(config_name):
        with open(os.path.join(config_path, config_name), 'r') as file:
            config = yaml.safe_load(file)

        return config

    config_path = "conf/base"

    config = load_config("catalog.yml")
    
    return config

def create_folders(root_path, img_path, subfolders):
               
        img_folder = os.path.join(root_path, img_path)
        
        # check if images folder exist in reports, if not create one
        if not os.path.exists(img_folder):
            os.makedirs(img_folder, exist_ok=True)
        
        # loop through the list of folders we want to create within the images folder
        for sf in subfolders:
            if not os.path.exists(os.path.join(img_folder, sf)):
                os.makedirs(os.path.join(img_folder, sf), exist_ok=True)
                
def model_SARIMA_auto(config, df, label, freq = 'W', n_periods = 15):
    
    # filter the data
    ts = df[['TaskDate', 'TaskCount']].set_index('TaskDate').resample(freq).sum()

    # iterate through the range of p,d,q, P,D,Q and obtain the best (lowes) AIC
    p = range(0, 3)
    d = range(1,2)
    q = range(0, 3)
    pdq = list(product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(product(p, d, q))]
    print(f"Optimizing Seasonal ARIMA for {label}...")

    aic = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
#             try:
            model = sm.tsa.statespace.SARIMAX(ts['TaskCount'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = model.fit()
            aic.append((param, param_seasonal, results.aic))
#             except:
#                 continue
    
    # get model best parameters and aic
    best_params = min(aic, key=lambda x:x[2])
    print(f"Best SARIMA parameters: {best_params} for {label}")
    
    # finalize and fit the best model with best parameters
    model_SARIMA = sm.tsa.statespace.SARIMAX(ts['TaskCount'],
                                order= best_params[0],
                                seasonal_order = best_params[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    
    # save best model as pickle (serialized) file
    joblib.dump(model_SARIMA, os.path.join(config["project_path"], config["models"], "model_SARIMA_" + label + '_' + freq + '.pkl'))

    results = model_SARIMA.fit()
    aic = round(best_params[2],2)
      
    # obtain the model prediction from the start of the data - baseline is formed from the mean predicted and the CIs
    pred = results.get_prediction(start=min(ts.index))
    pred_ci = pred.conf_int()
    
    # plot model results
    fig, ax = plt.subplots(figsize=(10,4))
    ax = ts['TaskCount'].plot(label='actual')
    pred.predicted_mean.plot(ax=ax, label='predicted', color='darkgreen', alpha=.7, linewidth=1, linestyle='dashed')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    legend = plt.legend(loc='upper left')  # remove later
    legend.get_frame().set_alpha(0) # remove later
    ax.ticklabel_format(useOffset=False, style='plain', axis='y') # remove later
    plt.title(f"SARIMA model forecast (W) - {label}, AIC = {aic}") # remove later
    ax.set_ylabel("Total Task Counts")
    sns.despine()
    
    
    fc_start = max(ts.index) + timedelta(days=1)
    index_of_fc = pd.date_range(fc_start, periods=n_periods, freq='W').tolist()
    
    pred_dy = results.get_prediction(start=fc_start , end=max(index_of_fc))
    pred_dy_ci = pred_dy.conf_int()
    
    pred_dy.predicted_mean.plot(ax=ax, label='forecast', color='red', linewidth=1, linestyle='dashed', alpha=0.7)
    ax.fill_between(pred_dy_ci.index,
                    pred_dy_ci.iloc[:, 0],
                    pred_dy_ci.iloc[:, 1], color='k', alpha=.25)
    
    ax.fill_betweenx(ax.get_ylim(), pd.Timestamp(fc_start), max(index_of_fc),
                     alpha=.1, zorder=-1) 
    legend = plt.legend(loc='upper left')
    legend.get_frame().set_alpha(0)
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')
    plt.title(f"SARIMA model forecast (W) - {label}, AIC = {aic}")
    sns.despine()
    
    # Adjusted baseline - if lower ci is > 0, get the lowest value of lower_ci
    adjusted_baseline = min(pred_ci[pred_ci['lower TaskCount'] > 0]['lower TaskCount'])
    ax.axhline(adjusted_baseline, ls='--', color='lightseagreen', linewidth=1)
    plt.savefig(os.path.join(config["project_path"], config["reports"]["images"], "model_plots/mp_" + label + "_" + freq + ".jpeg"))
    
    plt.close()
       
    return pred, pred_ci, pred_dy, pred_dy_ci, label                
                        
def generate_baselines(pred, pred_ci, pred_dy, pred_dy_ci, label):
    
    # Get Weekly predicted table - mean prediction from the model is used as Weekly Baseline
    W_predicted = pd.DataFrame()
    W_predicted['W_baseline'] = pred.predicted_mean
    W_predicted['W_lower_bl'] = pred_ci.iloc[:, 0]
    try:
        adjusted_baseline = min(pred_ci[pred_ci['lower TaskCount'] > 0]['lower TaskCount'])
        W_predicted['W_lower_bl'] = np.where(W_predicted['W_lower_bl'] <= 0, adjusted_baseline, W_predicted['W_lower_bl'])
    except:
        W_predicted['W_lower_bl'] = np.where(W_predicted['W_lower_bl'] <= 0, W_predicted['W_baseline'], W_predicted['W_lower_bl'])
    W_predicted['W_upper_bl'] = pred_ci.iloc[:, 1]
    W_predicted['month_num'] = pd.Series(W_predicted.index).dt.strftime('%m').astype(int).tolist()
    W_predicted['month'] = pd.Series(W_predicted.index).dt.month_name().str.slice(stop=3).tolist()   #.dt.strftime('%m')
    W_predicted['week_in_month'] = pd.to_numeric(W_predicted.index.day/7)
    W_predicted['week_in_month'] = W_predicted['week_in_month'].apply(lambda x: math.ceil(x))
    W_predicted['Language'] = label
    W_baseline = W_predicted.groupby(['Language','month_num','month','week_in_month']).mean().reset_index().sort_values(['month_num', 'week_in_month'])
    W_baseline = W_baseline[['Language', 'month', 'week_in_month','W_baseline', 'W_lower_bl', 'W_upper_bl']]
    
    # Get Monthly predicted table - aggregated from the Weekly baseline
    M_predicted = W_predicted[['W_baseline', 'W_lower_bl', 'W_upper_bl']].resample('M').sum()
    M_predicted.columns = ['M_baseline', 'M_lower_bl', 'M_upper_bl']
    M_predicted['Language'] =label
    M_predicted['month_num'] = pd.Series(M_predicted.index).dt.strftime('%m').astype(int).tolist()
    M_predicted['month'] = pd.Series(M_predicted.index).dt.month_name().str.slice(stop=3).tolist() 
    M_baseline = M_predicted.groupby(['Language','month_num','month']).mean().reset_index().sort_values('month_num')
    M_baseline = M_baseline[['Language','month','M_baseline', 'M_lower_bl', 'M_upper_bl']]
    
    # Get Quarterly predicted table - aggregated from the Monthly baseline
    Q_predicted = M_predicted[['M_baseline', 'M_lower_bl', 'M_upper_bl']].resample('Q').sum()
    Q_predicted.columns = ['Q_baseline', 'Q_lower_bl', 'Q_upper_bl']
    Q_predicted['Language'] = label
    Q_predicted['month_num'] = pd.Series(Q_predicted.index).dt.strftime('%m').astype(int).tolist()
    Q_predicted['month'] = pd.Series(Q_predicted.index).dt.month_name().str.slice(stop=3).tolist() 
    Q_baseline = Q_predicted.groupby(['Language','month_num','month']).mean().reset_index().sort_values(['month_num'])
    conditions  = [ Q_baseline['month_num'] == 3, Q_baseline['month_num'] == 6, Q_baseline['month_num'] == 9, Q_baseline['month_num'] == 12]
    quarter_name, quarter_num = [ 'Q1', 'Q2', 'Q3', 'Q4'], [1,2,3,4]
    Q_baseline['quarter'] = np.select(conditions, quarter_name, default=np.nan)
    Q_baseline['q'] = np.select(conditions, quarter_num, default=np.nan).astype(int)
    Q_baseline = Q_baseline[['Language','quarter','Q_baseline', 'Q_lower_bl', 'Q_upper_bl']]
    
    W_forecast = pd.DataFrame()
    W_forecast['W_forecast'] = pred_dy.predicted_mean
    W_forecast['W_lower_fc'] = pred_dy_ci.iloc[:, 0]  
    try:
        adjusted_baseline_2 = min(pred_dy_ci[pred_dy_ci['lower TaskCount'] > 0]['lower TaskCount'])
        W_forecast['W_lower_fc'] = np.where(W_forecast['W_lower_fc'] <= 0, adjusted_baseline_2, W_forecast['W_lower_fc'])
    except:
        W_forecast['W_lower_fc'] = np.where(W_forecast['W_lower_fc'] <= 0, W_forecast['W_forecast'], W_forecast['W_lower_fc'])
    W_forecast['W_upper_fc'] = pred_dy_ci.iloc[:, 1]
    W_forecast['month_num'] = pd.Series(W_forecast.index).dt.strftime('%m').astype(int).tolist()
    W_forecast['month'] = pd.Series(W_forecast.index).dt.month_name().str.slice(stop=3).tolist()   #.dt.strftime('%m')
    W_forecast['week_in_month'] = pd.to_numeric(W_forecast.index.day/7)
    W_forecast['week_in_month'] = W_forecast['week_in_month'].apply(lambda x: math.ceil(x))
    W_forecast['Language'] = label
    W_forecast = W_forecast[['Language', 'month', 'week_in_month', 'W_forecast', 'W_lower_fc', 'W_upper_fc']]
    
    M_forecast = W_forecast[['W_forecast', 'W_lower_fc', 'W_upper_fc']].resample('M').sum()
    M_forecast.columns = ['M_forecast', 'M_lower_fc', 'M_upper_fc']
    M_forecast['Language'] = label
    M_forecast['month_num'] = pd.Series(M_forecast.index).dt.strftime('%m').astype(int).tolist()
    M_forecast['month'] = pd.Series(M_forecast.index).dt.month_name().str.slice(stop=3).tolist() 
    M_forecast = M_forecast[['Language', 'month', 'M_forecast', 'M_lower_fc', 'M_upper_fc']]   
    
    Q_forecast = M_forecast[['M_forecast', 'M_lower_fc', 'M_upper_fc']].resample('Q').sum()
    Q_forecast.columns = ['Q_forecast', 'Q_lower_fc', 'Q_upper_fc']
    Q_forecast['Language'] = label
    Q_forecast['month_num'] = pd.Series(Q_forecast.index).dt.strftime('%m').astype(int).tolist()
    Q_forecast['month'] = pd.Series(Q_forecast.index).dt.month_name().str.slice(stop=3).tolist() 
    conditions = [Q_forecast['month_num'] == 3, Q_forecast['month_num'] == 6, Q_forecast['month_num'] == 9, Q_forecast['month_num'] == 12]
    quarter_name, quarter_num = [ 'Q1', 'Q2', 'Q3', 'Q4'], [1,2,3,4]
    Q_forecast['quarter'] = np.select(conditions, quarter_name, default=np.nan)
    Q_forecast['q'] = np.select(conditions, quarter_num, default=np.nan).astype(int)
    Q_forecast = Q_forecast.sort_values(['month_num'])
    Q_forecast = Q_forecast[['Language','quarter','Q_forecast', 'Q_lower_fc', 'Q_upper_fc']]
    
    W_baseline[['W_baseline', 'W_lower_bl', 'W_upper_bl']] = W_baseline[['W_baseline', 'W_lower_bl', 'W_upper_bl']].astype(int)
    M_baseline[['M_baseline', 'M_lower_bl', 'M_upper_bl']] = M_baseline[['M_baseline', 'M_lower_bl', 'M_upper_bl']].astype(int)
    Q_baseline[['Q_baseline', 'Q_lower_bl', 'Q_upper_bl']] = Q_baseline[['Q_baseline', 'Q_lower_bl', 'Q_upper_bl']].astype(int)
 
    W_forecast[['W_forecast', 'W_lower_fc', 'W_upper_fc']] = W_forecast[['W_forecast', 'W_lower_fc', 'W_upper_fc']].astype(int)
    M_forecast[['M_forecast', 'M_lower_fc', 'M_upper_fc']] = M_forecast[['M_forecast', 'M_lower_fc', 'M_upper_fc']].astype(int) 
    Q_forecast[['Q_forecast', 'Q_lower_fc', 'Q_upper_fc']] = Q_forecast[['Q_forecast', 'Q_lower_fc', 'Q_upper_fc']].astype(int) 
    
    return W_baseline, M_baseline, Q_baseline, W_forecast, M_forecast, Q_forecast

def write_baseline_report_to_excel(wb, mb, qb, wf, mf, qf, encoding=None):
    
    config = ingest_processed_dataset()
    
    # store all 4 reports into a dictionary set
    list_of_datasets = {"Weekly Baseline" : wb,
                        "Monthly Baseline" : mb,
                        "Quarterly Baseline" : qb,
                        "Weekly Forecast" : wf,
                        "Monthly Forecast" : mf,
                        "Quarterly Forecast" : qf}
    
    with pd.ExcelWriter(os.path.join(config["project_path"], config["reports"]["predictions"], 'Workflow_AI_Baseline_Report.xlsx')) as writer:  
        for key, value in list_of_datasets.items():
            value.to_excel(writer, sheet_name=key, index=False, encoding=None)

def model_train(df, label, freq = 'W', n_periods = 15, test=False): 
    
    config = ingest_processed_dataset()
        
    pred, pred_ci, pred_dy, pred_dy_ci, label = model_SARIMA_auto(config, df, label, freq = 'W', n_periods = 15)
    
    return pred, pred_ci, pred_dy, pred_dy_ci, label
    
def model_predict(pred, pred_ci, pred_dy, pred_dy_ci, label, model=None, steps = 15, test=False):
    
    W_baseline, M_baseline, Q_baseline, W_forecast, M_forecast, Q_forecast = generate_baselines(pred, pred_ci, pred_dy, pred_dy_ci, label)
    
    return W_baseline, M_baseline, Q_baseline, W_forecast, M_forecast, Q_forecast
    
# def model_load(test=False):
#     """
#     example funtion to load model
#     """
#     if test : 
#         print( "... loading test version of model" )
#         model = joblib.load(os.path.join("models","test.joblib"))
#         return(model)

#     if not os.path.exists(SAVED_MODEL):
#         exc = "Model '{}' cannot be found did you train the full model?".format(SAVED_MODEL)
#         raise Exception(exc)
    
#     model = joblib.load(SAVED_MODEL)
#     return(model)

def model_train_predict(eda_summary):
    
    freq = 'W'
    steps = 15

    config = ingest_processed_dataset()
    
    subfolder_list = ['model_plots']
    create_folders(config["project_path"], config["reports"]["images"], subfolder_list)
    
    # Start exploratory data analysis process
    print("\nStarting time series training with SARIMA model...\n")
    
    start_time = time.time()
    
    wb, mb, qb, wf, mf, qf = [],[],[],[],[],[]
    for f in glob.glob(os.path.join(os.path.join(config["project_path"], config["data_path"]["output"], "*.csv"))):
    
        # get labels or filenames
        fname = os.path.basename(f)
        start = fname.find("ts_") + len("ts_")
        end = fname.find(".csv")
        label = fname[start:end]

        df = pd.read_csv(f)
        
        #convert date columns as datetime
        df['TaskDate'] = pd.to_datetime(df['TaskDate'])  
        
        pred, pred_ci, pred_dy, pred_dy_ci, label = model_train(df, label, freq, steps, test=False)
        W_baseline, M_baseline, Q_baseline, W_forecast, M_forecast, Q_forecast = model_predict(pred, pred_ci, pred_dy, pred_dy_ci, label, model=None, steps = 15, test=False)
        wb.append(W_baseline)
        mb.append(M_baseline)
        qb.append(Q_baseline)
        wf.append(W_forecast)
        mf.append(M_forecast)
        qf.append(Q_forecast)
        
    wb = pd.concat(wb, ignore_index=True)
    mb = pd.concat(mb, ignore_index=True)  
    mb = pd.merge(mb,eda_summary[['Language','lowest_2_mths', 'highest_2_mths', 'overlap_confidence']],on='Language', how='left')
    qb = pd.concat(qb, ignore_index=True)    
    wf = pd.concat(wf, ignore_index=True) 
    mf = pd.concat(mf, ignore_index=True)
    mf = pd.merge(mf,eda_summary[['Language','lowest_2_mths', 'highest_2_mths', 'overlap_confidence']],on='Language', how='left')
    qf = pd.concat(qf, ignore_index=True)
    
    write_baseline_report_to_excel(wb, mb, qb, wf, mf, qf, encoding=None)
        
    end_time = time.time()
    processing_time = round((end_time - start_time)/60,2)
        
    print(f"\nTime series model training completed in {processing_time} minutes")   
    print(f"\nModel plots are stored in reports > images > model_plots folder")   
    
    return pred, pred_ci, pred_dy, pred_dy_ci, label        
        
    
if __name__ == '__main__':
    
    #eda_summary = []
    model_train_predict(eda_summary)

