#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime
import time,os,re,csv,sys,xlrd,yaml,glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.seasonal import seasonal_decompose, STL

class makePlotsTs:
    
    """ 
    Generate EDA plots for time series analysis eg. Year on Year trends, boxplots, decomposition, model plots
    """

    def __init__(self, config):
        self.path = config["project_path"]
        self.data_output = config["data_path"]["output"]
        self.report_img = config["reports"]["images"]
      
    def create_folders(self, subfolders):
        
        self.folders = subfolders
        
        img_folder = os.path.join(self.path, self.report_img)
        
        # check if images folder exist in reports, if not create one
        if not os.path.exists(img_folder):
            os.makedirs(img_folder, exist_ok=True)
        
        # loop through the list of folders we want to create within the images folder
        for sf in subfolders:
            if not os.path.exists(os.path.join(img_folder, sf)):
                os.makedirs(os.path.join(img_folder, sf), exist_ok=True)
            

    def decompose_ts(self, df, label, freq = 'W'):

        self.freq = freq
        
        ts = df[['TaskDate', 'TaskCount']].set_index('TaskDate').resample(freq).sum()

        sns.set(rc={"figure.figsize": (10, 8)})
        print(f"{freq} decomposition of {label}")

        try:

            # Decomposition 1
            result = seasonal_decompose(ts, model='additive') # {model='additive', model='multiplicative'}, optional
            fig = result.plot()
            fig.savefig(os.path.join(self.path, self.report_img, "decompose", "decompose_" + label + "_" + self.freq + ".png"))
            plt.close()

        except:

            # Decomposition with STL
            result = STL(ts).fit()
            fig = result.plot()
            fig.savefig(os.path.join(self.path, self.report_img, "decompose", "decompose_" + label + "_" + self.freq + ".png"))
            plt.close()
            
    def eda_ts(self, df, label, freq = 'M'):
        
        self.freq = freq
        
        # Gather 2019 data
        ts_1 = df[(df['TaskDate'] > '2018-12-31') & (df['TaskDate'] < '2020-01-01')]
        ts_1 = ts_1[['TaskDate', 'TaskCount']].set_index('TaskDate').resample(self.freq).sum().reset_index()

         # Gather 2020 data
        ts_2 = df[(df['TaskDate'] > '2019-12-31') & (df['TaskDate'] < '2020-12-31')]
        ts_2 = ts_2[['TaskDate', 'TaskCount']].set_index('TaskDate').resample(self.freq).sum().reset_index()

        comb = pd.concat([ts_1, ts_2], axis=0)
        comb['Month'] = comb['TaskDate'].dt.month
        comb['Year'] = comb['TaskDate'].dt.year

        # Line plot Y on Y
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))
        sns.lineplot(comb['Month'], comb['TaskCount'], hue=comb['Year'], palette='flare', ax=ax[0])
        ax[0].set_title(f"2019 vs 2020 Trend for {label}", size=15)

        # Box plot by month
        sns.boxplot(comb['Month'], comb['TaskCount'], ax=ax[1])

        fig.savefig(os.path.join(self.path, self.report_img, "eda", "eda_" + label + "_" + self.freq + ".png"))
        
        plt.close()

        def month_converter(num):

            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            months_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            dict_mth = dict(list(zip(months_num, months)))
            return dict_mth[num]

        median = [(i,np.median(comb[comb['Month'] == i]['TaskCount'].tolist())) for i in range(1,13)]
        std = [(i,np.std(comb[comb['Month'] == i]['TaskCount'].tolist())) for i in range(1,13)]

        median.sort(key=lambda x:x[1])

        median_2_low_mths = median[0:2]
        median_2_low_mths_num = [i[0] for i in median_2_low_mths]
        median_2_low_mths_names = [month_converter(j) for j in median_2_low_mths_num] 
        mth_handle_1 = ""
        for i in median_2_low_mths_names:
            mth_handle_1 += str(i + " ") 

        median_2_high_mths = median[-2:]
        median_2_high_mths_num = [i[0] for i in median_2_high_mths]
        median_2_high_mths_names = [month_converter(j) for j in median_2_high_mths_num] 
        mth_handle_2 = ""
        for i in median_2_high_mths_names:
            mth_handle_2 += str(i + " ") 

        std.sort(key=lambda x:x[1])
        overlap_confidence = std[0:4]
        overlap_confidence_mth_num = [i[0] for i in overlap_confidence]
        overlap_confidence_mth_names = [month_converter(j) for j in overlap_confidence_mth_num] 
        mth_handle_3 = ""
        for i in overlap_confidence_mth_names:
            mth_handle_3 += str(i + " ") 

        eda_result = pd.DataFrame({"Language": [label],
                                   "lowest_2_mths": [mth_handle_1],
                                   "highest_2_mths": [mth_handle_2] ,
                                   "overlap_confidence": [mth_handle_3]})
        
        return eda_result
    
def generate_plots():    
    
    # Function to load yaml configuration file
    def load_config(config_name):
        with open(os.path.join(config_path, config_name), 'r') as file:
            config = yaml.safe_load(file)

        return config

    config_path = "conf/base"

    config = load_config("catalog.yml")

    # Start exploratory data analysis process
    print("\nStarting time series decomposition and exploratory data analysis (EDA) process...\n")
    
    start_time = time.time()
    
    make_plots = makePlotsTs(config)
    
    subfolder_list = ["decompose", "eda"]
    make_plots.create_folders(subfolder_list)
    
    eda_summary = []
    for f in glob.glob(os.path.join(os.path.join(config["project_path"], config["data_path"]["output"], "*.csv"))):
    
        # get labels or filenames
        fname = os.path.basename(f)
        start = fname.find("ts_") + len("ts_")
        end = fname.find(".csv")
        label = fname[start:end]

        df = pd.read_csv(f)
        
        #convert date columns as datetime
        df['TaskDate'] = pd.to_datetime(df['TaskDate'])
        
        make_plots.decompose_ts(df, label)  # default to 'W' , if need to change to other resample, add freq next to label. 
              
        eda_result = make_plots.eda_ts(df, label)
        eda_summary.append(eda_result)
    
    eda_summary = pd.concat(eda_summary)
        
    end_time = time.time()
    processing_time = round((end_time - start_time)/60,2)
        
    print(f"\nTime series decomposition and EDA completed in {processing_time} minutes")   
    print(f"\nDecomposition plots are stored in reports > images > decompose folder")        
    print(f"\nEDA plots are stored in reports > images > eda folder\n")
    
    return eda_summary
    
if __name__ == '__main__':
    
    eda_summary = generate_plots()
    #print(eda_summary)

