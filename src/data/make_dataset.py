#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime
import time,os,re,csv,sys,xlrd,yaml,glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

class makeDatasetTs:
    
    """ 
    Data ingestion, cleaning and transformation for time series analysis
    """

    def __init__(self, config):
        self.path = config["project_path"]
        self.data_input = config["data_path"]["input"]
        self.data_output = config["data_path"]["output"]
        self.data_ref = config["data_path"]["ref"]
        self.report_img = config["reports"]["images"]

    def ingest_raw_dataset(self):
               
        
        filepaths = glob.glob(os.path.join(self.path, self.data_input, "*.xlsx"))
        all_data = pd.DataFrame()

        try:
            checksum = []
            for f in glob.glob(os.path.join(self.path, self.data_input, "*.xlsx")):
                df = pd.read_excel(f, sheet_name='Task Count')
                checksum.append((f, len(df), sum(df['TaskCount'])))
                all_data = all_data.append(df,ignore_index=True)

            #save checksum
            checksum = pd.DataFrame(checksum, columns = ['filename', 'row_count', 'TaskCountSum'])
            checksum.to_csv(os.path.join(self.path, self.data_ref, "metadata.csv"), index=False)

        except:
            print(f"Found problematic file: {f}")
            
        return all_data
    
    
    def clean_dataset(self, all_data):
        
        self.all_data = all_data
        all_data = self.all_data
        
        #Convert dates from object to timestamps
        all_data['TaskDate'] = all_data['TaskDate'].apply(lambda x: pd.Timestamp(x))
        all_data['ReportDate'] = all_data['ReportDate'].apply(lambda x: pd.Timestamp(x))
              
        #Convert string dates in excel to timestamps - this will take a few minutes to run
        rogue_list = [ '1970-01-01 00:00:00.0000' + str(rogue) for rogue in all_data['TaskDate'].unique().tolist() if len(str(rogue)) != 19]
        for rl in rogue_list:

            rogues = all_data[all_data['TaskDate'] == rl]
            rogues_idx = rogues.index.tolist()

            for idx in rogues_idx:
                r_int = int(rogues['TaskDate'].astype('str')[idx][-5:])
                r_int_2 = int(rogues['ReportDate'].astype('str')[idx][-5:])
                all_data.loc[idx, 'TaskDate'] = datetime(*xlrd.xldate_as_tuple(r_int, 0))
                all_data.loc[idx, 'ReportDate'] = datetime(*xlrd.xldate_as_tuple(r_int_2, 0))
        
        print("\n")
        print(f"Floor Date: {min(all_data['TaskDate'])}")
        print(f"Ceiling Date: {max(all_data['TaskDate'])}\n")
        
        all_data['EmailId']=pd.factorize(all_data['RaterEmail'])[0]+1
        all_data = all_data.drop(['RaterEmail'], axis=1)
        
        all_data.to_csv(os.path.join(self.path, self.data_ref, "all_data.csv"), index=False)
        
        return all_data
    
    def select_dataset(self, all_data, freq = 'W'):
        
        self.all_data = all_data
        self.freq = freq
        all_data = self.all_data
        
        df_summary = pd.DataFrame(all_data.groupby(["Language"])['TaskCount'].sum()).reset_index()
        df_summary['RaterCount'] = list(all_data.groupby('Language')['EmailId'].nunique())
        df_summary['ProjectCount'] = list(all_data.groupby('Language')['Rater Visible Name'].nunique())
        date_min_max = all_data.groupby('Language').agg({'TaskDate': [np.min,np.max]}).reset_index()['TaskDate'].rename(columns = {"amin": "MinTaskDate", "amax":"MaxTaskDate"})
        df_summary = pd.concat([df_summary, date_min_max], axis=1)
        df_summary = df_summary.sort_values(['TaskCount', 'RaterCount'], ascending=[False,False],ignore_index=True)
        df_summary['Market'] = np.where(df_summary['MinTaskDate'].dt.year > 2020 , 'New', 'Current')

        conditions = [
            (df_summary['TaskCount'] >= 10000000),
            ((df_summary['TaskCount'] < 10000000) & (df_summary['TaskCount'] >= 400000)),
            (df_summary['TaskCount'] < 400000)]

        choices = ['High','Medium','Low']
        df_summary['Volume'] = np.select(conditions, choices, default=np.nan)
        
        sns.set(rc={"figure.figsize": (15, 8)})
        sns.set_palette(reversed(sns.color_palette("Blues_d", 45)), 45)
        ax = sns.barplot(y = "Language", x= "TaskCount", data=df_summary, alpha=0.7)
        ax.axhline(30, ls='--', color='r')
        ax.text(200,31,"Cut-off : Low volume and new market", size=10, verticalalignment='center')
        ax.set_title("Total task count by Language-Location", size=15) 
        plt.savefig(os.path.join(self.path, self.report_img, "selected_data.png"))
        plt.close()
        
        selected_data = df_summary[(df_summary['Market'] == 'Current') & (df_summary['Volume'] != 'Low')]
        selected_data = selected_data['Language'].unique().tolist()
        
        for sd in selected_data:
            
            selection = all_data[all_data['Language'] == sd]
            ts = selection[['TaskDate', 'TaskCount']].set_index('TaskDate').resample(self.freq).sum().reset_index()
            ts.to_csv(os.path.join(self.path, self.data_output, "ts_" + sd + ".csv"), index=False)
        
        return selected_data
    
def make_dataset():
    
    # Function to load yaml configuration file
    def load_config(config_name):
        with open(os.path.join(config_path, config_name), 'r') as file:
            config = yaml.safe_load(file)

        return config

    config_path = "conf/base"

    config = load_config("catalog.yml")

    # Start data processing and transformation
    
    print("\nStarting data ingestion, cleaning and transformation ...")
    
    start_time = time.time()
    
    make_data = makeDatasetTs(config)
    
    all_data = make_data.ingest_raw_dataset()
    cleaned_data = make_data.clean_dataset(all_data)
    selected_data = make_data.select_dataset(all_data)
    
    #print(selected_data.head())
        
    end_time = time.time()
    
    processing_time = round((end_time - start_time)/60,2)
    
    print(f"\nData ingestion, cleaning and transformation completed in {processing_time} minutes")
    
    print(f"\nAll data is stored in data > reference folder")
    print(f"\nLocale data is stored in data > processed folder\n")


if __name__ == '__main__':
    
    make_dataset()

