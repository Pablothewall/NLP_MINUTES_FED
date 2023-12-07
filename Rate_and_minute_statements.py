# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:05:21 2023

@author: l11420
"""
from FedTools import MonetaryPolicyCommittee
from FedTools import BeigeBooks
from FedTools import FederalReserveMins
import pickle
import pandas as pd
import csv
import matplotlib.pyplot as plt
import spacy
from mit_NLP_function import spacy_npl


with open('statements.pkl', 'rb') as x:
    statements = pickle.load(x)

with open('minutes.pkl', 'rb') as f:
    minutes = pickle.load(f)
    
minutes = pd.DataFrame(minutes, columns=['Federal_Reserve_Mins'])
minutes.reset_index(inplace=True)
minutes.columns=['Date', 'Minutes']
   
statements = pd.DataFrame(statements, columns=['FOMC_Statements'])
statements.reset_index(inplace=True)
statements.columns=['Date', 'Statements']    

federal_fund = pd.read_csv('DFF.csv')
federal_fund['Date'] = pd.to_datetime(federal_fund['DATE'])

federal_fund['DATE_2days_before'] =  federal_fund['DFF'].shift(-3) 

merged_df = pd.merge(statements, federal_fund, on='Date', how='outer')



merged_df['Date'] = pd.to_datetime(merged_df['Date'])  # Ensure the 'DATE' column is of datetime type
df = merged_df.set_index('Date')  

df_filled = df.fillna(method='ffill')

df_filled = df_filled.reset_index()


merged_df = pd.merge(minutes, df_filled, on='Date', how='inner')
merged_df['Variacion']= merged_df['DATE_2days_before'].shift(-1)-merged_df['DATE_2days_before'] 
merged_df['Dummies'] = [1 if x > 0 else -1 if x < 0 else 0 for x in merged_df['Variacion']]
merged_df.dropna(inplace=True)

   
merged_df=merged_df[["Dummies", "Minutes"]]
#Plot
merged_df['len']=merged_df['Minutes'].apply(lambda x: len(x))
merged_df=merged_df[merged_df['len']<1000000]


merged_df["Minutes"]=merged_df["Minutes"].apply(lambda x: spacy_npl(x))
merged_df=pd.DataFrame(merged_df)
merged_df.to_csv(r"C:\Users\L11420\Documents\NLP\minutes.csv")


#print(merged_df[['Date','len']].sort_values(by='len'))



# Assuming 'Date' and 'len' are columns in your DataFrame

#merged_df['Date'] = pd.to_datetime(merged_df['Date'])  # Convert 'Date' to datetime format

# Plot the data
#merged_df[['Date', 'len']].plot(x='Date', y='len')
#plt.title('Title of the Plot')
#plt.xlabel('Date')
#plt.ylabel('Length')
#plt.show()
data=pd.read_csv("minutes.csv")








