# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:18:18 2023

@author: l11420
"""
from FedTools import MonetaryPolicyCommittee
from FedTools import BeigeBooks
from FedTools import FederalReserveMins
import pickle
import pandas as pd
import csv
#Statements
#dataset = MonetaryPolicyCommittee().find_statements()
#MonetaryPolicyCommittee().pickle_data(r"C:\Users\L11420\Documents\NLP\statements.pkl")

# with open('directory.pkl', 'rb') as f:
#     data = pickle.load(f)
# data.to_csv(r"C:\Users\pablo\Documents\Fed_Bert_model\statements.csv")



# #Beige Books
# dataset = BeigeBooks().find_beige_books()
# BeigeBooks().pickle_data(r"C:\Users\pablo\Documents\Fed_Bert_model\beige_books.pkl")
# import pickle
# with open('beige_books.pkl', 'rb') as f:
#     data = pickle.load(f)
# #data.to_csv(r"C:\Users\pablo\Documents\Fed_Bert_model\beige_books.csv")


# #Minutes

#dataset = FederalReserveMins().find_minutes()
#FederalReserveMins().pickle_data(r"C:\Users\pablo\Documents\Fed_Bert_model\minutes.pkl")
# import pickle


with open('minutes.pkl', 'rb') as f:
    minutes = pickle.load(f)

federal_fund = pd.read_csv('DFF.csv')
#print(federal_fund['DATE'])

federal_fund['DATE'] = pd.to_datetime(federal_fund['DATE'])

# Assuming 'data' is your data, and you have a column 'Federal_Reserve_Mins'
minutes = pd.DataFrame(minutes, columns=['Federal_Reserve_Mins'])

minutes.reset_index(inplace=True)
# Print the DataFrame to see the result
minutes.columns=['Date', 'Minutes']

# Assuming 'federal_fund' and 'minutes' are your DataFrames
filtered_federal_fund = federal_fund[federal_fund['DATE'] > minutes['Date'].min()]

filtered_federal_fund.columns=['Date', 'DFF']

merged_df = pd.merge(minutes, filtered_federal_fund, on='Date', how='inner')

merged_df['Variacion']= merged_df['DFF'].shift(-1)-merged_df['DFF'] 
merged_df['Dummies'] = [1 if x > 0 else -1 if x < 0 else 0 for x in merged_df['Variacion']]
merged_df.dropna(inplace=True)
print(merged_df)





#print(len(merged_df['Dummies']))





print(" 0","\t",len(merged_df[merged_df['Dummies']==0]),#/len(merged_df['Dummies'])*100, "\n"
      " 1","\t",len(merged_df[merged_df['Dummies']==1]),#/len(merged_df['Dummies'])*100, "\n"
      "-1","\t",len(merged_df[merged_df['Dummies']==-1]))#/len(merged_df['Dummies'])*100)
# federal_fund['DATE'] = pd.to_datetime(federal_fund['DATE'])

# federal_fund['DFF_1'] =federal_fund['DFF'].shift(3) 
# print(federal_fund)