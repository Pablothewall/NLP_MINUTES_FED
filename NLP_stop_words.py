# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:58:52 2023

@author: l11420
"""
import pandas as pd
from FedTools import MonetaryPolicyCommittee
from FedTools import BeigeBooks
from FedTools import FederalReserveMins
import pickle
import pandas as pd
import csv
import matplotlib.pyplot as plt
import spacy
from mit_NLP_function import spacy_npl
from remove_stop_words_function import remove_stop_lemma_words
from remove_stop_words_function import tokenIze
from remove_stop_words_function import join_lists_to_string

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

   

#Plot
merged_df['len']=merged_df['Minutes'].apply(lambda x: len(x))

print(len(merged_df["Minutes"]))


merged_df=merged_df[["Dummies", "Minutes"]]

merged_df['Minutes']=merged_df['Minutes'].apply(lambda x: remove_stop_lemma_words(x))

merged_df['Minutes']=merged_df['Minutes'].apply(lambda x: tokenIze(x))
merged_df['Minutes']=merged_df['Minutes'].apply(lambda x: join_lists_to_string(x))

print(merged_df)

print(merged_df.columns)


df = pd.DataFrame(merged_df)






from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(df["Minutes"], padding="max_length", truncation=True)


tokenized_datasets = df["Minutes"].map(tokenize_function, batched=True)








import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample DataFrame
df = pd.DataFrame(merged_df)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Text Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['Minutes'])
y_train = train_data['Dummies']

# Train a Classifier (using Naive Bayes as an example)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Vectorize the test data
X_test = vectorizer.transform(test_data['Minutes'])
y_true = test_data['Dummies']

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

















































