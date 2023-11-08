#features

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder

Audio_features = pd.read_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\Audio_features.csv',index_col = False)

df = Audio_features.copy()
#Applying label encoder to lyrics column (converting text to numbers)
df[['genres']] =  df[['genres']].apply(LabelEncoder().fit_transform)
#df = Audio_features.copy()
X = df.drop('genres', axis=1)  # Features (all columns except 'genres')
X = X.drop('id', axis=1)
y = df['genres']  

#https://sklearn-evaluation.ploomber.io/en/latest/optimization/feature_selection.html

#----------------------------------------------------------------------
from sklearn.feature_selection import RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
sel = RFE(RandomForestClassifier(n_estimators = 100, random_state = 0, n_jobs=-1), n_features_to_select = 20)
sel.fit(X, y)

sel.get_support()
#Features selection
selected_features = X.columns[sel.get_support()]
selected_feature_names = X.columns[sel.get_support()].tolist()

#Creating a dataframe for selected features
X_selected_df = pd.DataFrame(X[selected_features], columns=selected_feature_names)

#concatenating id and genre with RFE columns
Features_RFE = pd.concat([df['genres'], X_selected_df], axis=1)
Features_RFE = pd.concat([df['id'], Features_RFE], axis=1)
Features_RFE.reset_index(drop=True, inplace=True)
import os
os.makedirs(r'C:\Dissertation\Music4all_dataset', exist_ok=True)  
Features_RFE.to_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\Features_RFE.csv', index = False) 

#--------------------------------------------------------------------------------
