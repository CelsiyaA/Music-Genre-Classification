#Data manipulation and analysis
import pandas as pd
#Numerical calculation
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
#To work with dataframe
import pandas as pd
#To perform numerical labels
import numpy as np

#--------------------------------------------------------------------------
#DATA PREPROCESSING
#---------------------------------------------------------------------------

#Importing lyricsdata, metadata csv files
Audio_data = pd.read_csv("C:\Dissertation\Music4all_dataset\Vectorisation\Audio_features.csv")

df = Audio_data.copy()

print(df.head(5))

df.shape #4531 records with 89 features

#Information about the data
df.info()
#No null values present in the data
#Id and genre are of object type

#Null values
df.isnull().sum()
#There are no null values in it.

#Descriptive statistics for continous variables

Summ_stat = df.describe()
#Most of the variables has larger variations between the data point (right or left skewed),
#But we won't consider this as  outliers, as each data points carries its own characteristics.

#duplicate records
df.duplicated().value_counts() # No duplicate records

#Printing the unique values of each column
df.nunique()
#----------------------------------------------------------
#Visualizations
#Percentage of genres

#Count of genres
count_genres  = df['genres'].value_counts()

#Pie chart
plt.figure(figsize=(8, 8))
plt.pie(count_genres, labels= count_genres.index, autopct='%1.1f%%', startangle=120)
plt.axis('equal')
plt.title('Percentage of genres')
plt.show()


#----------------------------------------------------------------------------------
#Correlation between continous variables

#correlation
corr = df.corr()
#Masking for heatmap
mask_heatmap = np.zeros_like(corr)
mask_heatmap[np.triu_indices_from(mask_heatmap)] = True
plt.subplots(figsize=(20, 20))
sns.heatmap(corr, mask=mask_heatmap, cmap="BuPu")

#---------------------------------------------------------
#Pairwise correlation
df_mean_col = df[[x for x in df.columns if 'mean' in x]]
df_std_col = df[[x for x in df.columns if 'std' in x]]

#Correlation for columns with mean
plt.figure(figsize=(10,10))
sns.heatmap(df_mean_col.corr(), cmap='BuPu', vmin=-1, vmax=1,center=0, square=True, annot=True, annot_kws={'fontsize':5})
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plt.show()


#Correlation without mfcc
plt.figure(figsize=(10,10))
sns.heatmap(df_mean_col.iloc[:,:17].corr(), cmap='BuPu', vmin=-1, vmax=1,center=0, square=True, annot=True)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plt.show()

#Correlation for columns with standard deviation
plt.figure(figsize=(10,10))
sns.heatmap(df_std_col.corr(), cmap='BuPu', vmin=-1, vmax=1,center=0, square=True, annot=True, annot_kws={'fontsize':5})
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plt.show()

#correlation without mfcc
plt.figure(figsize=(10,10))
sns.heatmap(df_std_col.iloc[:,:17].corr(), cmap='BuPu', vmin=-1, vmax=1,center=0, square=True, annot=True)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plt.show()


#Mean and standard deviation of all variables without mfcc and chroma

# Merge the DataFrames
df_combined = pd.concat([df_mean_col.iloc[:,:17], df_std_col.iloc[:,:17]], axis=1)

# Calculate the correlation matrix
corr_matrix = df_combined.corr()

# Create the heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, cmap='BuPu', vmin=-1, vmax=1, center=0, square=True, annot=True, annot_kws={'fontsize':5})
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()



#https://www.kaggle.com/code/jojothepizza/genre-classification-and-musical-features-analysis

#--------------------------------------------------------
#Features selected with RFE

#Importing lyricsdata, metadata csv files
Audio_data = pd.read_csv("C:\Dissertation\Music4all_dataset\Vectorisation\Features_RFE.csv")

df = Audio_data.copy()

print(df.head(5))

df.shape #4531 records with 89 features

#Information about the data
df.info()
#No null values present in the data
#Id and genre are of object type

#Null values
df.isnull().sum()
#There are no null values in it.

#Descriptive statistics for continous variables

Summ_stat = df.describe()
#Most of the variables has larger variations between the data point (right or left skewed),
#But we won't consider this as  outliers, as each data points carries its own characteristics.

#duplicate records
df.duplicated().value_counts() # No duplicate records

#Printing the unique values of each column
df.nunique()
#----------------------------------------------------------
#Exploratory data analysis

correlation_plot = df.corr()

#correlation matrix heatmap
plt.figure(figsize=(12, 10))  
sns.heatmap(correlation_plot, annot=True, cmap='coolwarm')
plt.title("Correlation matrix of selected RFE features")
plt.show()



