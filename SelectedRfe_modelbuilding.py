#________________________________________________________________________________________
# Music genre classiifcation on fusing audio and lyrics features
#______________________________________________________________________________________

#To work with dataframe
import pandas as pd
#To perform numerical labels
import numpy as np
#Label encoder for categorical variable
from sklearn.preprocessing import LabelEncoder
#Train test split
from sklearn.model_selection import train_test_split
#Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_curve, auc

from sklearn import metrics

#XGboost model
from xgboost import XGBClassifier
import xgboost as xgb
#Multinominal logistic regression
from sklearn.linear_model import LogisticRegressionCV
#Linearsvc
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#Histgradient boosting
from sklearn.ensemble import HistGradientBoostingClassifier
#Gradient boost
from sklearn.ensemble import GradientBoostingClassifier
#For cross validatation
from sklearn.model_selection import cross_val_score, KFold
#Plotting
import matplotlib.pyplot as plt
#For roc_auc-score
from sklearn.metrics import roc_auc_score
#To compute the sample weight
from sklearn.utils.class_weight import compute_sample_weight
#To convert labels to one_hot encoding
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score

#from imblearn.over_sampling import ADASYN

#Neural network with gru
#https://towardsdatascience.com/machine-learning-multiclass-classification-with-imbalanced-data-set-29f6a177c1a
#--------------------------------------------------------------

#Reading the audio and lyrics data
Audio_features = pd.read_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\Features_RFE.csv',index_col = False)
Glove_embeddings = pd.read_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\Glove.csv',index_col = False)

#Value count of genre column
Glove_embeddings['genres'].value_counts()

#Dropping the genre column
Glove_embeddings = Glove_embeddings.drop('genres',axis = 1)

#Merging the audio and lyrics columns
df = pd.merge(Audio_features, Glove_embeddings, on = "id")

#Applying label encoder to lyrics column (converting text to numbers)
df[['genres']] =  df[['genres']].apply(LabelEncoder().fit_transform)
df['genres'].value_counts()

# Splitting the independent and target variable as 'x' and 'y' 
X = df.drop('genres', axis=1)  
X = X.drop('id', axis=1)
y = df['genres']  

# Splitting the dataset into train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape
y_train.value_counts()

#-------------------------------------------------------------------------
#XGBoost model
#-------------------------------------------------------------------------

#Learning rate of list
learning_rate_list = [0.001, 0.01,0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

#For loop for XGBoost
for learning_rate in learning_rate_list:
    #XGBoost classiifier
    XGBoost_classifier = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        max_depth=8,
        learning_rate= learning_rate,
        n_estimators=150,
        subsample=0.5
    )
    #Fitting the training data
    XGBoost_model = XGBoost_classifier.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))  

    #predictions on test data
    XGBoost_predictions = XGBoost_model.predict(X_test)
    #Cohen's kappa
    Cohens_kappa = cohen_kappa_score(y_test, XGBoost_predictions)
    #Classification reportx
    Classification_report = classification_report(y_test, XGBoost_predictions, zero_division = 1)
    print("Learning rate: ", learning_rate)
    print("Cohen's Kappa score:", Cohens_kappa)
    print("XGBoost Classification Report for each class:\n", Classification_report)
    print("==============================")
 
#https://stackoverflow.com/questions/42192227/xgboost-python-classifier-class-weight-option
#https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/#h-what-are-class-weights
#------------------------------------------------------------------------
#Hist Gradient boosting
#-------------------------------------------------------------------------------

for learning_rate in learning_rate_list:
    #classifier
    Hist_clf = HistGradientBoostingClassifier(learning_rate = learning_rate, max_iter = 100)
    #Model fit
    Hist_model = Hist_clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))  
    #Prediction on test data
    Hist_prediction = Hist_model.predict(X_test)
    #Cohens kappa
    Hist_Cohens_kappa = cohen_kappa_score(y_test, Hist_prediction)
    #Classification report
    Hist_Classification_report = classification_report(y_test, Hist_prediction, zero_division = 1)
    print("Learning rate: ", learning_rate)
    print("Cohen's Kappa score:", Hist_Cohens_kappa)
    print("Histgradient Classification Report of each class (all audio features with Doc2vec):\n", Hist_Classification_report)
    print("==============================")

#-------------------------------------------------------------------------------------
#Gradient boosting
#-----------------------------------------------------------------------------------

for learning_rate in learning_rate_list:
    gb_clf = GradientBoostingClassifier(n_estimators=500, learning_rate=learning_rate, max_features=4, max_depth=4, random_state=0)
    gb_model = gb_clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))  

    #Prediction on test data
    gb_prediction = gb_model.predict(X_test)
    #Cohens kappa
    gb_Cohens_kappa = cohen_kappa_score(y_test, gb_prediction)
    #Classification report
    gb_Classification_report = classification_report(y_test, gb_prediction, zero_division = 1)
    print("Learning rate: ", learning_rate)
    print("Cohen's Kappa score:", gb_Cohens_kappa)
    print("Classification Report of each class:\n", gb_Classification_report)
    print("==============================")


#https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/
#------------------------------------------------------
#Lightbgm

from lightgbm import LGBMClassifier

#Instance of class
lightbgm =LGBMClassifier(random_state=0,class_weight = 'balanced',scale_pos_weight=7, learning_rate = 0.05,n_estimators=150, num_leaves = 100)
#Model fit
lbm_model = lightbgm.fit(X_train, y_train)
#Predictions on test data
lbm_prediction = lbm_model.predict(X_test)

#Cohens kappa
lbm_Cohens_kappa = cohen_kappa_score(y_test, lbm_prediction)
#Classification report
lbm_Classification_report = classification_report(y_test, lbm_prediction, zero_division = 1)
print("Cohen's Kappa score:", lbm_Cohens_kappa)
print("Classification Report of each class:\n", lbm_Classification_report)
print("==============================")
#------------------------------------------------------------
#Catboost
#-----------------------------------------------------------------

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

class_weights = len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))

catboost = CatBoostClassifier(iterations=200, learning_rate= 0.5, depth=6, loss_function='MultiClass',class_weights=class_weights)
#Model fit
catboost_model = catboost.fit(X_train, y_train)
#Model prediction on test data
catboost_prediction = catboost_model.predict(X_test)
#Cohens kappa
catboost_Cohens_kappa = cohen_kappa_score(y_test, catboost_prediction)
#Classification report
catboost_Classification_report = classification_report(y_test, catboost_prediction, zero_division = 1)
#print("Learning rate: ", learning_rate)
print("Cohen's Kappa score:", catboost_Cohens_kappa)
print("Classification Report of each class:\n", catboost_Classification_report)
print("==============================")
    

fig, ax = plt.subplots(figsize=(8, 8))
plot_confusion_matrix(XGBoost_model, X_test, y_test, cmap=plt.cm.Blues, ax=ax);
#https://www.kaggle.com/code/kaanboke/xgboost-lightgbm-catboost-imbalanced-data
#-----------------------------------------------------------------
#LDA classiifer
#-------------------------------------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Creating an instance of class
LDA_clf = LinearDiscriminantAnalysis()

#Model fit
LDA_model = LDA_clf.fit(X_train, y_train)
#Predictions on test data
LDA_prediction = LDA_model.predict(X_test)
#Cohens kappa
LDA_Cohens_kappa = cohen_kappa_score(y_test, LDA_prediction)
#Classification report
LDA_Classification_report = classification_report(y_test, LDA_prediction, zero_division = 1)
print("Cohen's Kappa score:", LDA_Cohens_kappa)
print("Classification Report of each class:\n", LDA_Classification_report)
print("==============================")
#-------------------------------------------------



#____________________________________________________________________________
#Audio features with CBOW
#_________________________________________________________________________________
    
#Reading the audio and lyrics data
Audio_features = pd.read_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\Features_RFE.csv',index_col = False)
CBOW = pd.read_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\CBOW.csv',index_col = False)

#Value count of genre column
CBOW['genres'].value_counts()

#Dropping the genre column
CBOW  = CBOW.drop('genres',axis = 1)

#Merging the audio and lyrics columns
df = pd.merge(Audio_features,CBOW, on = "id")

#Applying label encoder to lyrics column (converting text to numbers)
df[['genres']] =  df[['genres']].apply(LabelEncoder().fit_transform)
df['genres'].value_counts()

# Splitting the independent and target variable as 'x' and 'y' 
X = df.drop('genres', axis=1)  
X = X.drop('id', axis=1)
y = df['genres']  

# Splitting the dataset into train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape
y_train.value_counts()

#-------------------------------------------------------------------------
#XGBoost model
#-------------------------------------------------------------------------

#Learning rate of list
learning_rate_list = [0.001, 0.01,0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

#For loop for XGBoost
for learning_rate in learning_rate_list:
    #XGBoost classiifier
    XGBoost_classifier = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=8,
        #reg_lambda=1.0,
        #reg_alpha=0.1,
        max_depth=8,
        learning_rate= learning_rate,
        n_estimators=200,
        subsample=0.5
    )
    #Fitting the training data
    XGBoost_model = XGBoost_classifier.fit(X_train, y_train) #sample_weight=compute_sample_weight("balanced", y_train))  

    #predictions on test data
    XGBoost_predictions = XGBoost_model.predict(X_test)
    #Cohen's kappa
    Cohens_kappa = cohen_kappa_score(y_test, XGBoost_predictions)
    #Classification report
    Classification_report = classification_report(y_test, XGBoost_predictions, zero_division = 1)
    print("Learning rate: ", learning_rate)
    print("Cohen's Kappa score:", Cohens_kappa)
    print("XGBoost Classification Report for each class:\n", Classification_report)
    print("==============================")
 
#https://stackoverflow.com/questions/42192227/xgboost-python-classifier-class-weight-option
#https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/#h-what-are-class-weights
#------------------------------------------------------------------------
#Hist Gradient boosting
#-------------------------------------------------------------------------------

for learning_rate in learning_rate_list:
    #classifier
    Hist_clf = HistGradientBoostingClassifier(learning_rate = learning_rate)
    #Model fit
    Hist_model = Hist_clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))  
    #Prediction on test data
    Hist_prediction = Hist_model.predict(X_test)
    #Cohens kappa
    Hist_Cohens_kappa = cohen_kappa_score(y_test, Hist_prediction)
    #Classification report
    Hist_Classification_report = classification_report(y_test, Hist_prediction, zero_division = 1)
    print("Learning rate: ", learning_rate)
    print("Cohen's Kappa score:", Hist_Cohens_kappa)
    print("Classification Report of each class:\n", Hist_Classification_report)
    print("==============================")

#-------------------------------------------------------------------------------------
#Gradient boosting
#-----------------------------------------------------------------------------------

for learning_rate in learning_rate_list:
    gb_clf = GradientBoostingClassifier(n_estimators=500, learning_rate=learning_rate, max_features=4, max_depth=4, random_state=0)
    gb_model = gb_clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))  

    #Prediction on test data
    gb_prediction = gb_model.predict(X_test)
    #Cohens kappa
    gb_Cohens_kappa = cohen_kappa_score(y_test, gb_prediction)
    #Classification report
    gb_Classification_report = classification_report(y_test, gb_prediction, zero_division = 1)
    print("Learning rate: ", learning_rate)
    print("Cohen's Kappa score:", gb_Cohens_kappa)
    print("Classification Report of each class:\n", gb_Classification_report)
    print("==============================")


#https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/
#------------------------------------------------------
#Lightbgm

from lightgbm import LGBMClassifier

#Instance of class
lightbgm =LGBMClassifier(random_state=0,class_weight = 'balanced',scale_pos_weight=7, learning_rate = 0.05,n_estimators=150, num_leaves = 100)
#Model fit
lbm_model = lightbgm.fit(X_train, y_train)
#Predictions on test data
lbm_prediction = lbm_model.predict(X_test)

#Cohens kappa
lbm_Cohens_kappa = cohen_kappa_score(y_test, lbm_prediction)
#Classification report
lbm_Classification_report = classification_report(y_test, lbm_prediction, zero_division = 1)
print("Cohen's Kappa score:", lbm_Cohens_kappa)
print("Classification Report of each class:\n", lbm_Classification_report)
print("==============================")
#------------------------------------------------------------
#Catboost
#-----------------------------------------------------------------

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

#Catboost classiifer
catboost = CatBoostClassifier(iterations=100,learning_rate=0.5)
#Model fit
catboost_model = catboost.fit(X_train, y_train)
#Model prediction on test data
catboost_prediction = catboost_model.predict(X_test)
#Cohens kappa
catboost_Cohens_kappa = cohen_kappa_score(y_test, catboost_prediction)
#Classification report
catboost_Classification_report = classification_report(y_test, catboost_prediction, zero_division = 1)
print("Learning rate: ", learning_rate)
print("Cohen's Kappa score:", catboost_Cohens_kappa)
print("Classification Report of each class:\n", catboost_Classification_report)
print("==============================")

#https://www.kaggle.com/code/kaanboke/xgboost-lightgbm-catboost-imbalanced-data
#-----------------------------------------------------------------
#LDA classiifer
#----------------------------------------------------------------

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Creating an instance of class
LDA_clf = LinearDiscriminantAnalysis()

#Model fit
LDA_model = LDA_clf.fit(X_train, y_train)
#Predictions on test data
LDA_prediction = LDA_model.predict(X_test)
#Cohens kappa
LDA_Cohens_kappa = cohen_kappa_score(y_test, LDA_prediction)
#Classification report
LDA_Classification_report = classification_report(y_test, LDA_prediction, zero_division = 1)
print("Cohen's Kappa score:", LDA_Cohens_kappa)
print("Classification Report of each class:\n", LDA_Classification_report)
print("==============================")

#__________________________________________________________________________________
#Audio features with Doc2Vec
#________________________________________________________________________________

#Reading the audio and lyrics data
Audio_features = pd.read_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\Features_RFE.csv',index_col = False)
Doc2Vec = pd.read_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\document_embeddings.csv',index_col = False)

#Value count of genre column
Doc2Vec['genres'].value_counts()

#Dropping the genre column
Doc2Vec  = Doc2Vec.drop('genres',axis = 1)

#Merging the audio and lyrics columns
df = pd.merge(Audio_features,Doc2Vec, on = "id")

#Applying label encoder to lyrics column (converting text to numbers)
df[['genres']] =  df[['genres']].apply(LabelEncoder().fit_transform)
df['genres'].value_counts()

# Splitting the independent and target variable as 'x' and 'y' 
X = df.drop('genres', axis=1)  
X = X.drop('id', axis=1)
y = df['genres']  

# Splitting the dataset into train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape
y_train.value_counts()

#-------------------------------------------------------------------------
#XGBoost model
#-------------------------------------------------------------------------

#Learning rate of list
learning_rate_list = [0.001, 0.01,0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

#For loop for XGBoost
for learning_rate in learning_rate_list:
    #XGBoost classiifier
    XGBoost_classifier = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=8,
        #reg_lambda=1.0,
        #reg_alpha=0.1,
        max_depth=8,
        learning_rate= learning_rate,
        n_estimators=200,
        subsample=0.5
    )
    #Fitting the training data
    XGBoost_model = XGBoost_classifier.fit(X_train, y_train) #sample_weight=compute_sample_weight("balanced", y_train))  

    #predictions on test data
    XGBoost_predictions = XGBoost_model.predict(X_test)
    #Cohen's kappa
    Cohens_kappa = cohen_kappa_score(y_test, XGBoost_predictions)
    #Classification report
    Classification_report = classification_report(y_test, XGBoost_predictions, zero_division = 1)
    print("Learning rate: ", learning_rate)
    print("Cohen's Kappa score:", Cohens_kappa)
    print("XGBoost Classification Report for each class:\n", Classification_report)
    print("==============================")
 
#https://stackoverflow.com/questions/42192227/xgboost-python-classifier-class-weight-option
#https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/#h-what-are-class-weights
#------------------------------------------------------------------------
#Hist Gradient boosting
#-------------------------------------------------------------------------------

for learning_rate in learning_rate_list:
    #classifier
    Hist_clf = HistGradientBoostingClassifier(learning_rate = learning_rate)
    #Model fit
    Hist_model = Hist_clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))  
    #Prediction on test data
    Hist_prediction = Hist_model.predict(X_test)
    #Cohens kappa
    Hist_Cohens_kappa = cohen_kappa_score(y_test, Hist_prediction)
    #Classification report
    Hist_Classification_report = classification_report(y_test, Hist_prediction, zero_division = 1)
    print("Learning rate: ", learning_rate)
    print("Cohen's Kappa score:", Hist_Cohens_kappa)
    print("Classification Report of each class:\n", Hist_Classification_report)
    print("==============================")

#-------------------------------------------------------------------------------------
#Gradient boosting
#-----------------------------------------------------------------------------------

for learning_rate in learning_rate_list:
    gb_clf = GradientBoostingClassifier(n_estimators=500, learning_rate=learning_rate, max_features=4, max_depth=4, random_state=0)
    gb_model = gb_clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))  

    #Prediction on test data
    gb_prediction = gb_model.predict(X_test)
    #Cohens kappa
    gb_Cohens_kappa = cohen_kappa_score(y_test, gb_prediction)
    #Classification report
    gb_Classification_report = classification_report(y_test, gb_prediction, zero_division = 1)
    print("Learning rate: ", learning_rate)
    print("Cohen's Kappa score:", gb_Cohens_kappa)
    print("Classification Report of each class:\n", gb_Classification_report)
    print("==============================")


#https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/
#------------------------------------------------------
#Lightbgm

from lightgbm import LGBMClassifier

#Instance of class
lightbgm =LGBMClassifier(random_state=0,class_weight = 'balanced',scale_pos_weight=7, learning_rate = 0.05,n_estimators=150, num_leaves = 100)
#Model fit
lbm_model = lightbgm.fit(X_train, y_train)
#Predictions on test data
lbm_prediction = lbm_model.predict(X_test)

#Cohens kappa
lbm_Cohens_kappa = cohen_kappa_score(y_test, lbm_prediction)
#Classification report
lbm_Classification_report = classification_report(y_test, lbm_prediction, zero_division = 1)
print("Cohen's Kappa score:", lbm_Cohens_kappa)
print("Classification Report of each class:\n", lbm_Classification_report)
print("==============================")
#------------------------------------------------------------
#Catboost
#-----------------------------------------------------------------

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

#Catboost classiifer
catboost = CatBoostClassifier(iterations=100,learning_rate=0.5)
#Model fit
catboost_model = catboost.fit(X_train, y_train)
#Model prediction on test data
catboost_prediction = catboost_model.predict(X_test)
#Cohens kappa
catboost_Cohens_kappa = cohen_kappa_score(y_test, catboost_prediction)
#Classification report
catboost_Classification_report = classification_report(y_test, catboost_prediction, zero_division = 1)
print("Learning rate: ", learning_rate)
print("Cohen's Kappa score:", catboost_Cohens_kappa)
print("Classification Report of each class:\n", catboost_Classification_report)
print("==============================")


#https://www.kaggle.com/code/kaanboke/xgboost-lightgbm-catboost-imbalanced-data
#-----------------------------------------------------------------
#LDA classiifer
#----------------------------------------------------------------

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Creating an instance of class
LDA_clf = LinearDiscriminantAnalysis()

#Model fit
LDA_model = LDA_clf.fit(X_train, y_train)
#Predictions on test data
LDA_prediction = LDA_model.predict(X_test)
#Cohens kappa
LDA_Cohens_kappa = cohen_kappa_score(y_test, LDA_prediction)
#Classification report
LDA_Classification_report = classification_report(y_test, LDA_prediction, zero_division = 1)
print("Cohen's Kappa score:", LDA_Cohens_kappa)
print("Classification Report of each class:\n", LDA_Classification_report)
print("==============================")

#---------------------------------------------------------------------------------------
