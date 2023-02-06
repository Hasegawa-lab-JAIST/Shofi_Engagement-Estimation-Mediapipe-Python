# Train the dataset and save the trained model.
# Input: dataset.csv
# Output: model.pkl
# ===========================================================================================

import pandas as pd
import numpy as np
from datetime import datetime
# from pandas.core.algorithms import mode
# from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score
import pickle

#01. Load Data
def load_data(datafile):
    df = pd.read_csv(datafile)
    df = df.fillna(0)
    df = df.drop(df.columns[0], axis=1)
    return df

#Split features and label
def split_label(df):
    df = np.array(df)
    Y = df[:,-1]
    # print('Y original =', Y)
    Y = np.where(Y==1, 0, Y) #combine label 0 and 1
    Y = np.where(Y==2, 1, Y) #change label 2 to 1
    Y = np.where(Y==3, 2, Y) #change label 3 to 2
    # print('Y before encode =', Y)
    Y = LabelEncoder().fit_transform(Y) #encode label value as label variable
    # print('Y after encode =', Y)
    X = df[:,0:-1]    
    return X, Y

#Concatenate train and validation set
path = '../../Shofi_Engagement-Model-LSTM/'
df_train_av = load_data(path+'dataset/average_train_mediapipe_noDropna.csv')
df_val_av = load_data(path+'dataset/average_val_mediapipe_noDropna.csv')
df_test_av = load_data(path+'dataset/average_test_mediapipe_noDropna.csv')
print('Train = ', df_train_av.shape)
print('Val = ', df_val_av.shape)
print('Test = ', df_test_av.shape)
df_train_concat = pd.concat([df_train_av,df_val_av], axis=0, ignore_index=True)
print(df_train_concat.shape)

# Split feature
X_train, Y_train = split_label(df_train_concat)
X_test, Y_test = split_label(df_test_av)

# 02. Train ML classification (create 4 models)
pipeline = {
    'lr': make_pipeline(MinMaxScaler(), LogisticRegression()),
    'rc': make_pipeline(MinMaxScaler(), RidgeClassifier()),
    'rf': make_pipeline(MinMaxScaler(), RandomForestClassifier()),
    'gb': make_pipeline(MinMaxScaler(), GradientBoostingClassifier())
}

# print(list(pipeline.values())[0])

fit_models = {}
for algo, pipeline in pipeline.items():
    tic = datetime.now()
    model = pipeline.fit(X_train, Y_train)
    time = datetime.now()-tic
    time = str(time)
    print('Training time {} = {}'.format(algo,time))
    fit_models[algo] = model

print(fit_models)

# 03. Evaluate and Serialize Model
for algo, model in fit_models.items():
    tic = datetime.now()
    yhat = model.predict(X_test)
    time = datetime.now()-tic
    print(algo, accuracy_score(Y_test,yhat))
    print('Evaluation time {} = {}\n'.format(algo,time))
    
# print(fit_models['lr'].predict(X_test))

# 04. Save model in pickle
with open('engagement_DAiSEE.pkl','wb') as f:
    pickle.dump(fit_models['rc'],f) # only save random forest (rf) model 
