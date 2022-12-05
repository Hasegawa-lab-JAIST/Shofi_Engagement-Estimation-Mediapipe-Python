# Train the dataset and save the trained model.
# Input: dataset.csv
# Output: model.pkl
# ===========================================================================================

import pandas as pd
from pandas.core.algorithms import mode
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score
import pickle

# 01. Read data
df = pd.read_csv('engagement.csv')
x = df.drop('class',axis=1) #features
y = df['class'] #target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1234)

# 02. Train ML classification (create 4 models)
pipeline = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

# print(list(pipeline.values())[0])

fit_models = {}
for algo, pipeline in pipeline.items():
    model = pipeline.fit(x_train, y_train)
    fit_models[algo] = model

# print(fit_models)

# 03. Evaluate and Serialize Model
for algo, model in fit_models.items():
    yhat = model.predict(x_test)
    print(algo, accuracy_score(y_test,yhat))

print(fit_models['rf'].predict(x_test))

# 04. Save model in pickle
with open('engagement_from_js.pkl','wb') as f:
    pickle.dump(fit_models['rf'],f) # only save random forest (rf) model 
