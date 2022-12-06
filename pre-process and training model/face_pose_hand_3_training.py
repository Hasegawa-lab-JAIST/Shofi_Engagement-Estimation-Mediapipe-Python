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
df = pd.read_csv('engagement_from_js.csv')
df = df.fillna(0)
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

# Current result with engagement_from_js.csv
# lr 1.0
# rc 0.9975550122249389
# rf 0.9951100244498777
# gb 0.9975550122249389

print(fit_models['lr'].predict(x_test))

# 04. Save model in pickle
with open('engagement_from_js.pkl','wb') as f:
    pickle.dump(fit_models['lr'],f) # only save random forest (rf) model 

# # 04. Save model in json
# with open('engagement.json','wb') as json_file:
#     json_file.write(fit_models['rf']) # only save random forest (rf) model 