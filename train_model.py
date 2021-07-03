import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('t4 - Sheet1.csv')
df['op'] = '0'

df.loc[165:572, 'op'] = 'prc_500_mg'

df.loc[573:1078, 'op'] = 'neurobion'

df.loc[1079:1631, 'op'] = 'dolo_650'

df.loc[1632:2103, 'op'] = 'supradyn'

df.loc[2104:2439, 'op'] = 'crocin'
df.loc[5441:5707, 'op'] = 'crocin'

df.loc[2441:2979, 'op'] = 'combiflame'

df.loc[2980:3247, 'op'] = 'prc_650_mg'
df.loc[5053:5205, 'op'] = 'prc_650_mg'
df.loc[5210:5220, 'op'] = 'prc_650_mg'
df.loc[5261:5440, 'op'] = 'prc_650_mg'

df.loc[3248:3403, 'op'] = 'aspirin_75_mg'
df.loc[3412:3727, 'op'] = 'aspirin_75_mg'

df.loc[3728:4138, 'op'] = 'danp'
df.loc[5708:5758, 'op'] = 'danp'

df.loc[4162:4719, 'op'] = 'ltk_h'

data = df.drop(df[df['op'] == '0'].index)

data.drop(columns=['Date ', 'Time '], axis=1, inplace=True)

tab_map = {
    "prc_500_mg":0,
    "neurobion":1,
    "dolo_650":2,
    "supradyn":3,
    "crocin":4,
    "combiflame":5,
    "prc_650_mg":6,
    "aspirin_75_mg":7,
    "danp":8,
    "ltk_h":9
}

op_map = {value:key for key, value in tab_map.items()}

data['op'] = data['op'].map(tab_map)

# shuffle the data
from sklearn.utils import shuffle
data = shuffle(data)
data.reset_index(inplace=True, drop=True)

# Split data into X and y
X = data.drop(columns=['op'], axis=0)
y = data[['op']]

print(X.shape)
print(y.shape)

# Split data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, plot_confusion_matrix, f1_score, recall_score, classification_report

# Build model pipeline and train it
xgb_pipeline = make_pipeline(StandardScaler(), XGBClassifier(random_state = 18))
xgb_pipeline.fit(X_train, y_train)
# Accuray On Test Data
predictions = xgb_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on Test Data: {accuracy*100}%")
plot_confusion_matrix(xgb_pipeline, X_test, y_test)
plt.title("Confusion Matrix for Test Data")
plt.show()
print(classification_report(y_test, predictions))
print()
# Accuray On Whole Data
predictions = xgb_pipeline.predict(X.values)
accuracy = accuracy_score(y, predictions)
print(f"Accuracy on Whole Data: {accuracy*100}%")
plot_confusion_matrix(xgb_pipeline, X.values, y)
plt.title("Confusion Matrix for Whole Data")
plt.show()
print(classification_report(y, predictions))

import pickle
filename = 'model.pkl'

pickle.dump(xgb_pipeline, open(filename, 'wb'))
