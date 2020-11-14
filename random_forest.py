import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
# Setting The Random Seed :
np.random.seed()
# Load The Data :
iris = load_iris()
df = pd.DataFrame(iris.data , columns = iris.feature_names)
# Implemention :
df["Species"] = pd.Categorical.from_codes(iris.target,iris.target_names)
# Create Test And Train Data :
df["Is Train"] = np.random.uniform(0 , 1 , len(df)) <= .75
# Create A Data Frame With Test Rows and Train Rows :
train = df[df["Is Train"] == True]
test = df[df["Is Train"] == False]
print("Train Data : " , len(train))
print("Test Data : " , len(test))
# Create A list Of Features :
features = df.columns[:4]
# Convert eash Spicies into name into digits :
y = pd.factorize(train["Species"])[0]
# Create A Random Forest Classifier :
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs = 2 , random_state = 0)
# Training The Model :
rfc.fit(train[features] , y)
# Applay The Trained Classifier To The test :
rfc.predict(test[features])
# View The Predict probalitiiy of The First 10 Observation
rfc.predict_proba(test[features])[0:10]
# Mapping Names for The Plants for eash predicted Class :
pred = iris.target_names[rfc.predict(test[features])]
# View The Actual species for the first five observatrion :
test["Species"].head()
# Creating Confusioion Matrix :
pd.crosstab(test["Species"] , pred , rownames= ["Actual Species"], colnames= ["Predicted Species"])