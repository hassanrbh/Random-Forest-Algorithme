import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data , columns = iris.feature_names)
# Split The Data into Target Data :
df["Target"] = iris.target

def transform_(x):
    if x == 0:
        return "setosa"
    if x == 1:
        return "versicolor"
    if x == 2:
        return "virginica"

# Applay This function To Create A New Columns to replace 0 and 1 and 2 :
df["Flowers_names"] = df.Target.apply(lambda x : transform_(x))
# Split The 1 Data Into 3 Data :
df_0 = df[df["Target"] == 0]
df_1 = df[df["Target"] == 1]
df_2 = df[df["Target"] == 2]
# Plot The 3 Data Into Scatter :
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.scatter(df_0["sepal length (cm)"] , df_0["sepal width (cm)"] , color = "green" , marker = "+" )
plt.scatter(df_1["sepal length (cm)"] , df_1["sepal width (cm)"] , color = "red" , marker = "+")
# Plot The 3 Data into Scatter :
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.scatter(df_0["petal length (cm)"] , df_0["petal length (cm)"] , color = "green" , marker = "+" )
plt.scatter(df_1["petal length (cm)"] , df_1["petal width (cm)"] , color = "red" , marker = "+")
# Split The Data Into Train and TesT Data :
from sklearn.model_selection import train_test_split
X = df.drop(["Target" , "Flowers_names"] , axis = "columns")
y = df.Target
X_train , X_test , y_train , y_test = train_test_split(X , y, test_size = 0.2)*
# Build The Model :
from sklearn.svm import SVC
classifier = SVC()
# Fit The Model :
classifier.fit(X_train , y_train)