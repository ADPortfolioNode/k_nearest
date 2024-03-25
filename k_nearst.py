import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


#load data
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()

#Data Visualization and Analysis
#how many of each class is in our data set
df['custcat'].value_counts()

#### 281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers

df.hist(column='income', bins=50)

#feature set (x)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  
#.astype(float)
X[0:5]

#labels (y)
y = df['custcat'].values
y[0:5]

#NORMALIZE DATA##
#train / test split training set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print('~~~~~~~test shape~~~~~~~~')
print ('Train  fullset:', X_train.shape,  y_train.shape)
print ('Test  fullset:', X_test.shape,  y_test.shape)

#CLASSIFICATION
#training
#start the algorithm with k=4 for now:
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

#PREDICTING 
yhat = neigh.predict(X_test)
yhat[0:5]


#Accuracy Evaluation
print("Train set 4 neighbor prediction  Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set 4 neighbor Accuracy: ", metrics.accuracy_score(y_test, yhat))


print("//////PRACTICE/////")
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
k = 5
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

#PREDICTING 
yhat = neigh.predict(X_test)
yhat[0:5]


#Accuracy Evaluation
print("Train set :5 neighbor predicted Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set :5 neighbor test  Accuracy: ", metrics.accuracy_score(y_test, yhat))

print('!!!!!!!!!!!!!!now tuning!!!!!!!!!!')

# Split your data into features (X) and target (y) 
X = df.drop(['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside'], axis=1)  # replace 'target_column' with your actual target column
y = df['custcat']

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the KNN model
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)

# Predict
yhat = neigh.predict(X_test)

# Accuracy Evaluation
print("Train set suggested (1) neighbor  Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set suggested (1) Accuracy: ", metrics.accuracy_score(y_test, yhat))

# Now tuning
param_grid = {'n_neighbors': list(range(1, 31))}
grid = GridSearchCV(neigh, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train, y_train)

print('tuned: ',X)
# Print the best parameters and the corresponding score
print("Best parameters: ", grid.best_params_)
print("Best cross-validation score possible: ", grid.best_score_)