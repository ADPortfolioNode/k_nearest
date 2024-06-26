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

print('data loaded')
print(df.head())

#Data Visualization and Analysis
#how many of each class is in our data set
df['custcat'].value_counts()

#### 281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers

df.hist(column='income', bins=50)

#feature set (x)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  
#.astype(float)
print("loaded features: ", X[0:5])

#labels (y)
y = df['custcat'].values
print("loaded labels: ", y[0:5])
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
print('k=4 trained model   :',neigh)

#PREDICTING 
yhat = neigh.predict(X_test)
print('k=4 models predicted values:',yhat[0:5])


#Accuracy Evaluation
print('Accuracy Evaluation')
print("Train set k=4  neighbor prediction  Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set k=4 neighbor Accuracy: ", metrics.accuracy_score(y_test, yhat))


print("//////PRACTICE/////")
print("//////PRACTICE/////")
print('training with k=5 now')
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
print ('k=5 Train set:', X_train.shape,  y_train.shape)
k = 5
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print('k=5 trained model: ',neigh)
#PREDICTING 
yhat = neigh.predict(X_test)
print('k=5 models predicted values:',yhat[0:5]) 
plt.scatter(y_test, yhat)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.title('Actual vs Predicted Values')
plt.show()


#Accuracy Evaluation
print("Train set k=5  neighbor predicted Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set k=5 neighbor test  Accuracy: ", metrics.accuracy_score(y_test, yhat))

print('!!!!!!!!!!!!!!k=5 now tuning!!!!!!!!!!')

# Split your data into features (X) and target (y) 
X = df.drop(['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside'], axis=1)  # replace 'target_column' with your actual target column
y = df['custcat']
print('X:',X.head())
print('y:',y.head()) 
# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train set:',X_train.shape, y_train.shape)


print('KNN model>>>>>>>>>>>')
# Instantiate the KNN model
neigh = KNeighborsClassifier(n_neighbors=1)
print('instantiated model')

neigh.fit(X_train, y_train)
print('fitted model',neigh)
# Predict
yhat = neigh.predict(X_test)
print('predicted values:',yhat[0:5])

# Accuracy Evaluation
print("Train set suggested (1) neighbor  Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set suggested (1) Accuracy: ", metrics.accuracy_score(y_test, yhat))
plt.scatter(y_test, yhat)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values') 
plt.title('Actual vs Predicted Values')
plt.show()

print('NOW TUNING>>>>>>>>>>>>>>>>')
# Now tuning
param_grid = {'n_neighbors': list(range(1, 31))}

grid = GridSearchCV(neigh, param_grid, cv=10, scoring='accuracy')
print('grid:',grid)
grid.fit(X_train, y_train)
print('fitted grid:',grid)

print('tuned:')
# Print the best parameters and the corresponding score
print("Best parameters: ", grid.best_params_)
print("Best cross-validation score possible: ", grid.best_score_)