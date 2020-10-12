import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes


# In[5]:


data, target = load_diabetes(return_X_y=True, as_frame=True)
data['Target'] = target


# **Assign Diabetic or non Diabetic binary values to target variable**

# In[6]:


#Target values < thresholdUnit means no Diabetes and >= thresholdUnits means Diabetes
thresholdUnit = 126

isDiabetic = np.array([])

for i in target.values:
    if i > 126:
        isDiabetic = np.append(isDiabetic, True)
    else:
        isDiabetic = np.append(isDiabetic, False)

#Add the new Diabetic non Diabetic in main dataframe        
data['Diabetic'] = isDiabetic


# In[7]:


no_of_diabetic = np.count_nonzero(data['Diabetic'].values)
no_of_non_diabetic = len(data['Diabetic'].values)-no_of_diabetic
print("Diabetic: ", no_of_diabetic, "\nNon Diabetic: ", no_of_non_diabetic)
data


# **Time for appropriate feature selection**

# In[8]:


fig, axs = plt.subplots(figsize=(15,10))
sns.heatmap(data.corr(), annot=True)


# In[9]:


fig, axs = plt.subplots(nrows = 5, ncols = 2, figsize=(20,30))
colIndex = 0
for i in range(5):
    for j in range(2):
        axs[i][j].scatter(data[data.columns[colIndex]].values, data["Diabetic"])
        axs[i][j].set_xlabel(data.columns[colIndex])
        axs[i][j].set_ylabel("Target")
        colIndex += 1
plt.show()


# In[ ]:





# **Split Dataframe to training and testing dataframes**

# In[49]:


#Split train test dataframe uniformly
diabeticDataFrame = data[data['Diabetic'] == 1]
nonDiabeticDataFrame = data[data["Diabetic"] == 0]
print("Diabetic: ", diabeticDataFrame.shape, "\nNon Diabetic: ", nonDiabeticDataFrame.shape)

no_of_samples = 1
train_test_sample = {}
for i in range(no_of_samples):
    sampledDiabetic = diabeticDataFrame.sample(n=193).reset_index(drop=True).append(nonDiabeticDataFrame)
    sampledDiabetic = sklearn.utils.shuffle(sampledDiabetic)
    train_test_sample[i]= {
        'test':sampledDiabetic[:193//3*2],
        'train':sampledDiabetic[193//3*2:]
                          }
print("Train:Test data ratio = ", train_test_sample[0]['train'].shape[0]/train_test_sample[0]['test'].shape[0])


# In[47]:


logReg = LogisticRegression()

x_train = train_test_sample[0]['train'].iloc[:, :-2]
y_train = train_test_sample[0]['train'].iloc[:, -1]

logReg.fit(x_train,y_train)

#For training data
predictedTrainingData = logReg.predict(x_train)

r2_train = r2_score(y_train, predictedTrainingData)
con_mat_train = confusion_matrix(y_train, predictedTrainingData)
print("R2 Score: ", r2_train, "\nConfusion Matrix: \n", pd.DataFrame(data=con_mat_train).rename(columns={0:'True', 1:'False'}, index = {0:'True', 1:'False'}))


# In[48]:


x_test = train_test_sample[0]['test'].iloc[:, :-2]
y_test = train_test_sample[0]['test'].iloc[:, -1]
predicted = logReg.predict(x_test)
r2_test = r2_score(y_test, predicted)
con_mat_test = confusion_matrix(y_test, predicted)
print("R2 Score: ", r2_test, "\nConfusion Matrix: \n", pd.DataFrame(data=con_mat_test).rename(columns={0:'True', 1:'False'}, index = {0:'True', 1:'False'}))

