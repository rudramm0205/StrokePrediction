#!/usr/bin/env python
# coding: utf-8

# ## <h1><center> MA336: Artificial intelligence and machine learning with applications</center></h1>
# ## <h1><center><font color='Purple'>STROKE PREDICTION</font></center></h1> 
# # <h1><center>Registration ID: 2111901</center></h1>

# ## <font color='green'>1.Introduction:</font>
# Machine Learning(ML) is a branch of computer science that is simply based on training a computer by leveraging a set of data to improve the performance of some tasks.Machine Learning algorithms are used to train the computer, based on that training data or algorithm provided it should perform the tasks given to it without being explicitily programmed to do so.By implementing various Machine Learning algorithms, we have tried to predict the factors which are main causes of stroke in an individual from the given sample dataset. In the following dataset we have features like gender,age, hypertension,heart_disease, smoking_status which can help us identify the individual with what kind of features is more likely to receive a stroke. I have decided to go for Stroke prediction as stroke in individuals has been an alarming issue in the world and according to WHO data, annually, 15 million people suffer from a stroke out of which 5 million die from stroke and 5 million are left disabled which puts a drastic burden on the family. With the help of this project I would like to point out the factors responsible for stroke which will help to identify the underlying problem in the Healthcare Sector and providing insights on it using Machine Learning algorithms like Logistic Regression, Random Forest Classifiers (RFC), Decision Tree Classifiers(CART).

# # <h1><center>Data Loading and Pre-Processing</center></h1>

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# In[2]:


##To read the heart stroke data.
new_data = pd.read_csv("healthcare-dataset-stroke-data (1).csv")


# In[3]:


## To see the first five observations of the dataframe.
new_data.head(10)


# In[4]:


## To see the total number of rows and columns.
new_data.shape


# In[5]:


## The function info() represents the data types of every variable.
new_data.info()


# In[6]:


## To check the null values present in our data frame.
new_data.isnull().sum()


# ## Data Cleaning

# In[7]:


## Imputation of null values that is present in 'bmi' with median.
new_data['bmi']=new_data['bmi'].fillna(new_data['bmi'].median())


# In[8]:


## To check again whether our above method removed the null values or not.
new_data.isnull().sum()


# In[9]:


## To present the statistical summary of the dataframe.
new_data.describe()  


# # <h1><center>Exploratory data analysis (EDA)</center></h1>

# In[10]:


## BMI Boxplot to identify Outliers
new_data['bmi'].plot.box()


# In[11]:


## Average Glucose Level Boxplot to show Outliers
new_data['avg_glucose_level'].plot.box()


# In[12]:


## To check the frequency of the 'age' with the help of histogram.
new_data['age'].plot.hist()


# In[13]:


## To count different levels of smoking status and plotting a bar chart.
new_data['smoking_status'].value_counts().plot.bar()


# In[14]:


## To see the rate of stroke amoung different age groupes.
sns.boxplot(data=new_data,y=new_data['age'],x=new_data['stroke'])


# In[15]:


## To check whether glucose level has any affect on stroke
sns.boxplot(data=new_data,y=new_data['avg_glucose_level'],x=new_data['stroke'])


# In[16]:


# Cross-Tabulation of Gender and Stroke variables for plotting 2 categorical variables
gender_stroke=pd.crosstab(new_data['gender'],new_data['stroke'])
gender_stroke


# In[17]:


# Gender-Stroke Plot
gender_stroke.plot(kind='bar',figsize=(8,8))


# In[18]:


worktype_stroke=pd.crosstab(new_data['work_type'],new_data['stroke'])
worktype_stroke


# In[19]:


worktype_stroke.plot(kind='bar',figsize=(8,8), stacked=True)


# In[20]:


residence_stroke=pd.crosstab(new_data['Residence_type'],new_data['stroke'])
residence_stroke


# In[21]:


residence_stroke.plot(kind='bar',figsize=(8,8))


# In[ ]:





# In[22]:


sns.distplot(new_data['bmi'])


# In[23]:


sns.boxplot(data=new_data,y=new_data['bmi'],x=new_data['stroke'])


# In[24]:


sns.distplot(new_data['avg_glucose_level'])


# In[25]:


sns.countplot(new_data['stroke'],hue=new_data['heart_disease'])


# In[26]:


sns.countplot(new_data['stroke'],hue=new_data['smoking_status'])


# In[27]:


sns.distplot(new_data['age'][new_data['heart_disease']==1], kde=True, color='blue', label='heart disease')
sns.distplot(new_data['age'][new_data['heart_disease']==0], kde=True, color='green', label='no heart disease')
plot.legend()


# In[28]:


## To count the total number of males and females in our data frame.
new_data["gender"].value_counts()


# In[29]:


## To count the total number of married and unmarried people in our data frame.
new_data['ever_married'].value_counts()


# In[30]:


## To count the different types of work present in our data frame.
new_data['work_type'].value_counts()


# In[31]:


## To count the types of residence found in our data frame.
new_data['Residence_type'].value_counts()


# In[32]:


## To check the correlation between all the continuous variables present in our data frame.
new_data.corr()


# In[33]:


## To show the columns names present in our data frame.
new_data.columns


# In[34]:


## Used Label Encoder for converting the categorical variables into integer type.
new_col = ["gender", "ever_married" ,"Residence_type","smoking_status","work_type",]
new_encoder = preprocessing.LabelEncoder()
for col in new_col:
    new_data[col]=  new_encoder.fit_transform(new_data[col])


# In[35]:


## To check whether it has changed the variables properly or not...
new_data.head()


# In[36]:


## This correlation heatmap takes all varibles into account and it shows that which factor effects the stroke the MOST.
c, axes = plot.subplots(figsize = (12,10))
sns.heatmap(new_data.corr(), annot=True, ax=axes)


# # <h1><center><font color='green'>2.Methods</font></center></h1>
# • In this Dataframe, we have total 12 Variables and 5 Categorical Variables.
# <br>
# • We have used Label Encoder to convert these 5 categorical Variables into Integer type.
# <br>
# • In BMI variable, we found there are some Missing(NA) Values, we Imputed the variable with Median.
# <br>
# • We Used Standard Scaler in SkLearn Package to remove outliers and to normalise the data.
# <br>
# • We have Splitted the data into 60:40 Train Test Ratio.
# <br>
# • Performed Oversampling using SMOTE function to Balance our Dataset which was rather imbalanced.
# <br>
# 
# 
# ## <h1><center>2.1. Machine learning Algorithms used:</center></h1>
# 1. Logistic Regression - We chose this method because our predictor variable stroke is a bi-variate variable which has values stroke(1) or no stroke(0).We are also going to compare it with other Models based on the accurancy, F1 accuracy score [F1 = 2 / (1/Precision + 1/Recall)] and Feature Importance of the Logistic Regression Model.
# 2. Random Forest Classifier - As it is a classification problem, Random Forest Classifier will be able to provide good accuracy and it helps to detect a class which is more infrequent then other classes.  
# 3. Decision Tree Classifier -  This Classifier Algorithm is used to detect the most important features and also the relations between them. A Classification tree is used as a target to classify whether the patient has a stroke or no stroke.
# 
# 
# 
# 

# In[37]:


## To drop the predictor variable 'stroke' from the independent variables for model training. 
x=new_data.drop(['stroke'],axis=1)
y=new_data['stroke']


# In[38]:


x.shape


# In[39]:


y.shape


# In[40]:


## Performing Oversampling to Balance the imbalanced dataset
oversampling = SMOTE(random_state=123)
x_oversampling , y_oversampling = oversampling.fit_resample(x,y)

print(f'''Before SMOTE:{x.shape}
After SMOTE:{x_oversampling.shape}''',"\n")

print(f'''Distribution before SMOTE:\n{y.value_counts(normalize=True)}
Distribution after SMOTE :\n{y_oversampling.value_counts(normalize=True)}''')


# In[41]:


## To split the data into train and test for model building. We seperated into 60:40 ratio.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_oversampling,y_oversampling,test_size=0.4,random_state=0)


# In[42]:


## To represent the number of observations after splitting.
x_train.shape,x_test.shape


# In[43]:


## To count the total number of strokes present in our data frame.
new_data['stroke'].value_counts()


# In[44]:


## Standard scaler is used for normalization of our training and testing data.
sca=StandardScaler()
x_train=sca.fit_transform(x_train)
x_test=sca.fit_transform(x_test)


# In[45]:


## For importing Logistic regression and its following metrices...
from sklearn.linear_model import LogisticRegression as LgRg
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,precision_score,recall_score


# In[46]:


## To fit the Logistic Regression model into the training set and then predict on the testing set. 
lg= LgRg()
lg.fit(x_train,y_train)
y_lg_pred = lg.predict(x_test)
score_lg=accuracy_score(y_test,y_lg_pred)*100
print("training accuracy score: ",accuracy_score(y_train,lg.predict(x_train))*100)
print("testing accuracy score: ",score_lg)
print("F1 score", f1_score(y_train,lg.predict(x_train)))


# In[47]:


## Predicting Feature Importance of Logistic Regression Model
imp = lg.coef_[0]


# In[48]:


for i,v in enumerate(imp):
	print('Feature: %0d, Score: %.5f' % (i,v))


# In[49]:


# Plotting Feature Importance of Each Variable
plot.bar([x for x in range(len(imp))], imp)
plot.show()


# In[50]:


# Performing Random Forest Classifier on Training and Test Data Set
from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(random_state=200)
rfc = rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
ac = accuracy_score(y_test, y_pred_rfc)
print('Testing Accuracy score is:', ac)
print('Training Accuracy score is:',accuracy_score(y_train,rfc.predict(x_train)))
cm = confusion_matrix(y_test, y_pred_rfc)
sns.heatmap(cm, annot = True, fmt = "d")


# In[51]:


## Predicting Feature Importance of Random Forest Classifier
imp2 = rfc.feature_importances_


# In[52]:


for i,v in enumerate(imp2):
	print('Feature: %0d, Score: %.5f' % (i,v))


# In[53]:


## Plotting Feature importance of Random Forest Classifier
plot.bar([x for x in range(len(imp2))], imp2)
plot.show()


# In[54]:


## Performing Decision Tree Classifier with Random State = 20
from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier(random_state=20)


# In[55]:


## Fitting the Decision Tree Classifier into our training dataframe
dt_model.fit(x_train,y_train)


# In[56]:


## Predicting the score of our Decision Tree Classifier on Training DataFrame
dt_model.score(x_train,y_train)


# In[57]:


## Predicting the score of our Decision tree Classifier on Test DataFrame
dt_model.score(x_test,y_test)


# In[58]:


dt_model.predict(x_train)


# In[59]:


dt_model.predict_proba(x_test)


# In[60]:


y_pred=dt_model.predict_proba(x_test)[:,1]


# In[61]:


y_new=[]
for i in range(len(y_pred)):
    if y_pred[i]<=0.7:
        y_new.append(0)
    else:
        y_new.append(1)


# In[62]:


accuracy_score(y_test,y_new)


# In[63]:


train_accuracy=[]
test_accuracy=[]
for depth in range(1,10):
    dt_model=DecisionTreeClassifier(max_depth=depth,random_state=20)
    dt_model.fit(x_train,y_train)
    train_accuracy.append(dt_model.score(x_train,y_train))
    test_accuracy.append(dt_model.score(x_test,y_test))


# In[64]:


frame=pd.DataFrame({'max_depth':range(1,10),'train_acc':train_accuracy,'test_cc':test_accuracy})
frame.head()


# In[65]:


## Finding the Depth Of Tree to optimize the Decision Tree Classifier 
plot.figure(figsize=(12,6))
plot.plot(frame['max_depth'],frame['train_acc'],marker='o')
plot.plot(frame['max_depth'],frame['test_cc'],marker='o')
plot.xlabel('Depth of a tree')
plot.ylabel('performance')
plot.legend()


# In[66]:


## Predicting Feature Importance of Decision Tree Classifier
imp3 = dt_model.feature_importances_


# In[67]:


for i,v in enumerate(imp3):
	print('Feature: %0d, Score: %.5f' % (i,v))


# In[68]:


## Plotting Feature Importance of Decision Tree Classifier
plot.bar([x for x in range(len(imp3))], imp3)
plot.show()


# # <h1><center><font color='green'>3. Results</font></center></h1>
# 
# ## <h1><center>3.1 Exploratory Data Analysis(EDA) Results</center></h1>
# ‣ More Observations are seen between the age of 45 and 60.
# <br>
# ‣ Maximum Number of people in our Dataset are non-smokers followed by Unknown smoking status, formerly smoked, Smokes Respectively.
# <br>
# ‣ Observations with Age>60  have the highest possibility of having a stroke.
# <br>
# ‣ 80% of Observations in the Data who have stroke have Average Glucose Levels over 115 which is over the threshold Glucose level.
# <br>
# ‣ Number of Strokes in Male and Female are almost Identical.
# <br>
# ‣ According to the stacked Bar Plot, observations having a private job are having more stroke cases then other types of jobs or those who haven’t worked.
# <br>
# ‣ Residence Type (Urban or Rural) of observations has no effect on Stroke.
# <br>
# ‣ Number of Smokers and Non Smokers in theDataset has no Drastic Effect on our number of Stroke observation.
# <br>
# ‣ Most of Patients having a stroke are in the Age between 60-80 and having a Higher Glucose Level.
# 
# 
# ## <h1><center>3.2 Machine Learning Results</center></h1>
# 
# ## 3.2.1 Logistic Regression Results
# ‣ Our training and testing Accuracy Score is 82.22% and 82.18% respectively is almost similar so there is no overfitting in this model.
# <br>
# ‣ We Calculate the F1 Score with the formula which is close to 1 it means it has compared our 2 Classifiers (Stroke or no stroke) accurately.
# <br>
# ‣ In Feature importance bar plot, age is the most Important Feature according to Logistic Regression Model.
# <br>
# 
# ## 3.2.2 Random Forest Classifier Results
# ‣ Testing Accuracy score is around 82% which means it has measured correctly the mean of the subsets.
# <br>
# ‣ In the confusion matrix, We have the Highest true negative value which is a good sign for detection of a stroke in a  healthcare dataset as we have predicted Actual Value and Prediction outcome are more accurate.
# <br>
# ‣ In Feature importance bar plot, age,work_type, glucose Levels and bmi is the most Important feature according to Random Forest Classifier Model.
# <br>
# 
# ## 3.2.3 Decision Tree Classifier Results
# ‣ We have chosen the Hyper-parameter which is Random State = 20 that is why we will get different train and test sets with different integer values and we received an accuracy score of 87%.
# <br>
# ‣ In the Decision Tree Classifier, we have found that the depth of the Tree will be 4.
# <br>
# ‣In Feature importance bar plot, age is shown is the highest and most important feature according to Decision Tree Classifier.
# 
#  
# 

# # <font color='green'>4. Conclusion</font>
# 
# As we can analyse from the Results, Random Forest Classifier algorithm is the best fitted model for this dataset, so we can say that patients having older age, High Glucose Level and High bmi values have the highest probability of getting a Stroke. The dataset was imbalanced and to balance the dataset we had to use the SMOTE technique which equally splits the predictor variable into 2 halves.As we have seen from this data, older-aged patients with high glucose levels and high bmi have high probability of getting a stroke. There is no plausible evidence that patients having a heart disease will have a high probability of having a Stroke.There is also some evidence that work lifestyle also has a high probability when it comes to stroke in a patient. 
# 
# 
# 

# # <font color='green'>5. References</font>
# [https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
# <br>
# [https://en.wikipedia.org/wiki/Machine_learning](https://en.wikipedia.org/wiki/Machine_learning)
