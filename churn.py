#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[36]:


df=pd.read_csv("C:/Users/rama/OneDrive/Desktop/data analytics raw data sets/churn prediction/churn_data_raw.csv")


# In[37]:


df


# ## info getting and cleaning

# In[38]:


df.info()


# In[39]:


df.isnull().sum()


# In[40]:


df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')


# In[41]:


df.info()


# In[42]:


df['SeniorCitizen']=df['SeniorCitizen'].map({1:'Yes',0:'No'})


# In[104]:


cleaned_data=df
cleaned_data=cleaned_data.to_csv("C:/Users/rama/OneDrive/Desktop/data analytics raw data sets/churn prediction/cleaned_data.csv",index=False)


# In[43]:


df.info()


# In[44]:


df.nunique()


# In[45]:


df['MultipleLines']


# ## EDA

# In[46]:


sns.countplot(x='Churn',data=df)
plt.title("churn distribution")
plt.xlabel("churn ('no'=stayed,yes=left")
plt.ylabel("customer count")

plt.show()

churn_rate=df['Churn'].value_counts(normalize=True)*100
churn_rate


# In[47]:


## churn by categorial features
features='Contract'
features


# In[48]:


df[features]


# In[49]:


feature='Contract'
sns.countplot(x=feature,hue='Churn',data=df)
plt.title('churn by (feature)')
plt.xticks(rotation=30)
plt.show()


# In[50]:


df


# In[51]:


sns.countplot(x='PaymentMethod',hue='Churn',data=df)
plt.title('churn by (feature)')
plt.xticks(rotation=30)
plt.show()
pd.crosstab(df['PaymentMethod'],df['Churn'],normalize=True)*100


# In[ ]:





# In[52]:


sns.countplot(x="gender", hue="Churn",data=df)
plt.title("churn bsed on gender")
plt.show()

pd.crosstab(df['gender'],df['Churn'])


# In[53]:


sns.histplot(data=df,x='tenure',hue='Churn',bins=30,kde=True)
plt.title("Churn by tenure")
plt.show()


# In[54]:


sns.boxplot(x='Churn',y='MonthlyCharges',data=df)
plt.title("monthly charges  vs Churn")
plt.show()


# In[ ]:





# In[55]:


sns.boxplot(x='Churn',y='TotalCharges',data=df)
plt.title('total charges vs churn')
plt.show()


# In[ ]:





# In[56]:


sns.countplot(x="SeniorCitizen", hue='Churn', data=df)
plt.title("Churn based on Senior Citizen status")
plt.show()


# In[57]:


sns.countplot(hue='Churn',x='InternetService',data =df)
plt.show()


# In[58]:


cols_to_clean=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','MultipleLines']


# In[59]:


for col in cols_to_clean:
    df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})


# In[60]:


df


# ## encode cotegorical columns

# In[61]:


df_model=df.drop(['customerID'],axis=1)


# In[62]:


df_model


# ##  converting all the values to  boolean 

# In[63]:


df_encoded=pd.get_dummies(df_model,drop_first=True)


# In[64]:


df_encoded


# In[65]:


df_encoded.dtypes


# ## converting only bollean values to 0 and 1

# In[67]:


bool_cols=df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols]=df_encoded[bool_cols].astype(int)


# In[106]:


df_encoded.to_csv("C:/Users/rama/OneDrive/Desktop/data analytics raw data sets/churn prediction/encoded.csv")


# In[69]:


df_encoded.isnull().sum()


# In[70]:


df['TotalCharges'].unique()


# In[71]:


df['TotalCharges'].value_counts()


# In[72]:


df.describe()


# In[73]:


df_encoded.describe()


# In[77]:


df_encoded['TotalCharges'].isnull().sum()


# In[76]:


df_encoded['TotalCharges']=df['TotalCharges'].fillna(2280)


# ## deviding Train and Test (Split data)

# In[78]:


from sklearn.model_selection import train_test_split

x = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)


# In[79]:


df_encoded.columns


# In[80]:


df_encoded.count()


# In[81]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=5000)
model.fit(x_train,y_train)


# In[82]:


from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

y_pred=model.predict(x_test)
y_proba=model.predict_proba(x_test)[:,1]

print("Classification Report:")
print(classification_report(y_test,y_pred))

print("\n confusion matrix:")
print(confusion_matrix(y_test,y_pred))


print("\n ROC AUC Score")
print(roc_auc_score(y_test,y_proba))


# In[83]:


from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)


rf_model.fit(x_train, y_train)


# In[84]:


from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
rf_pred=rf_model.predict(x_test)
rf_proba=rf_model.predict_proba(x_test)[:,1]

print("Classification report (Random Forest):")
print(classification_report(y_test,rf_pred))

print("Confusion MAtrix:")
print(confusion_matrix(y_test,rf_pred))


print("ROC AUC score:")
print(roc_auc_score(y_test,rf_proba))


# In[85]:


import pandas as pd 
import matplotlib.pyplot as plt

importances=rf_model.feature_importances_
features=pd.Series(importances,index=x_train.columns)

features.sort_values(ascending=False).head(10).plot(kind='barh',figsize=(9,4))
plt.title("Top 10 important Features (Random Forest)")
plt.gca().invert_yaxis()
plt.show()


# In[83]:


pip install shap


# In[91]:


print("x_testfixed shape:",x_test_fixed.shape)
print("shap_values[1] shape:",np.array(shap_values[1]).shape)


# In[94]:


import shap
import matplotlib.pyplot as plt
import pandas as pd


x_test_fixed = x_test.reindex(columns=x_train.columns, fill_value=0)


explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(x_test_fixed)


print("x_test_fixed shape:", x_test_fixed.shape)
print("shap_values[1] shape:", np.array(shap_values[1]).shape)


if np.array(shap_values[1]).shape == x_test_fixed.shape:
    shap.summary_plot(shap_values[1], x_test_fixed)
else:
    print("Shape mismatch. Try this alternative approach 👇")


# In[ ]:





# In[ ]:





# In[96]:


print(shap_values.shape)


# In[101]:


shap_values_class1=shap_values[...,1]

shap.plots.beeswarm(shap_values_class1,max_display=9)


# In[ ]:




