#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('E:\Certification_Courses\Simplilearn\ML\Final_assignment\HR_comma_sep.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.drop_duplicates(inplace = True)


# In[6]:


df.left.value_counts()


# In[7]:


df.info()


# ## EDA

# In[8]:


import matplotlib.pyplot as plt

cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']

for i in cols:
    plt.figure(figsize=(6, 4)) 

    counts, bins, patches = plt.hist(df[i], bins=5, edgecolor='black', color='skyblue')

    for patch, count in zip(patches, counts):
        plt.text(patch.get_x() + patch.get_width() / 2, count, str(int(count)),
                 ha='center', va='bottom', fontsize=12, color='black')

    plt.xlabel(i)
    plt.ylabel('Count')
    plt.title(f'Histogram of {i}')

    plt.show()  


# In[9]:


col_names = ['salary','Work_accident','promotion_last_5years', 'sales','number_project']

for i in col_names:
    plt.figure(figsize=(10,6))
    ax = sns.countplot(x=i, hue='left', data=df, palette='Set2')

    for container in ax.containers:
        ax.bar_label(container, fmt='%g', label_type='edge')
        
    plt.title(f'Count plot of {i}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ## Percentage of Employees left
# 
# 

# In[10]:


col_names = ['salary','Work_accident','promotion_last_5years', 'sales','number_project']

for i in col_names:
    grouped_data = df.groupby(i)['left'].apply(lambda x: x.sum()/x.count()*100).reset_index()
    
    plt.figure(figsize=(10,6))
    ax = sns.barplot(x=i, y='left', data=grouped_data, palette='Set2')

    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge')
        
    plt.title(f'% Employees left across {i} segment')
    plt.ylabel('% Employees left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# More turnover is observed among employees with very low number of projects (2) and among those with very high project count(7).

# ### Boxplot

# In[11]:


col_names = ['satisfaction_level','last_evaluation','average_montly_hours','time_spend_company']

for i in col_names:
    plt.figure(figsize=(5,2))
    ax = sns.boxplot(x=df[i], color='purple')
    plt.title(f'Box plot of {i}')


# #### Outlier Treatment

# In[12]:


def treat_outlier_iqr(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR =  Q3-Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    
    df[column] = np.where(df[column]<lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column]>upper_bound, upper_bound, df[column])


# In[13]:


for col in ['satisfaction_level','last_evaluation','average_montly_hours','time_spend_company']:
    treat_outlier_iqr(col)


# In[14]:


col_names = ['satisfaction_level','last_evaluation','average_montly_hours','time_spend_company']

for i in col_names:
    plt.figure(figsize=(5,2))
    ax = sns.boxplot(x=df[i], color='purple')
    plt.title(f'Box plot of {i}')


# ## Correlation

# In[15]:


correlation_matrix = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident',   'left','promotion_last_5years']].corr()

plt.figure(figsize = (12,8))
sns.heatmap(correlation_matrix, annot = True)


# ## Clustering of Employees

# Perform clustering of employees who left based on their satisfaction and evaluation.
# 
# 1. Choose columns satisfaction_level, last_evaluation, and left.
# 2. Do K-means clustering of employees who left the company into 3 clusters?
# 3. Based on the satisfaction and evaluation factors, give your thoughts on the employee clusters.

# In[16]:


df_left = df[df['left'] == 1]
X = df_left[['satisfaction_level','last_evaluation']]


# In[17]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,random_state=42)
df_left['cluster'] = kmeans.fit_predict(X)


# In[18]:


plt.figure(figsize=(8,5))
sns.scatterplot(x='satisfaction_level', y='last_evaluation', hue='cluster', data=df_left, palette = 'viridis', s=100)
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')


# Cluster 0: They have low level of satisfaction and High evaluation
#            These are high performing employees who attrite due to dissatisfaction in their current job.
#            
# Cluster 1: High satisfaction level and high evaluation
#            They are high performers and are satisfied with their job. The major reason for turnover could be for better                    opportunities.
#            
# Cluster 2: Low satisfaction level and low evaluation
#            These employees are likely leaving due to dissatisfaction and poor performance evaluations.

# In[ ]:





# Handle the left Class Imbalance using the SMOTE technique.
# 4.1 Pre-process the data by converting categorical columns to numerical columns by:
# 1. Separating categorical variables and numeric variables
# 2. Applying get_dummies() to the categorical variables
# 3. Combining categorical variables and numeric variables
# 
# 4.2 Do the stratified split of the dataset to train and test in the ratio 80:20 with random_state=123.
# 4.3 Upsample the train dataset using the SMOTE technique from the imblearn module.

# #### Plotting class distribution

# In[19]:


class_distribution = pd.DataFrame(df['left'].value_counts().reset_index())


# In[20]:


class_distribution


# In[21]:


plt.subplot(1,2,1)
plt.bar(x='left',height='count',data=class_distribution, color=['blue','red'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(ticks=[0, 1], labels=['Did Not Leave (0)', 'Left (1)'], rotation=0)
plt.grid(axis='y')

plt.subplot(1, 2, 2)
plt.pie(class_distribution['count'], labels=['Did Not Leave (0)', 'Left (1)'], colors=['blue', 'red'], autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution')
plt.ylabel('')

# Show the plots
plt.tight_layout()
plt.show()


# Preprocessing

# In[22]:


df1 = pd.get_dummies(df, columns=['salary','sales'],drop_first=True).astype(int)


# In[23]:


df1.info()


# In[24]:


from sklearn.model_selection import train_test_split
X = df1.drop('left', axis=1)
y = df1['left']

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123, stratify=y)


# In[25]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(x_train,y_train)


# In[26]:


print(f'length of train data: {len(X_train_res)}')
print(f'length of test data: {len(x_test)}')


# In[27]:


print(len(X_train_res), len(x_train))
y_train_res.value_counts()


# Perform 5-fold cross-validation model training and evaluate performance.
# 
# 1. Train a logistic regression model, apply a 5-fold CV, and plot the classification report.
# 2. Train a Random Forest Classifier model, apply the 5-fold CV, and plot the classification report.
# 3. Train a Gradient Boosting Classifier model, apply the 5-fold CV, and plot the classification report.

# In[28]:


from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict,RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline


# In[29]:


log_reg = LogisticRegression()
rf_clf = RandomForestClassifier(random_state=123)
gb_clf = GradientBoostingClassifier(random_state=123)

log_reg_param = {
    'C': np.arange(0.01,10,0.5),
    'penalty': ['l1','l2'],
    'solver': ['liblinear','lbfgs']
}

rf_param = {
    'n_estimators' : np.arange(50, 500, 50),
    'max_depth' : [5,10,20,None],
    'min_samples_split': np.arange(2, 10, 2)
}

gb_param = {
    'n_estimators' : np.arange(50, 500, 50),
    'learning_rate' : [0.01,0.02,0.05,0.1],
    'max_depth' : [3,5,10]
}

def tune_and_evaluation(model, param_grid, X_train_res, y_train_res, x_test, y_test, cv=5):
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
 
    r_search = RandomizedSearchCV(model, param_distributions = param_grid, cv=skf, scoring = 'accuracy', n_jobs=-1,n_iter=20)
    r_search.fit(X_train_res,y_train_res)
    
    best_model = r_search.best_estimator_
    
    y_train_pred = cross_val_predict(best_model, X_train_res, y_train_res, cv=skf)
    y_test_pred = best_model.predict(x_test)
    y_test_proba = best_model.predict_proba(x_test)[:,1]
    
    print(f'\n Best Parameters: {r_search.best_params_}')
    print("\n Classification Report (Train Data - Final Evaluation)")
    print(classification_report(y_train_res,y_train_pred))
    
    print("\n Classification Report (Test Data - Final Evaluation)")
    print(classification_report(y_test, y_test_pred))
    
print('\n Training and Evaluating: Logistic Regression')
tune_and_evaluation(log_reg,log_reg_param, X_train_res, y_train_res, x_test, y_test)

print('\n Training and Evaluating: Randome Forest')
tune_and_evaluation(rf_clf,rf_param, X_train_res, y_train_res, x_test, y_test)  

print('\n Training and Evaluation: Gradient Boost Classifier')
tune_and_evaluation(gb_clf,gb_param,X_train_res, y_train_res, x_test, y_test)


# ## ROC/AUC Curve

# In[30]:


best_log_reg = LogisticRegression(C= 4.01, penalty= 'l2', solver= 'liblinear')  
best_rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=4, random_state=123)  
best_gb = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, random_state=123) 


# In[31]:


from sklearn.metrics import roc_curve, roc_auc_score

# Fit model
best_log_reg.fit(X_train_res, y_train_res)
best_rf.fit(X_train_res, y_train_res)
best_gb.fit(X_train_res, y_train_res)

# Predict
y_test_prob_log_reg = best_log_reg.predict_proba(x_test)[:,1]
y_test_prob_rf = best_rf.predict_proba(x_test)[:,1]
y_test_prob_gb = best_gb.predict_proba(x_test)[:,1]

# ROC and AUC score
fpr_log, tpr_log,thresholds = roc_curve(y_test,y_test_prob_log_reg)
fpr_rf, tpr_rf,thresholds = roc_curve(y_test,y_test_prob_rf)
fpr_gb, tpr_gb,thresholds = roc_curve(y_test,y_test_prob_gb)

auc_log = roc_auc_score(y_test, y_test_prob_log_reg)
auc_rf = roc_auc_score(y_test, y_test_prob_rf)
auc_gb = roc_auc_score(y_test, y_test_prob_gb)

#plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC={auc_log:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_log:.2f})')
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC={auc_log:.2f})')

#reference line
plt.plot([0,1],[0,1], 'k--', label='Random Classifier (AUC=0.5)')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()


# In[32]:


models = {
    'Logistic Regression': LogisticRegression(), #class_weight='balanced', solver='liblinear'
    'Random Forest': RandomForestClassifier(random_state=123), #, class_weight='balanced'
    'Gradient Boosting': GradientBoostingClassifier(random_state=123)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

plt.figure(figsize=(8, 6))

for model_name, model in models.items():
    train_probs = cross_val_predict(model, X_train_res, y_train_res, cv=cv, method = 'predict_proba')[:,1]
    #fpr, tpr, _ = roc_curve(y_train_res, train_probs)
    train_auc = roc_auc_score(y_train_res, train_probs)
    print(f'{model_name}: Cross validation AUC {train_auc:.4f}')
    
    model.fit(X_train_res,y_train_res)
    
    test_prob = model.predict_proba(x_test)[:,1]
    test_auc = roc_auc_score(y_test,test_prob)
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    plt.plot(fpr,tpr, label=f'{model_name}: AUC {test_auc:.2f}')
    
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves on Test Data')
plt.legend()
plt.show()


# Baseline Random forest and gradient boost model has better AUC score compared to the tuned models.

# ## Classification Report and Confusion Matrix

# Identify the best model and justify the evaluation metrics used.
# 
# 1. Find the ROC/AUC for each model and plot the ROC curve.
# 2. Find the confusion matrix for each of the models.
# 3. Explain which metric needs to be used from the confusion matrix: Recall or Precision?

# In[33]:


for model_name, model in models.items():
    
    print(f'\n{model_name}')
    y_train_pred = model.predict(X_train_res)
    print('Classification Report - Train data \n')
    print(classification_report(y_train_res,y_train_pred))

    y_test_pred = model.predict(x_test)
    print('\n Classification Report - Test data \n')
    print(classification_report(y_test,y_test_pred))
    
    conf_matrix = confusion_matrix(y_test,y_test_pred)
    print(f'Confusion Matrix: \n', conf_matrix)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels=['Left','Didnt Leave'])
    cm_display.plot()
    plt.show()
    


# Recall should be prioritized in employee turnover prediction, as failing to identify an employee who is likely to attrite might lead to loss of skilled employees and disruption of operation.

# Model with highest recall is Gradient Boosting Classifier

# ## Retention Strategy
# 
# Suggest various retention strategies for targeted employees.
# 
# 1. Using the best model, predict the probability of employee turnover for the test data.
# 
# 2. Categorization Based on Probability Score
# Employees are categorized into four zones based on their probability scores:
# 
# - **🟢 Safe Zone (Green)**: Score < 20%  
# - **🟡 Low-Risk Zone (Yellow)**: 20% ≤ Score < 60%  
# - **🟠 Medium-Risk Zone (Orange)**: 60% ≤ Score < 90%  
# - **🔴 High-Risk Zone (Red)**: Score > 90%  
# 
# 3. Suggested Retention Strategies

# In[34]:


df_test = x_test.copy()
df_test['left_prob'] = models['Gradient Boosting'].predict_proba(x_test)[:,1]


# In[35]:


df_test.head()


# In[36]:


def turnover_category(prob):
    if prob < 0.2:
        return 'Safe Zone (Green)'
    if 0.2 <= prob <= 0.6:
        return 'Low-Risk Zone (Yellow)'
    if 0.6 <= prob <= 0.9:
        return 'Medium-Risk Zone (Orange)'
    else:
        return 'High-Risk Zone (Red)'
    
df_test['attrition_band'] = df_test['left_prob'].apply(turnover_category)
    


# In[38]:


df_test = df_test.merge(y_test, left_index=True, right_index=True, how='left')


# In[39]:


ct = pd.crosstab(df_test['attrition_band'], df['left'])
ct


# #### 🟢 Safe Zone (Green)
# - Employees in this zone are **least likely to leave**.  
# - Maintain engagement through periodic check-ins and career growth opportunities.  
# 
# #### 🟡 Low-Risk Zone (Yellow)
# - Moderate risk of turnover.  
# - Provide **mentorship, upskilling, and growth opportunities** to maintain satisfaction.  
# 
# #### 🟠 Medium-Risk Zone (Orange)
# - Higher likelihood of leaving.  
# - Identify pain points through **employee feedback surveys** and improve work conditions.  
# - Offer **incentives, flexible work policies, or recognition programs**.  
# 
# #### 🔴 High-Risk Zone (Red)
# - Employees in this category have a **very high chance of leaving**.  
# - **Immediate action needed:** one-on-one discussions, salary adjustments, leadership engagement.  
# - Provide **personalized retention plans** based on concerns and aspirations.  
# 

# In[41]:


feature_importances = models['Gradient Boosting'].feature_importances_
feature_names = x_train.columns  # Replace with actual feature names

# Create a DataFrame
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='green')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()


# In[ ]:




