#!/usr/bin/env python
# coding: utf-8

# In[159]:


#####################################################################################################
######################### CLASSIFIED DATA SET  #####################################################
#####################################################################################################


# In[160]:


##########################################################################
############### Part I - Importing 
##########################################################################


import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[161]:


df = pd.read_csv('KNN_Project_Data')


# In[162]:


df.head()


# In[163]:


#####################################################################
########################### Part II - Duplicates
#####################################################################

df[df.duplicated()]                                   #### no duplicates found


# In[164]:


####################################################################
############## Part III - Missing Values
####################################################################


# In[165]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='summer',ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### no missing values either


# In[9]:


df.isnull().any()


# In[11]:


df.info()


# In[12]:


######################################################################
############## Part IV - EDA
######################################################################


# In[166]:


df.head()


# In[167]:


df['XVPM'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Classified Graph')

plt.xlabel('Number of Data points')

plt.ylabel('XVPM')


#### its going to be challenge to really makes sense of this data set because we are handling some classified data so we don't know what each columns mean


# In[168]:


df.XVPM.mean()


# In[169]:


df.XVPM.std()


# In[170]:


df['GWYH'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Classified Graph')

plt.xlabel('Number of Data Points')

plt.ylabel('GWYH')


#### again same issue with one here, mean seems to be same as the previous feature column
#### lets see the correlation and then go from there


# In[171]:


corr = df.corr()


# In[172]:


corr


# In[173]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


#### because we don't have much infomation about the data we will have to figure out a way to make sense of the data
#### for now we will only pay attention to three feature columns EDFS, TRAT and IGGA


# In[174]:


custom = {0:'black',
         1:'green'}

g = sns.jointplot(x=df.EDFS,y=df.IGGA,data=df,hue='TARGET CLASS',palette=custom)

g.fig.set_size_inches(17,9)


#### seems like the positive peaks around 2000 on EDFA scale and similar behavior at around 1000 on IGGA scale


# In[175]:


custom = {0:'black',
          1:'red'}

g = sns.jointplot(x=df.TRAT,y=df.IGGA,data=df,hue='TARGET CLASS',kind='kde',fill=True,palette=custom)

g.fig.set_size_inches(17,9)


#### we see our target range from this


# In[176]:


#### the pearson should be very strong on this one lets check it out

from scipy.stats import pearsonr


# In[177]:


co_eff,p_value = pearsonr(df.TRAT,df.IGGA)


# In[178]:


p_value                          #### makes sense, obviously we reject the null hypothesis here


# In[179]:


co_eff,p_value = pearsonr(df['TRAT'],df['TARGET CLASS'])


# In[180]:


p_value                          ##### same the case with this one which doesn't suprise me at all


# In[181]:


co_eff                           #### almost like 0.50 correlation with both columns
                                 #### co_eff goes from 0-1, more closer to 1 means more correlation


# In[182]:


pl = sns.FacetGrid(df,hue='TARGET CLASS',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'TRAT',fill=True)

pl.set(xlim=(0,df.TRAT.max()))

pl.add_legend()


#### clearly we can see our target class peaks and low peaks


# In[183]:


custom = {0:'black',
          1:'green'}

pl = sns.FacetGrid(df,hue='TARGET CLASS',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'IGGA',fill=True)

pl.set(xlim=(0,df.IGGA.max()))

pl.add_legend()


#### the peak for the target class on IGGA peaks around 1200


# In[184]:


custom = {0:'black',
          1:'purple'}

pl = sns.FacetGrid(df,hue='TARGET CLASS',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'EDFS',fill=True)

pl.set(xlim=(0,df.EDFS.max()))

pl.add_legend()

#### positive spikes around 1800-2000 for feature column EDFS


# In[185]:


custom = {0:'red',
          1:'green'}

sns.lmplot(x='EDFS',y='IGGA',data=df,height=7,aspect=2,hue='TARGET CLASS',palette=custom)


#### not the best fit honestly


# In[186]:


sns.lmplot(x='EDFS',y='TARGET CLASS',data=df,height=7,aspect=2,x_bins=[0,5,20,32,40,70,100,150,190,220,270,300,400,500,700,1000,1500,1800,2000,2500,3000,3196],line_kws={'color':'red'},scatter_kws={'color':'black'})

#### seems like a proper linear model


# In[187]:


sns.lmplot(x='TRAT',y='TARGET CLASS',data=df,height=7,aspect=2,x_bins=[32,40,70,100,150,190,220,270,300,400,500,700,1000,1500,1800,2000,2500,3000,3180],line_kws={'color':'red'},scatter_kws={'color':'purple'})


#### similar case here we see, proper linear correlation


# In[188]:


###################################################
########### PART V - PCA
###################################################


# In[189]:


X = df.drop(columns='TARGET CLASS')

X.head()


# In[190]:


y = df['TARGET CLASS']

y.head()


# In[191]:


from sklearn.preprocessing import StandardScaler

#### honestly this kind of data is the BEST type for PCA


# In[192]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[193]:


from sklearn.decomposition import PCA


# In[194]:


pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])
final_df = pd.concat([principal_df, y], axis=1)


# In[195]:


final_df.head()


# In[196]:


colors = {0: 'red', 1: 'green'}

plt.figure(figsize=(15, 6))

for i in final_df['TARGET CLASS'].unique():
    subset = final_df[final_df['TARGET CLASS'] == i]
    plt.scatter(subset['principal_component_1'], subset['principal_component_2'], 
                color=colors[i], label=f'TARGET CLASS = {i}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Titanic Dataset')
plt.legend()
plt.grid(True)

#### see how you can make cluster or classfication with perfection with this method


# In[197]:


pca.n_features_


# In[198]:


#### now lets say we were not provided the Target column and we wanted to cluster based on PCA so this is how you do it

X.columns


# In[199]:


df_comp = pd.DataFrame(pca.components_,columns=['XVPM', 'GWYH', 'TRAT', 'TLLZ', 'IGGA', 'HYKR', 'EDFS', 'GUUB', 'MGJM','JHZC'])


# In[200]:


df_comp.head()


# In[201]:


fig, ax = plt.subplots(figsize=(20,8))                     

sns.heatmap(df_comp,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


#### if we are not given the target column this is how we can use it


# In[202]:


#######################################################################
######################## Part VI - PCA Model
#######################################################################


# In[203]:


final_df.head()


# In[204]:


X = final_df.drop(columns='TARGET CLASS')

X.head()


# In[205]:


y = final_df['TARGET CLASS']

y.head()


# In[206]:


from sklearn.model_selection import train_test_split


# In[207]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[208]:


from sklearn.linear_model import LogisticRegression         #### for classification


# In[209]:


model = LogisticRegression()


# In[210]:


model.fit(X_train,y_train)


# In[211]:


y_predict = model.predict(X_test)


# In[212]:


from sklearn import metrics


# In[213]:


print(metrics.classification_report(y_test,y_predict))            #### not a bad model honestly


# In[214]:


#############################################################################
################# PART VII - Classification
#############################################################################


# In[215]:


df.head()                          #### we did the PCA models but now we will do the proper way of modelling


# In[216]:


X = df.drop(columns='TARGET CLASS')

X.head()


# In[217]:


y = df['TARGET CLASS']

y.head()


# In[218]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# In[219]:


X.columns


# In[220]:


preprocessor = ColumnTransformer(transformers=[
                                               ('num', StandardScaler(),['XVPM', 'GWYH', 'TRAT', 'TLLZ', 'IGGA', 'HYKR', 'EDFS', 'GUUB', 'MGJM','JHZC'])
                                              ]
                                )


# In[221]:


from sklearn.pipeline import Pipeline


# In[222]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[223]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[224]:


model.fit(X_train,y_train)


# In[225]:


y_predict = model.predict(X_test)


# In[226]:


metrics.accuracy_score(y_test,y_predict)


# In[227]:


print(metrics.classification_report(y_test,y_predict))             #### seems like PCA model was better then this


# In[228]:


from sklearn.ensemble import RandomForestClassifier                #### lets bring our boy random forest


# In[229]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])


# In[230]:


model.fit(X_train,y_train)


# In[231]:


y_predict = model.predict(X_test)


# In[232]:


print(metrics.classification_report(y_test,y_predict))         #### interesingly it performed worst then logistic


# In[233]:


from sklearn.model_selection import GridSearchCV              #### time to go for advanced


# In[234]:


get_ipython().run_cell_magic('time', '', "\nparam_grid = {\n    'classifier__n_estimators': [100, 200, 300],\n    'classifier__max_depth': [None, 10, 20, 30],\n    'classifier__min_samples_split': [2, 5, 10],\n    'classifier__min_samples_leaf': [1, 2, 4]\n}\n\nmodel_grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',verbose=2)\nmodel_grid.fit(X_train, y_train)")


# In[235]:


best_model = model_grid.best_estimator_


# In[236]:


y_predict = best_model.predict(X_test)


# In[237]:


metrics.accuracy_score(y_test,y_predict)                #### still not better then logistic


# In[238]:


###############################################
###### PART VIII - KNN
###############################################


# In[239]:


from sklearn.neighbors import KNeighborsClassifier


# In[240]:


get_ipython().run_cell_magic('time', '', "\nk_range = range(1,100)\n\naccuracy = []\n\nfor i in k_range:\n    \n    model = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', KNeighborsClassifier(n_neighbors=i))\n    ]) \n    \n    model.fit(X_train,y_train)\n    \n    y_predict = model.predict(X_test)\n    \n    accuracy.append(metrics.accuracy_score(y_test,y_predict))")


# In[241]:


plt.figure(figsize=(15,7))

plt.plot(k_range,accuracy,color='red', marker='o', linestyle='dashed',linewidth=2, markersize=10,markerfacecolor='black')

plt.xlabel('K Values')

plt.ylabel('Accuracy')

#### seems like we can achieve better with KNN if we increase the k value


# In[242]:


get_ipython().run_cell_magic('time', '', "\nk_range = range(90,200)\n\naccuracy = []\n\nfor i in k_range:\n    \n    model = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', KNeighborsClassifier(n_neighbors=i))\n    ]) \n    \n    model.fit(X_train,y_train)\n    \n    y_predict = model.predict(X_test)\n    \n    accuracy.append(metrics.accuracy_score(y_test,y_predict))")


# In[243]:


plt.figure(figsize=(15,7))

plt.plot(k_range,accuracy,color='red', marker='o', linestyle='dashed',linewidth=2, markersize=10,markerfacecolor='black')

plt.xlabel('K Values')

plt.ylabel('Accuracy')

#### seems like 90-100 is the sweet spot and after that it degrades


# In[244]:


get_ipython().run_cell_magic('time', '', "\nk_range = range(90,100)\n\naccuracy = []\n\nfor i in k_range:\n    \n    model = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', KNeighborsClassifier(n_neighbors=i))\n    ]) \n    \n    model.fit(X_train,y_train)\n    \n    y_predict = model.predict(X_test)\n    \n    accuracy.append(metrics.accuracy_score(y_test,y_predict))")


# In[245]:


plt.figure(figsize=(15,7))

plt.plot(k_range,accuracy,color='red', marker='o', linestyle='dashed',linewidth=2, markersize=10,markerfacecolor='black')

plt.xlabel('K Values')

plt.ylabel('Accuracy')

#### from the plot it seems k value 91 is our best shot


# In[246]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=91))
    ]) 
    
model.fit(X_train,y_train)
    
y_predict = model.predict(X_test)
    


# In[247]:


metrics.accuracy_score(y_test,y_predict)                    #### the best one yet


# In[248]:


print(metrics.classification_report(y_test,y_predict))      #### much better


# In[249]:


cm = metrics.confusion_matrix(y_test,y_predict)

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

fig, ax = plt.subplots(figsize=(20,10))

disp.plot(ax=ax)


# In[141]:


##############################################################################
################### PART IX - Advanced methods
##############################################################################


# In[142]:


#### although our requirement is satisfied but it doesn't hurt to see if we can improve even further

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[143]:


import xgboost as xgb


# In[144]:


clf_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

param_grid_xgb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.7, 0.8, 0.9],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9]
}


# In[145]:


from sklearn.model_selection import RandomizedSearchCV


# In[146]:


get_ipython().run_cell_magic('time', '', "\nrandom_search_xgb = RandomizedSearchCV(clf_xgb, param_grid_xgb, n_iter=50, cv=5, scoring='accuracy', random_state=42)\nrandom_search_xgb.fit(X_train, y_train)")


# In[147]:


best_model = random_search_xgb.best_estimator_


# In[148]:


y_predict = best_model.predict(X_test)


# In[149]:


metrics.accuracy_score(y_test,y_predict)


# In[150]:


print(metrics.classification_report(y_test,y_predict))          #### no improvement


# In[152]:


from sklearn.ensemble import StackingClassifier               #### one last try before we wrap it up


# In[153]:


base_models = [
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

meta_model = LogisticRegression()

stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)


# In[154]:


get_ipython().run_cell_magic('time', '', "\nmodel = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', stacking_clf)\n])\n\nmodel.fit(X_train, y_train)")


# In[155]:


y_predict = model.predict(X_test)


# In[156]:


metrics.accuracy_score(y_test, y_predict)


# In[157]:


print(metrics.classification_report(y_test,y_predict))               #### no improvement from here either


# In[158]:


#######################################################################################################################
#### We conducted extensive EDA, PCA, and classification modeling on a classified dataset. ############################
#### The restricted knowledge of column meanings limited our ability to perform a more in-depth EDA. ##################
#### Despite this constraint, we tested several classification models, ################################################
#### with KNN achieving the highest accuracy of 0.86. #################################################################
#######################################################################################################################

