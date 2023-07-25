#!/usr/bin/env python
# coding: utf-8

# # Decision Tree
# 
# 決策樹在訓練過程中會從最後上方的樹根開始將資料的特徵將資料分割到不同邊<br>
# 分割的原則是：這樣的分割要能得到最大的資訊增益(Information gain, 簡稱IG)。
# 
# 由於我們希望獲得的資訊量要最大，因此經由分割後的資訊量要越小越好。
# 
# 常見的資訊量有兩種：熵(Entropy) 以及 Gini不純度(Gini Impurity)
# 
# 
# 

# ## 載入Iris資料集

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os,sys
sys.path.append(os.path.abspath('./')) #for import common.utility
from utility import plot_confusion_matrix,plot_decision_regions,testcase_report


# ## 只用2個特徵進行分類
# 'sepal length (cm)', 'petal length (cm)'

# In[4]:


iris = datasets.load_iris()
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("target_names: "+str(iris['target_names']))
y = pd.DataFrame(iris['target'], columns=['target'])
iris_data = pd.concat([x,y], axis=1)
#iris_data = iris_data[['sepal length (cm)','petal length (cm)','target']]
#只取 target 0,2
iris_data = iris_data[iris_data['target'].isin([1,2])]
print(iris_data.shape)


# In[5]:


iris['feature_names']


# ## 將資料的70%拿出來train，剩下的30％用來檢測train的好壞

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(
    iris_data.drop(['target'],axis=1), iris_data['target'], test_size=0.3,random_state = 1)


# # Decision Tree 不需要做特徵標準化

# In[5]:


# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)


# ## DecisionTreeClassifier 參數
#     criterion: entropy, gini impurity
#     max_depth : 主要是可以防止樹長得過高造成overfit
#     max_features: 最多只能挑K種feature去分類
#     min_samples_leaf: leaf node最小sample數
#     

# In[7]:


from sklearn.tree import DecisionTreeClassifier


# ### use entropy as a criterion

# In[32]:


clf = DecisionTreeClassifier(criterion = 'entropy',max_depth=3, random_state=0) #max_depth=1,3,12


# ### class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)

# In[33]:


clf.fit(X_train,y_train)


# ## 計算正確率

# In[34]:


print('training score:',clf.score(X_train,y_train))
print('test score:',clf.score(X_test,y_test))


# In[35]:


clf.predict(X_test)


# In[36]:


y_test.values


# ### training Report

# In[37]:


report=testcase_report(iris_data,clf,X_train,X_train,y_train)
report[0]


# ###  test Report

# In[38]:


report=testcase_report(iris_data,clf,X_test,X_test,y_test)
report[0]


# ## 樹視覺化
# 
# 　scikit-learn中决策树的可视化一般需要安装graphviz。主要包括graphviz的安装和python的graphviz插件的安装。
# 
# 　　　　(可省略)安装graphviz。下載windows GraphViz's 工具  https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# 
# 　　　　安装python插件graphviz： conda install graphviz
# 
# 　　　　安装python插件pydotplus。 conda install pydotplus
# 
# 　　　　这样环境就搭好了，有时候python会很笨，仍然找不到graphviz，这时，可以在代码里面加入这一行：
# 
# 　　　　os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#     
#     
# ### Open "command or conda prompt" and run:
# 
# #### conda update conda
# #### conda update anaconda
# 
# 　　　　

# In[39]:


from sklearn.tree import export_graphviz
from sklearn import tree
import os
#os.environ['PATH'] = os.environ['PATH'] + (';c:\\Program Files (x86)\\Graphviz2.38\\bin\\')


# ### 方法1：使用 pydotplus 直接生成 iris.pdf

# In[40]:


import pydotplus
dot_data=tree.export_graphviz(clf,out_file=None)
graph=pydotplus.graph_from_dot_data(dot_data)
if not os.path.exists('output'):
    os.mkdir('output')
graph.write_pdf('output/iris.pdf')


# ### 方法2：直接在 jupyter notebook 中生成

# In[41]:


from IPython.display import Image  
#將 Decisson Tree Classifier 放入
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris['feature_names'],  
                         class_names=iris.target_names,
                         filled=True, rounded=True,  
                         special_characters=True)   
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  


# ## Return the feature importances.
# 
# The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It

# In[42]:


iris['feature_names']


# In[43]:


clf.feature_importances_


# ### use gini index as a criterion

# In[44]:


clf_2 = DecisionTreeClassifier(criterion = 'gini', random_state=0,max_depth=3)


# In[45]:


X_train.head(5)


# In[46]:


clf_2.fit(X_train,y_train)


# In[47]:


from IPython.display import Image  
#將 Decisson Tree Classifier 放入
dot_data = tree.export_graphviz(clf_2, out_file=None,
                         feature_names=iris['feature_names'],  
                         class_names=iris.target_names,
                         filled=True, rounded=True,  
                         special_characters=True)   
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  


# # 只挑2個特徵進行分類
# petal width (cm)','petal length (cm)

# In[48]:


iris = datasets.load_iris()
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("target_names: "+str(iris['target_names']))
y = pd.DataFrame(iris['target'], columns=['target'])
iris_data = pd.concat([x,y], axis=1)
iris_data = iris_data[['petal width (cm)','petal length (cm)','target']]
iris_data.head(3)


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(
    iris_data[['petal width (cm)','petal length (cm)']], iris_data['target'], test_size=0.3, random_state=0)


# In[50]:


from sklearn.tree import DecisionTreeClassifier
clf_3 = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state=0)
clf_3.fit(X_train,y_train)


# In[51]:


from IPython.display import Image  
#將 Decisson Tree Classifier 放入
dot_data = tree.export_graphviz(clf_3, out_file=None,
                         feature_names=['petal width (cm)','petal length (cm)'],  
                         class_names=iris.target_names,
                         filled=True, rounded=True,  
                         special_characters=True)   
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  


# ## 視覺化決策樹的決策邊界

# In[52]:


plot_decision_regions(X_train.values, y_train, classifier=clf_3)
plt.xlabel('petal width')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# # Random Forest 隨機森林分類器
# 
# * Step1. 建立特徵X，與目標y
# * Step2. 將資料區分成訓練集與測試集，可自行設定區分的百分比
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
# * Step3. 選擇隨機森林分類器，內容可決定決策數的棵樹、剪枝葉等等，以提升模型的效率及避免過度配適。
# rfc=RandomForestClassifier(n_estimators=100,n_jobs = -1,random_state =50, min_samples_leaf = 10)
# * Step4. 用建立好的模型來預測資料rfc.predict(X_test)
# * Step5. 檢驗模型的正確率
# rfc.score(X_test,y_test)

# In[53]:


# 從sklearn.ensemble 套件中引入 RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier 


# In[54]:


# n_estimator表示樹的數量, random_state表示亂數選擇(每次特徵值都不相同), n_jobs表示平行運算同時進行中
forest = RandomForestClassifier(criterion='entropy', n_estimators=10,random_state=0,n_jobs=8) 


# In[55]:


# 分類的主要特徵為何
X_train.keys()


# In[56]:


# 執行 forest的實際配適
forest.fit(X_train,y_train)


# ## 計算分數

# In[57]:


print('training score:',forest.score(X_train,y_train))
print('test score:',forest.score(X_test,y_test))


# In[58]:


# 有兩筆資料分類錯誤 找出哪兩筆 通常都為分類1與分類2
# 可以看到分類為2的部分被劃分到分類1
# 訓練資料
report=testcase_report(iris_data,forest,X_train,X_train,y_train)
report[0]


# In[59]:


# 測試資料
report=testcase_report(iris_data,forest,X_test,X_test,y_test)
report[0]


# In[60]:


# 畫出隨機森林的分類圖
# 訓練資料
plot_decision_regions(X_train.values, y_train, classifier=forest)
plt.title('Random Forest for training data',fontsize=16)
plt.xlabel('petal width')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# In[61]:


# 畫出隨機森林的分類圖
# 測試資料
plot_decision_regions(X_test.values,y_test, classifier=forest)
plt.title('Random Forest for test data',fontsize=16)
plt.xlabel('petal width')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# ## 利用RandomForest找出Iris data 的分類主要特徵

# In[62]:


forest_1 = RandomForestClassifier(criterion='entropy', n_estimators=10,random_state=0,n_jobs=8) 

iris = datasets.load_iris()
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("target_names: "+str(iris['target_names']))
y = pd.DataFrame(iris['target'], columns=['target'])
iris_data = pd.concat([x,y], axis=1)

iris_data.head()


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(
     iris_data.drop(['target'],axis=1), iris_data['target'], test_size=0.3, random_state=0)


# In[64]:


forest_1.fit(X_train,y_train)


# In[65]:


importances = forest_1.feature_importances_ #現在找出特徵的重要性，共有四個特徵
print(importances)
indices = np.argsort(importances)  # np.argsort :Returns the indices that would sort an array. 
print(indices) #排序索引(最小開始編號)


# In[66]:


features =X_train.keys()
features


# In[67]:


features[indices]


# In[68]:


plt.figure(1) # 連續執行圖像，數字為編碼
print(features[indices])

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')


# In[ ]:





# In[ ]:




