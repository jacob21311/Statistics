#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd  # python datasets 的縮寫
import numpy as np   # python numbers 的縮寫
import matplotlib.pyplot as plt  
from sklearn import datasets   # scikit-learn 的縮寫
from sklearn.tree import export_graphviz # 從 sklearn的決策樹模型 取得畫決策樹之套件


# In[3]:


# 資料讀取
iris = datasets.load_iris()

# 資料當中表達花瓣資料型態的特徵
x = pd.DataFrame(iris['data'], columns = iris['feature_names'])
print(x)


# In[4]:


# 資料當中表達資料分類的結果，並改為 pandas表格形式

y = pd.DataFrame(iris['target'], columns = ['target'] )
print('target_names :'+ str(iris['target_names']))


# In[5]:


# 把花瓣特徵與結果('target')的表格合併
iris_data = pd.concat([x,y],axis = 1)
iris_data = iris_data[['sepal length (cm)', 'petal length (cm)','target' ]]

# 我們先只處理兩個分類的情形 ('setosa = 0', 'versicolor = 1')
iris_data = iris_data[iris_data['target'].isin([0,1])]
iris_data


# In[36]:


from sklearn.model_selection import train_test_split       # 資料切割器
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  


# In[9]:


# 將資料區分成 70% 訓練資料與 30% 測試資料

x_train,x_test,y_train,y_test = train_test_split(
    iris_data[['sepal length (cm)','petal length (cm)']],iris_data[['target']],
    test_size = 0.3, random_state = 0)


# In[11]:


# 載入決策樹函式，criterion使用我們之前介紹過的entropy
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0)
tree.fit(x_train,y_train)


# In[12]:


# 訓練完成以後，接下來就是測試預測結果(從沒有看過的30%資料來去測試)

# 決策樹的預測結果(從 'sepal length (cm)', 'petal length (cm)'分類出來的結果)
tree.predict(x_test)


# In[15]:


# 真實的分類結果 

y_test['target'].values


# In[16]:


# 看看是否有答錯
# i 表示30個順序、v 表示 target的內容

error = 0
for i, v in enumerate(tree.predict(x_test)):
    if v != y_test['target'].values[i]:
        print(i,v)
        error = error + 1
        
print(error)


# In[26]:


test_score = tree.score(x_test,y_test['target'])
test_score


# In[24]:


export_graphviz(tree, 
                out_file = 'tree.dot', 
                feature_names =['sepal length (cm)', 'petal length (cm)'],
                class_names = iris.target_names,
                rounded=True, # 開啟round
                proportion=False, # 不顯示比例 顯示target的數量
                precision=2, # 小數點後第二位
                filled=True) # 是否根據Target來填顏色))


# In[34]:


# 印出預測精確率
print(f'Accuracy: {test_score:.1f}%')

# 印出文字版的決策樹
feature_names = iris_data[['sepal length (cm)', 'petal length (cm)']]
class_names = iris_data[['target']]

print(export_text(tree, feature_names=list(feature_names)))

