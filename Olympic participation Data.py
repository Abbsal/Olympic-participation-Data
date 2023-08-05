#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[31]:


country_names= np.array(['Algeria','Angola','Argentina','Australia','Austria','Bahamas','Bangladesh','Belarus','Belgium','Bhutan','Brazil','Bulgaria','Cambodia','Cameroon','Chile','China','Colombia','Cyprus','Denmark','El Salvador','Estonia','Ethiopia','Fiji','Finland','France','Georgia','Ghana','Grenada','Guinea','Haiti','Honduras','Hungary','India','Indonesia','Ireland','Italy','Japan','Kenya', 'South Korea','Liberia','Malaysia','Mexico', 'Morocco','Nepal','New Zealand','Norway','Pakistan', 'Peru','Qatar','Russia','Singapore','South Africa','Spain','Sweden','Switzerland','Thailand', 'United Arab Emirates','United Kingdom','United States','Uruguay','Venezuela','Vietnam','Zimbabwe'

])


# In[30]:


country_gdp= np.array([2255.225482,629.9553062,11601.63022,25306.82494,27266.40335,19466.99052,588.3691778,2890.345675,24733.62696,1445.760002,4803.398244,2618.876037,590.4521124,665.7982328,7122.938458,2639.54156,3362.4656,15378.16704,30860.12808,2579.115607,6525.541272,229.6769525,2242.689259,27570.4852,23016.84778,1334.646773,402.6953275,6047.200797,394.1156638,385.5793827,1414.072488,5745.981529,837.7464011,1206.991065,27715.52837,18937.24998,39578.07441,478.2194906,16684.21278,279.2204061,5345.213415,6288.25324,1908.304416,274.8728621,14646.42094,40034.85063,672.1547506,3359.517402,36152.66676,3054.727742,33529.83052,3825.093781,15428.32098,33630.24604,39170.41371,2699.123242,21058.43643,28272.40661,37691.02733,9581.05659,5671.912202,757.4009286,347.7456605])


# In[18]:


# to obtain max and min value of two arrays to make sense of data.
max_gdp_per_capita= country_gdp.argmax() # .argmax() is the funnction for max/min value from table.
countries_max_gdp_per_capita = country_names[max_gdp_per_capita]
countries_max_gdp_per_capita


# In[19]:


min_gdp_per_capita= country_gdp.argmin()
countries_min_gdp_per_capita= country_names[min_gdp_per_capita]
countries_min_gdp_per_capita


# In[22]:


for country in country_names:
    print('list country {}'.format(country))


# In[27]:


# to obtain list of both arrays together

for i in range(len(country_names)):
    country = country_names[i]
    country_gdp_per_capita = country_gdp[i]
    print('country {} per capita gdp is {}'.format(country,country_gdp_per_capita))


# In[28]:


max_country_gdp = country_gdp.argmax()
country_max_gdp = country_names[max_country_gdp]
country_max_gdp


# In[33]:


min_country_gdp = country_gdp.argmin()
country_min_gdp = country_names[min_country_gdp]
country_min_gdp


# In[38]:


# basic data informaton
print(country_gdp.max())
print(country_gdp.min())
print(country_gdp.mean())
print(country_gdp.std())


# In[43]:


# olympic medals won by different countries
np.country = np.array(['Great Britain','China','Russia','United States','Korea','Japan','Germany'])
np.gold = np.array([29,38,24,46,13,7,11])
np.silver = np.array([17,28,25,28,8,14,11])
np.bronze = np.array([19,22,32,29,7,17,14])
            


# In[46]:


max_gold = np.gold.argmax()
max_gold_country = np.country[max_gold]
max_gold_country


# In[47]:


max_silver= np.silver.argmax()
max_silver_country = np.country[max_silver]
max_silver_country


# In[48]:


print(np.country[np.gold>20])


# In[49]:


print(np.country[np.silver<10])


# In[52]:


for i in range(len(np.country)):
    gold_medal = np.gold[i]
    country = np.country[i]
    total_medals = np.gold[i]+np.silver[i]+np.bronze[i]
    print('{},gold medal {},total medals {}'.format(country,gold_medal,total_medals))


# In[53]:


# scipy for scintific computing
# integration and optimization functions
import numpy as np
from scipy import optimize


# In[58]:


def f(x):
    return x**2 + 5*np.sin(x)


# In[56]:


minimavalue = optimize.minimize(f,x0=2,method='bfgs',options= {'disp':True})


# In[62]:


minimavalue = optimize.minimize(f,x0=2,method='bfgs')
minimavalue


# In[63]:


import numpy as np
from scipy import linalg


# In[69]:


matrix= np.array([[4,5],[3,5]])
matrix


# In[70]:


type(matrix) # to find type of dataframe


# In[71]:


linalg.inv(matrix) #inverse of a square matrix only


# In[5]:


import numpy as np
import pandas as pd 


# In[6]:


olympic_data = {'Hostcity':['London','Beijing','Athens','Sydney','Atlanta'],
               'year':['2012','2015','2008','2004','2000'],
               'no of participating countries':[205,204,201,200,240]}


# In[9]:


olympic_data_df = pd.DataFrame(olympic_data)
olympic_data_df


# In[10]:


import numpy as np
import pandas as pd


# In[12]:


olympic_series_participation = pd.Series([205,208,201,200,197],index=[2012,2008,2004,2000,1996])
olympic_series_country = pd.Series(['London','Beijing','Athens','Sydney','Atlanta'],
                         index=[2012,2008,2004,2000,1996])
df_olympic_series = pd.DataFrame({'No of participating countries':olympic_series_participation,
                                 'Host cities':olympic_series_country})


# In[13]:


df_olympic_series


# In[1]:


import pandas as pd


# In[3]:


movie_ratings = pd.DataFrame(
                {'Movie 1':[5,4,3,2,1],
                'Movie 2':[4,5,3,4,2]},
                index = ['Tom','Jeff','Peter','Ram','Ted']
)


# In[4]:


movie_ratings


# In[6]:


def movie_grade(rating):
    if rating==5:
    return 'A'
    if rating==4:
    return 'B'
    if rating==3:
    return 'C'
    else:
    return 'F'


# In[7]:


movie_grade


# In[3]:


# To create dataframe and series using pandas
import numpy as np
import pandas as pd


# In[9]:


olympic_series_part = pd.Series([205,206,250,263,239],
                               index=[2012,2008,2004,2000,1996])
olympic_series_country = pd.Series(['London','Beijing','Athens','Sydney','Atlanta'],
                                  index=[2012,2008,2004,2000,1996])
df_olympic_series= pd.DataFrame({'No of participating countries':olympic_series_part,
                                'Host citites':olympic_series_country})


# In[10]:


df_olympic_series


# In[11]:


olympic_series_country


# In[12]:


df_olympic_series.head(2)


# In[13]:


df_olympic_series.tail(3)


# In[14]:


df_olympic_series.index


# In[16]:


df_olympic_series['Host citites']


# In[1]:


#Pandas SQL operation

import pandas as pd


# In[2]:


import sqlite3


# In[3]:


#create a table
create_table = """
CREATE TABLE student_score
(Id INTEGER, Name VARCHAR(20),math REAL,
Science REAL);"""


# In[5]:


#execute the sql statement
executeSQL = sqlite3.connect(':memory:')
executeSQL.execute(create_table)
executeSQL.commit()


# In[6]:


#prepare a sql query
SQL_query = executeSQL.execute('select * from student_score')


# In[8]:


#fetching result from SQLite database
resulset = SQL_query.fetchall()


# In[10]:


resulset


# In[1]:


import pandas as pd


# In[2]:


df_faa_dataset = pd.read_csv('C:\\Users\\OMEN\\Downloads\\Lesson 7 -1\\faa_ai_prelim\\faa_ai_prelim.csv')


# In[3]:


df_faa_dataset.head()


# In[5]:


df_faa_dataset.shape


# In[6]:


df_faa_dataset.columns


# In[9]:


df_analyse_dataset = df_faa_dataset[['ACFT_MODEL_NAME','LOC_STATE_NAME','ACFT_MODEL_NAME','RMK_TEXT',
                                    'FLT_PHASE','EVENT_TYPE_DESC','FATAL_FLAG']] 


# In[10]:


type(df_analyse_dataset)


# In[12]:


df_analyse_dataset.head()


# In[15]:


#replace all NaN for fatal flag with No
df_analyse_dataset['FATAL_FLAG'].fillna(value= 'No',inplace= True)


# In[20]:


#replace all NaN for fatal flag with No
df_analyse_dataset['FATAL_FLAG'].fillna(value ='No',inplace= True)


# In[19]:


df_analyse_dataset.head()


# In[ ]:




