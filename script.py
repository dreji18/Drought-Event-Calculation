# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 18:09:06 2021

@author: Deepak.Reji
"""

import os
import pandas as pd

# path were files are stored
os.chdir(r'C:\Users\deepak.reji\Documents\OneDrive - ERM\Personal References\Personal Experiment\Copula')

df = pd.read_excel("raw data.xlsx")

col = "Unnamed: 11" #<---- Change the column name here

#%% ------------------ Step-1----------------------------------------------
data = df[col].to_frame()

data.loc[data[col] < 0,'event'] = 1
data.loc[data[col] > 0,'event'] = 0

# 0 condition
for id in range(0, len(data)):
    try:
        if data['event'][id] == 0:
            if data['event'][id+1] ==1 and data['event'][id-1] == 1:
                data['event'][id] =1
    except:
        continue
    
    try:
        if data['event'][id] == 0 and data['event'][id+1] == 0:
            if data['event'][id-1] == 1 and data['event'][id+2] == 1:
                data['event'][id] =1
                data['event'][id+1] = 1
    except:
        continue

# 1 condition
for id in range(0, len(data)):
    try:
        if data['event'][id] == 1:
            if data['event'][id+1] ==0 and data['event'][id-1] == 0:
                data['event'][id] =0
    except:
        continue
    
    try:
        if data['event'][id] == 1 and data['event'][id+1] == 1:
            if data['event'][id-1] == 0 and data['event'][id+2] == 0:
                data['event'][id] =0
                data['event'][id+1] = 0
    except:
        continue

data = data.dropna()
data = data.reset_index(drop=True)

# counting the event
import numpy as np
event_count = []
counter = 0

for id in range(0,len(data)):
    if data['event'][id] == 0:
        event_count.append(np.nan)
    elif data['event'][id] == 1:
        if data['event'][id-1] == 0:
            counter+=1
            event_count.append(counter)
        else:
            event_count.append(np.nan)

data['event num'] = event_count
data['event num'] = data['event num'].ffill(axis = 0)

data2 = data.copy()
data2 = data2[[col, 'event num']]
event_arivaltime = data2.groupby(['event num']).count().reset_index()


data.loc[data['event'] == 0,'event num'] = np.nan


#%%
df1 = data[[col, 'event num']]
event_sum = df1.groupby(['event num']).sum().reset_index()

event_count = df1.groupby(['event num']).count().reset_index()

df2 = df1[df1[col] < 0]
event_max = df2.groupby(['event num']).min().reset_index()

event_df = pd.DataFrame()
event_df[['event no.', 'Severity']] = event_sum
event_df['Peak'] = event_max[col]
event_df['Duration'] = event_count[col]
event_df['Inter-Arrival time'] = event_arivaltime[col]

event_df.to_csv(col +'_{}.csv'.format(str(pd.datetime.now().strftime("%Y-%m-%d %H%M%S"))))

print(event_df) #<---- event_df will contain the output

##----------------------------------------------------------------------------------------##
#%%














# #%%
# # event Duration

# data = df['SSI'].to_frame()

# event = []
# for id in range(0, len(data)):
    
#     if data['SSI'][id] < 0:
#         event.append(1)
#     else:
#         #event.append(0)
#         try:
#              if data['SSI'][id+1] < 0 and data['SSI'][id-1] < 0:
#                  event.append(1)
#                  #break
#              elif data['SSI'][id+2] < 0 or data['SSI'][id-2] < 0:
#                  event.append(1)
#                  #break
#              else:
#                  event.append(0)
#         except:
#              event.append(0)
        
# data['drought event'] = event

# #data['event'] = data['SSI'] < 0


# #%%

# data = df['SSI'].to_list()

# event = []
# for id in range(0, len(data)):
#     id = 17
#     if data[id] < 0:
#         print("1")
#         event.append(1)
#     else:
#         #id < len(data)-2:
#         try:
#             if data[id + 1] < 0 or data[id +2] < 0:
#                 if id != 0 and id!=1:
#                     print("2")
#                     if data[id+2] < 0:
#                         if data[id-1] < 0:
#                             print("21")
#                             event.append(1)
#                         else:
#                             if data[id+1] < 0:
#                                 event.append(0)
#                             else:
                                
#                                 print("22")
#                                 event.append(0)
#                     else:
#                         print("23")
#                         event.append(1)
#                 else:
#                     print("3")
#                     event.append(0)
#             else:
#                 print("4")
#                 event.append(0)
#         except:
#             print("5")
#             event.append(0)

# #%%
# # working

# data = df['SSI'].to_frame()

# data.loc[data['SSI'] < 0,'event'] = 1
# data.loc[data['SSI'] > 0,'event'] = 0

# #data.to_excel("condition2 events.xlsx")


# event_new = []
# for id in range(0, len(data)):
#     if data['event'][id] == 0:
#         try:
#             if data['event'][id+1] ==1 and data['event'][id-1] ==1:
#                 event_new.append(1)
#             # else:
#             #     event_new.append(0)
            
#             elif data['event'][id+2] ==1 and data['event'][id-2] ==1:
#                 event_new.append(1)
           
#             else:
#                 event_new.append(0)
            
            
#         except:
#             event_new.append(0)
#     else:
#         event_new.append(1)

# data['new event'] = event_new

# df1 = data[['SSI', 'new event']]   

# df1.to_excel('condition1 events.xlsx')
# ####################

# #%%
# # 09/11/2021
# data = df['SSI'].to_frame()

# data.loc[data['SSI'] < 0,'event'] = 1
# data.loc[data['SSI'] > 0,'event'] = 0


# for id in range(0, len(data)):
#     if data['event'][id]!= "drought" or data['event'][id]!= "no drought":
#         if data['event'][id] == 0:
#             if data['event'][id+1] ==0 and data['event'][id+2] == 0:
#                 data['event'][id] = "no drought"
#                 data['event'][id+1] = "no drought"
#                 data['event'][id+2] = "no drought"
                
# #%%








# #%%
# # event count
# import numpy as np

# event_count = []
# counter = 0

# for id in range(0,len(df1)):
#     if df1['new event'][id] == 0:
#         event_count.append(np.nan)
#     elif df1['new event'][id] == 1:
#         if df1['new event'][id-1] == 0:
#             counter+=1
#             event_count.append(counter)
#         else:
#             event_count.append(np.nan)

# df1['event count'] = event_count
# df1['event count'] = df1['event count'].ffill(axis = 0)

# df1.loc[df1['new event'] == 0,'event count'] = np.nan

# #%%
# # severity
# df1.groupby(['event count']).sum()
            


#%%



