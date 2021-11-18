# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 08:19:56 2021

@author: Deepak.Reji
"""

import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from fitter import Fitter, get_common_distributions, get_distributions


st.set_option('deprecation.showPyplotGlobalUse', False)

#%%
def drought_events(df, col, option):
    data = df[col].to_frame()

    data.loc[data[col] < 0,'event'] = 1
    data.loc[data[col] > 0,'event'] = 0
    
    if option == "3-Month":
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
    event_count = []
    counter = 0
    
    for id in range(0,len(data)):
        if data['event'][id] == 0:
            event_count.append(np.nan)
        elif data['event'][id] == 1:
            if id == 0:
                counter+=1
                event_count.append(counter)
            elif data['event'][id-1] == 0:
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
    
    return event_df

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def normal_distribution(x):
    #Creating a Function.
    def normal_dist(x , mean , sd):
        prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
        return prob_density
     
    #Calculate mean and Standard deviation.
    mean = np.mean(x)
    sd = np.std(x)
     
    #Apply function to the data.
    pdf = normal_dist(x,mean,sd)
    
    return pdf


def main():
    """App with Streamlit"""
    
    st.info("Extracton of Drought Events and its parameters")
    uploaded_file = st.sidebar.file_uploader("Choose the raw data file", type="xlsx")
    
    col_names = ""
    df = pd.DataFrame()
    if uploaded_file:    
        df = pd.read_excel(uploaded_file)
        st.warning("Columns: " + ', '.join(list(df.columns)))
        
        col_names = df.columns
        remove_words = ["Year", "year", "date", "Date", "month", "Month"]
        col_names = [word for word in col_names if word not in remove_words]
    
    option = st.sidebar.selectbox(
        'Select the span',
        ('1-Month', '3-Month'))
    
    
    #search_string = st.sidebar.text_input("Enter the column name and press enter", "")
    
    search_string = st.sidebar.selectbox(
        'Select the column',
        (col_names))
    
    if search_string!= "":
        if search_string in list(df.columns):
            col = search_string
            event_df = drought_events(df, col, option)
            st.write("\n")
            if option == "3-Month":
                num = 3
                st.subheader("3-Month period drought events")
            else:
                num = 1
                st.subheader("1-Month period drought events")
            
            st.dataframe(event_df)
            
            btn_download = st.button("Click to Download the Spreadsheet")
            filename = col + "_"+str(num)+ "_month"+'_{}.csv'.format(str(pd.datetime.now().strftime("%Y-%m-%d %H%M%S")))

            if btn_download:
                tmp_download_link = download_link(event_df, filename, 'Click here to download your data!')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
            
            st.write("\n")
            
            tickbox = st.sidebar.checkbox("Show Distributions")
                       
            if tickbox:
                option1 = st.sidebar.selectbox(
                    'Select the drought parameter',
                    ([word for word in event_df.columns if word not in ['event no.']]))
                
                # st.subheader("Normal Distribution")
                # col1, col2 = st.columns([8,4])
                
                # with col1:
                #     sns.distplot(np.array(event_df[option1]))
                #     plt.show()
                #     st.pyplot()
                
                # with col2:
                #     pdf = normal_distribution(event_df[option1])
                #     normal_df = pd.DataFrame(event_df[option1])
                #     normal_df['normal'] = pdf
                #     st.dataframe(normal_df)
                
                filename1 = col + "_"+str(num)+ "_month"+ '_summary'+'_{}.csv'.format(str(pd.datetime.now().strftime("%Y-%m-%d %H%M%S")))
                f = Fitter(event_df[option1].values,
                           distributions=['norm', 'genextreme', 'expon', 'weibull_max', 'weibull_min', 'gamma', 'lognorm', 'logistic'])
                f.fit()
                st.write("\n")
                st.subheader("Summary")
                st.write(f.summary(Nbest=8))
                download_link1 = download_link(f.summary(Nbest=8), filename1,'download your summary!')
                st.markdown(download_link1, unsafe_allow_html=True)
                plt.show()
                st.write("\n")
                st.subheader("Combined Distribution Plot")
                st.pyplot()
                
                option2 = st.selectbox(
                    'Selection Criteria',
                    ('aic', 'bic', 'sumsquare_error'))
                
                st.success("Best Fitted Distribution Parameters")
                st.write(f.get_best(method = option2))
                            
if __name__ == "__main__":
    main()    