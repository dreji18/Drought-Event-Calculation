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
from copulas.visualization import scatter_3d
            
import plotly.graph_objects as go
import numpy as np
import plotly.express as px        

import plotly.figure_factory as ff

from copulas.univariate import Univariate


#st.set_option('deprecation.showPyplotGlobalUse', False)

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

def plot_3d(data):
    x, y, z = data.iloc[:, 0].values, data.iloc[:, 1].values, data.iloc[:, 2].values
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=12,
            color=z,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
            )
    )])
    
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return st.write(fig)
    
def plot_2d(data):
    data = data.abs()
    fig = px.scatter(data, x=data.columns[0], y=data.columns[1], color=data.columns[0],
                     size=data.columns[1])
    
    return st.write(fig)
    
def plot_1d(data):
    colors = ['#4b0082']
    fig = ff.create_distplot([data[c] for c in data.columns], data.columns, bin_size=.25, colors=colors)
    
    return st.write(fig)


def main():
    """Drought events and parameters"""
    
    st.info("Extracton of Drought Events and its parameters")
    
    select_options = ("Drought Parameters & Distributions", "Copula")
    select_value = st.sidebar.radio("Select the Desired Option", select_options,0)
    
    if select_value == "Drought Parameters & Distributions":
    
        st.header("🎲 Drought Parameters & Distributions") 
        uploaded_file = st.sidebar.file_uploader("Choose the raw data file", type="xlsx")
        
        col_names = ""
        df = pd.DataFrame()
        if uploaded_file:    
            df = pd.read_excel(uploaded_file)
            st.warning("Fields: " + ', '.join(list(df.columns)))
            
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
                        
                        filename1 = col + "_"+str(num)+ "_month"+ '_summary'+'_{}.csv'.format(str(pd.datetime.now().strftime("%Y-%m-%d %H%M%S")))
                        f = Fitter(event_df[option1].values, bins=len(event_df),
                                   distributions=['norm', 'genextreme', 'expon', 'weibull_max', 'weibull_min', 'gamma', 'lognorm', 'logistic'])
                        f.fit()
                        st.write("\n")
                        st.subheader("Summary")
                        st.write(f.summary(Nbest=8))
                        download_link1 = download_link(f.summary(Nbest=8), filename1,'download your summary!')
                        st.markdown(download_link1, unsafe_allow_html=True)
                        #plt.show()
                        st.write("\n")
                        st.subheader("Combined Distribution Plot")
                        st.pyplot(plt)
                        
                        option2 = st.selectbox(
                            'Selection Criteria',
                            ('aic', 'bic', 'sumsquare_error'))
                        
                        st.success("Best Fitted Distribution Parameters")
                        st.write(f.get_best(method = option2))
    
    # """copula"""
    if select_value == "Copula":
    
        st.header("🎲 Copula") 
        uploaded_file1 = st.sidebar.file_uploader("Choose the drought parameters file", type="csv")
        
        if uploaded_file1:
            df1 = pd.read_csv(uploaded_file1)
            
            st.subheader("Visualize the data using 3D plots")
            
            multi_options = st.multiselect(
                'Select upto three drought parameter',
                ['Severity', 'Peak', 'Duration', 'Inter-Arrival time'],
                ['Duration', 'Severity'])
            st.write("Note: Severity and Peak are in the order of negative magnitude")
            
            data = df1[multi_options]
            
            if len(multi_options) == 0:
                st.write("")
                
            elif len(multi_options) == 3:
                plot_3d(data)
            
            elif len(multi_options) == 2:
                plot_2d(data)
            
            elif len(multi_options) == 1:
                plot_1d(data)
              
            else:
                st.error("Sorry Can't be processed")
            
            # calculating distributions and select best fit
            st.subheader("Create Copula Instance")
            multi_options1 = st.multiselect(
                'Select upto three drought parameter',
                ['Severity', 'Peak', 'Duration', 'Inter-Arrival time'])
            
            if multi_options1:
                data1 = df1[multi_options1]
                
                fitted = []
                for col in data1.columns:
                    col_data = data1[col]
                    
                    f = Fitter(col_data,bins=len(col_data),
                               distributions=['gamma','lognorm',"beta","burr","norm"])
                    f.fit()
                    
                    curve = ""
                    for i in f.get_best(method = 'aic').keys():
                        curve = i
                        break
                    
                    st.write(col + " field : Best fit was " + curve + " curve")
                    fit_values = f.fitted_pdf
                    fitted.append(fit_values[curve])
                
                fit_df = pd.DataFrame(np.vstack(fitted)).T
                fit_df.columns = data1.columns
                
                st.write("")
                if len(multi_options1) == 1:
                    st.warning('Univariate Copulas: Beta, Gamma, Gaussian, Gaussian KDE, Log-Laplace, Student T, Truncated Gaussian, Uniform')
                    univariate = Univariate()
                    univariate.fit(fit_df)
                    parameters = univariate.to_dict()
                    st.success("Best fitted Copula :" + parameters['type'])
                    
                
                elif len(multi_options1) == 2:
                    st.warning('Bivariate Copulas: Clayton, Frank, Gumbel')
                
                elif len(multi_options1) == 3:
                    st.warning('Multivariate Copulas: Gaussian Copula, D-Vine, C-Vine, R-Vine')
                
                else:
                    st.error("Sorry Can't be processed")
                    
                    
                    

                
if __name__ == "__main__":
    main()    