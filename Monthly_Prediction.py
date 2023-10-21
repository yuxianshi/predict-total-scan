# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from matplotlib import ticker
from scipy.stats import t

LOGGER = get_logger(__name__)

# functions to get prediction matrix X and response Y
def genX(df, variables):
    X = df[variables].to_numpy()
    X = np.concatenate((np.repeat(1, X.shape[0]).reshape(-1, 1), X), axis=1)
    return X

def genY(df):
    return df[['count']].to_numpy().reshape(-1, 1)

# function to fit model
def fit_LR_model(X, Y):
        XTX = np.linalg.inv(np.matmul(X.transpose(), X))
        return np.matmul(XTX, np.matmul(X.transpose(), Y))

# function to calculate Residue Sum of Squares of Linear Regression Model
def RSS_LR_model(X, Y, beta_hat):
        Y_hat = np.matmul(X, beta_hat)
        residue = Y - Y_hat
        return np.matmul(residue.transpose(), residue)[0][0]

# function for prediction
def LR_predict_X(X, beta_hat):
    Y = np.matmul(X, beta_hat)
    return Y

# helper function for plot_prediction() -- use million as unit in x-axis
@ticker.FuncFormatter
def million_formatter(x, pos):
    return '{} M'.format(x/1E6)

# update prediction using user input
@st.cache_data
def update_prediction(input_2022, X, Y, X_month, X_month_adjusted, sqrt_nday):
    # incorporate new input to training data
    m_observed = np.where(~np.isnan(input_2022))[0].tolist()#.reshape(1, -1)
    m_not_observed = np.where(np.isnan(input_2022))[0].tolist()
    input_2022 = np.array(input_2022).reshape(-1, 1)
    X_month_2022 = X_month_adjusted[m_observed]

    X_new = np.concatenate((X, X_month_2022), axis = 0)
    Y_new = np.concatenate((Y, (input_2022 / sqrt_nday)[m_observed]), axis = 0)

    # update model
    beta_hat = fit_LR_model(X_new, Y_new)
    XTX = np.linalg.inv(np.matmul(X_new.transpose(), X_new))

    # calculate the estimated standard deviation of \epsilon
    RSS = RSS_LR_model(X_new, Y_new, beta_hat) # residue sum of squres for full model
    degFreedom = X_new.shape[0] - X_new.shape[1]
    sigma_hat = np.sqrt(RSS / degFreedom) 

    X_month_to_predict = X_month[m_not_observed]
    pred = LR_predict_X(X_month_to_predict, beta_hat)
    sd = sigma_hat * np.sqrt(X_month_to_predict[:,0] + X_month_to_predict.dot(XTX).dot(X_month_to_predict.transpose()).diagonal()).reshape(-1, 1)

    Y_hat = input_2022
    Y_hat[m_not_observed] = pred
    yerr = np.zeros((12,1))
    yerr[m_not_observed] = -t.ppf(0.025, degFreedom) * sd

    CI_2 = Y_hat + yerr
    CI_1 = Y_hat - yerr

    res = np.concatenate((np.array(range(0,12)).reshape(-1, 1), Y_hat, yerr, CI_1, CI_2), axis = 1)
    res = pd.DataFrame(res, columns = ['pos', 'Y_hat', 'yerr', 'CI_1', 'CI_2'])
    res['month'] = ['Jan.', ' Feb.', 'Mar.', 'Apr.', 'May.', 'Jun.', 'Jul.', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.'] 

    return res, m_observed

# plot updated prediction
def plot_prediction(res, m_observed = []):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 8.4))
    
    col = [('tab:blue' if i in m_observed else 'orange') for i in range(12)]
    fig = ax.barh(res['pos'], res['Y_hat'],  xerr = res['yerr'], align='center', color = col)
    ax.set_yticks(res['pos'], labels = res['month'], fontsize = 15)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.xaxis.set_major_formatter(million_formatter)
    lab = res['Y_hat'].map(lambda x: '{} ± '.format(round(x/1E6,2))).str.cat(res['yerr'].map(lambda x : '{}M'.format(round(x/1E6,2))), sep = '')
    if len(m_observed) > 0:
        lab.iloc[m_observed] = res['Y_hat'].map(lambda x: '{}'.format(round(x/1E6,2)))[m_observed]
    ax.bar_label(fig, lab, padding =15, fontsize = 14)
    ax.set_xlim(right=550_000_000)#150_000_000 + res['Y_hat'].max())
    ax.set_xlabel('')
    ax.set_title('2022 - Monthly observation and prediction of number of scanned receipts' , {'fontsize':17})
    return ax.get_figure() 

# the main function
def run():
    st.set_page_config(
        page_title="Scanned Numbers Prediction",
        page_icon="📈",
    )

    st.write("# Prediction for number of scanned receipts in 2022")

    st.markdown(
        """
        Monthly count of scanned receipts is predicted for 2022 using 
        [daily observations in 2021](https://fetch-hiring.s3.amazonaws.com/machine-learning-engineer/receipt-count-prediction/data_daily.csv). 
        Monthly counts of 2021 are also shown for comparason. For predictions at daily level, choose *Daily Prediction* in the sidebar
    """
    )

    st.sidebar.markdown("""
    Fill in 2022 monthly observations (if available) to update the prediction for the rest of the months.
    """)
    m1 = st.sidebar.number_input("Jan.", min_value=0, max_value=500000000, value=None, step=1, help = "Put the total number of scanned receipts in January 2022 here (if available).")
    m2 = st.sidebar.number_input("Feb.", min_value=0, max_value=500000000, value=None, step=1)
    m3 = st.sidebar.number_input("Mar.", min_value=0, max_value=500000000, value=None, step=1)
    m4 = st.sidebar.number_input("Apr.", min_value=0, max_value=500000000, value=None, step=1)
    m5 = st.sidebar.number_input("May.", min_value=0, max_value=500000000, value=None, step=1)
    m6 = st.sidebar.number_input("Jun.", min_value=0, max_value=500000000, value=None, step=1)
    m7 = st.sidebar.number_input("Jul.", min_value=0, max_value=500000000, value=None, step=1)
    m8 = st.sidebar.number_input("Aug.", min_value=0, max_value=500000000, value=None, step=1)
    m9 = st.sidebar.number_input("Sept.", min_value=0, max_value=500000000, value=None, step=1)
    m10 = st.sidebar.number_input("Oct.", min_value=0, max_value=500000000, value=None, step=1)
    m11 = st.sidebar.number_input("Nov.", min_value=0, max_value=500000000, value=None, step=1)
    m12 = st.sidebar.number_input("Dec.", min_value=0, max_value=500000000, value=None, step=1)

    # Display monthly image for 2021
    image_2021 = Image.open('monthly_2021.png')
    st.image(image_2021)

    # Create default monthly image for 2022 -- AR(3) model used for prediction
    image_2022 = Image.open('monthly_2022.png')

    ################ Model & Prediction Update################
    # read and process auxiliary data about holidays
    holiday = pd.read_csv('400_Years_of_Generated_Dates_and_Holidays.csv', header = 0)
    holiday = holiday[holiday['YEAR_FULL'].isin([2021, 2022])]
    holiday.fillna(0, inplace = True)
    holiday['IS_HOLIDAY'] = holiday[['IS_HOLIDAY', 'IS_HOLIDAY_LEAVE']].max(axis=1)
    holiday['date'] = pd.to_datetime(holiday['A_DATE'])
    holiday = holiday[['date', 'IS_HOLIDAY']]

    # read data and join with holidays data
    df = pd.read_csv('data_daily.csv', header = 0)
    df.rename(columns = {"# Date" : "date", "Receipt_Count" : "count"}, inplace = True)
    df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')
    df = df.merge(holiday, 'outer', on = 'date')

    # add variables regarding date
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    df_2021 = df[df['year'] == 2021] 
    df_2022 = df[df['year'] == 2022] 
    df_2022.loc[:,'month'] = df_2022.loc[:,'month'] + 12 # the month variable in a count index starting from 2021-01

    # update model -- prepare data
    predictors = ['month', 'day', 'IS_HOLIDAY']
    X = genX(df_2021, predictors)
    Y = genY(df_2021)

    df_2022_month = df_2022
    #df_2022_month.loc[:,'intercept'] = 1
    df_2022_month = df_2022_month.assign(intercept = np.ones(365))
    df_2022_month = df_2022_month.groupby(['month'])[['intercept'] + predictors].sum()
    X_month = df_2022_month.to_numpy()

    sqrt_nday = np.sqrt(X_month[:,0])[:, np.newaxis]
    X_month_adjusted = X_month / sqrt_nday

    # collect input in sidebar
    input_2022 = [np.nan for i in range(0, 12)]
    if m1 is not None:
        input_2022[0] = m1
    if m2 is not None:
        input_2022[1] = m2
    if m3 is not None:
        input_2022[2] = m3
    if m4 is not None:
        input_2022[3] = m4
    if m5 is not None:
        input_2022[4] = m5 
    if m6 is not None:
        input_2022[5] = m6
    if m7 is not None:
        input_2022[6] = m7
    if m8 is not None:
        input_2022[7] = m8
    if m9 is not None:
        input_2022[8] = m9
    if m10 is not None:
        input_2022[9] = m10   
    if m11 is not None:
        input_2022[10] = m11
    if m12 is not None:
        input_2022[11] = m12 

    # st.table(input_2022)    

    [res, m_observed] = update_prediction(input_2022, X, Y, X_month, X_month_adjusted, sqrt_nday)
    if len(m_observed) == 0:
        st.image(image_2022)
    else:
        fig = plot_prediction(res, m_observed)
        fig.format = "PNG"
        st.pyplot(fig)

    st.components.v1.html(
        """
        <div style="text-align: center"> The black horizontal bars indicate the 95% confidence intervals for the predictions.</div>
        """)


if __name__ == "__main__":
    run()
