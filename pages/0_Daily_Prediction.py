import streamlit as st
from PIL import Image

def plot_daily_prediction():
    image = Image.open('daily.png')
    st.image(image, caption='')

    st.components.v1.html(
        """
        <div style="text-align: center"> The shaded area in the 2022 plot indicates the 95% confidence interval for daily count predictions.</div>
        """)

st.set_page_config(page_title="Daily Prediction")
st.write("# Daily Prediction for 2022")
st.markdown("""
            Daily count of scanned receipts is predicted for 2022 using 
        [daily observations in 2021](https://fetch-hiring.s3.amazonaws.com/machine-learning-engineer/receipt-count-prediction/data_daily.csv). 
        Daily counts of 2021 are also shown for comparason.
            """)
st.sidebar.header("Daily Prediction")

# st.sidebar.markdown("The shaded area in the 2022 plot indicates the 95% confidence interval of daily counts.")
plot_daily_prediction()
