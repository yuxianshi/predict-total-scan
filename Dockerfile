FROM ubuntu:23.10
EXPOSE 8501
RUN apt-get update
RUN apt install -y git
RUN apt-get install -y unzip python3 python3-pip
RUN pip3 install pandas==2.1.0 matplotlib streamlit Pillow scipy --break-system-packages
RUN git clone https://github.com/yuxianshi/predict-total-scan.git
WORKDIR "./predict-total-scan"
CMD ["streamlit","run","Monthly_Prediction.py","--server.enableCORS","false","--server.enableXsrfProtection","false"]
