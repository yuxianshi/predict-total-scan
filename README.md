# Predict Number of Scanned Receipts for Each Month of 2022

Click [here](https://yuxianshi-predict-total-scan-monthly-prediction-wmwam9.streamlit.app/) for the web app of the predictions. 

Specially, this app allows users to update the prediction by filling in the monthly counts for the first few months of 2022 (if available), e.g., the observed counts in January can be used to update the predictions for the rest of the months.

* The process of modeling (three models have been explored) is presented in Modeling.ipynb.
* Dockerfile can be used to build the docker container that established the web app. Copy Dockerfile into your directory and, inside this directory, run the following commands.
```
  docker pull ubuntu:23.10
  docker build . -t app
  docker run -d -p 8501:8501 app
```
  After these steps, you should be able to access the app at http://localhost:8501/ 

## Summary

**Methods**: Three methods have been experimented with, from simpler model to more complex model.
* Linear Regregssion Model (LR) 
    - variable selection conducted here.
* Auto-Regressive Time Series Model AR(3)
    - this model uses the same covariates (predictors) as those selected in the LR model, but is more complex, since it models the correlation of $y_t$ and $y_{t-3}$ ($y_t$ is the count on $t$th day).
* Simple Fully Connected Neural Network Model (NN)
    - this model uses the same covariates (predictors) as those selected in the LR model, but is more complex, since it is non-linear and considers interactions among covariates.
    
**Conclusion**: The AR(3) model demonstrate best test data prediction accuracy, with LR performs slightly suboptimal but similarly. The LR model, however, allows straight forward way for inference (calculating the confidence intervel) at month level and also supports update of the trained model with monthly observations in 2022 (e.g., if at the time this model is used, monthly count of January of 2022 is available). Therefore, the final app include a combination of the AR(3) model and the LR model.   

**Auxillary Data:**
Holidays in USA -- https://www.kaggle.com/datasets/jeremygerdes/us-federal-pay-and-leave-holidays-2004-to-2100-csv
