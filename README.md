# Cyber-Security-Attack-Detection

## Project Summary

This project aimed to design and deploy a complete machine learning pipeline to predict the **Attack Type** (DDoS, Intrusion, Malware) from a cybersecurity dataset containing 40,000 raw logs and 25 features. 

The workflow included exploratory data analysis, feature engineering, preprocessing, model training, evaluation, and web deployment.

Three models (Logistic Regression, Decision Tree, Random Forest) were trained using a standard preprocessing pipeline (scaling and one-hot encoding). All models achieved performance close to random baseline (~33% accuracy), with tree-based models exhibiting strong overfitting :contentReference[oaicite:1]{index=1}. These results support the hypothesis that the target variable may be independent of the available features.

## Folders & Files

In this repositroy you can find 3 folders :
- **csv** : with examples of csv to use in the web app (see below)
- **data** : with the original dataset
- **models** : with the joblib files of the models used in the project

The main files are :
- **cybersecurity_EDA.ipynb** : the jupiter notebook of the complete Exploratory Data Analysis
- **cybersecurity_main.ipynb** : the jupiter notebook of the complete pipeline of the project
- **app.py** : the streamlit application file
- **GeoLite2-City.mmdb** : the database used to convert IP addresses in geolocation data

## Links 

- The web application to test the prediction can be found [here](https://pythonlabdsticybersecuritygroup10-iavraxx8zuneq2mdkcyna5.streamlit.app/)
