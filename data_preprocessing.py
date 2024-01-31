import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(raw_data_path):
    data = pd.read_csv(raw_data_path)
    data = data.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)
    data = data.fillna(method='ffill')
    data.to_csv('preprocess.csv',index=False)
    


preprocess_data(r'D:\VED\model_tracking\ChanDarren_RaiTaran_Lab2a.csv')
