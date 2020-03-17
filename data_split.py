from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np

def Train_Test_Val_Split(data , test_data_fraction = 0.3, val_data_fraction = 0.1) :
    
    mlb = MultiLabelBinarizer()
    data_genres_one_hot_encoding = mlb.fit_transform(data['genres'])
    Label_names = mlb.classes_
    data_genres_one_hot_encoding = pd.DataFrame(data_genres_one_hot_encoding, columns = mlb.classes_)
    Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, data_genres_one_hot_encoding, test_size = test_data_fraction)
    Data_train, Data_val, Labels_train, Labels_val = train_test_split(Data_train, Labels_train, test_size = val_data_fraction)
    
    Labels_train = torch.tensor(Labels_train.values)
    Labels_test = torch.tensor(Labels_test.values)
    Labels_val = torch.tensor(Labels_val.values)

    Data_train = Data_train.reset_index(drop=True)
    Data_test = Data_test.reset_index(drop=True)
    Data_val = Data_val.reset_index(drop=True)
    

    return (Data_train, Data_test, Data_val, Labels_train, Labels_test, Labels_val, Label_names)