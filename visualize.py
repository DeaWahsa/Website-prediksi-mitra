import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
import numpy as np
import warnings
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime
import os
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from scipy.optimize import minimize
from sklearn.ensemble import VotingClassifier
import json

#HASIL KLASIFIKASI MITRA BAIK SEKALI
# Membaca dataset
data_klasifikasi = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\data prediksi mitra.csv')

# Menghapus baris dengan nilai NaN di 'Label Soft Voting'
data_klasifikasi.dropna(subset=['Label Soft Voting'], inplace=True)

# Membuat filter untuk data dengan label 0
data_label0 = data_klasifikasi[data_klasifikasi['Label Soft Voting'] == 0]

# Menghitung jumlah vendor dengan label 0
vendor_label0_counts = data_label0['Name of Vendor'].value_counts()

# Konversi Series ke dictionary
vendor_label0_counts_dict = vendor_label0_counts.to_dict()

# Buat dictionary dengan informasi yang ingin disimpan
data_to_save = {
    "vendor_label0_counts": vendor_label0_counts_dict
}

# Menyimpan ke dalam file JSON dengan path absolut
path_absolut = "d:\\xampp\\htdocs\\skripsiku\\api\\data\\hasil_klasifikasi_mitra_percobaan.json"
with open(path_absolut, 'w') as json_file:
    json.dump(data_to_save, json_file, indent=4)
