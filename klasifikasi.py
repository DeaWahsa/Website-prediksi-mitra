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

# Menyembunyikan semua peringatan
warnings.simplefilter(action="ignore", category=Warning)


def function_klasifikasi(data):
    # Baca file GAPED.csv
    prediksi_data = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\combined.csv')
    features_data_GAPED = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\GAPED.csv')
    features_data_SMILE = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\SMILE.csv')
    features_data_Witel = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\Witel.csv')
    labelling = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\labelling.csv')
    
    #GABUNG DATA
    kumpulan_data =pd.concat([features_data_GAPED, features_data_SMILE, features_data_Witel], axis='columns')
    
    # Print contoh data kosong dalam bentuk tabel
    kumpulan_data = kumpulan_data.dropna()

    # DATA TRANSFORMATION
    # Mengubah kolom tanggal awal dan tanggal akhir menjadi tipe data datetime
    kumpulan_data['PO Date'] = pd.to_datetime(kumpulan_data['PO Date'])
    kumpulan_data['Delivery Date'] = pd.to_datetime(kumpulan_data['Delivery Date'])

    # Menghitung durasi hari antara tanggal awal dan tanggal akhir
    kumpulan_data['Durasi Kontrak'] = (
        kumpulan_data['Delivery Date'] - kumpulan_data['PO Date']).dt.days

    # Mengubah kolom tanggal awal dan tanggal akhir menjadi tipe data datetime
    kumpulan_data['PO Date'] = pd.to_datetime(kumpulan_data['PO Date'])
    kumpulan_data['Doc Date GR'] = pd.to_datetime(kumpulan_data['Doc Date GR'])

    # Menghitung durasi hari antara tanggal awal dan tanggal akhir
    kumpulan_data['Durasi Penyelesaian'] = (
        kumpulan_data['Doc Date GR'] - kumpulan_data['PO Date']).dt.days

    # Mengonversi kolom 'Local_Amount' ke tipe data string
    kumpulan_data['Local Amount'] = kumpulan_data['Local Amount'].astype(str)

    # Menghilangkan koma dari kolom 'Local_Amount'
    kumpulan_data['Local Amount'] = kumpulan_data['Local Amount'].str.replace(
        ',', '').astype(float)

    # Multiply 'jumlah_projek' with 'nilai_projek' and store the result in a new column 'total_nilai'
    kumpulan_data['nilai projek per LoP'] = kumpulan_data['Jumlah Projek'] * \
        kumpulan_data['Local Amount']

    # Calculate the average 'nilai projek per LoP' per banyaknya baris data
    average_per_banyaknya_baris_data = kumpulan_data['nilai projek per LoP'].mean()

    # Create a new column 'Sesuai' based on the condition
    kumpulan_data['Kategori Anggaran Projek'] = kumpulan_data['nilai projek per LoP'].apply(
        lambda x: 'anggaran sesuai' if x <= average_per_banyaknya_baris_data else 'anggaran tidak sesuai')

    # Create a new column 'Kecepatan' based on the condition
    kumpulan_data['Kategori Durasi Projek'] = ''
    for index, row in kumpulan_data.iterrows():
        if row['Durasi Penyelesaian'] <= row['Durasi Kontrak']:
            kumpulan_data.loc[index, 'Kategori Durasi Projek'] = 'pengerjaan cepat'
        else:
            kumpulan_data.loc[index,
                            'Kategori Durasi Projek'] = 'pengerjaan lambat'

    kumpulan_data['Nilai Performansi KHS'] = kumpulan_data['Nilai Performansi KHS'].str.lower()

    #SIMPAN FILE KUMPULAN DATA
    path_direktori = r"D:\xampp\htdocs\skripsiku\api\data"
    kumpulan_data_csv = "kumpulan_data.csv"
    path_file_csv = os.path.join(path_direktori, kumpulan_data_csv)
    kumpulan_data.to_csv(path_file_csv, index=False)

    data_preprocessing = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\kumpulan_data.csv')

    #MULTILABEL BINARIZER
    # Mengganti nilai NaN dengan nilai default atau nilai yang sesuai dengan konteks data Anda
    data_preprocessing['Nilai Performansi KHS'].fillna('default', inplace=True)
    data_preprocessing['Alker/Salker'].fillna('default', inplace=True)
    data_preprocessing['Stok Material'].fillna('default', inplace=True)
    data_preprocessing['Jumlah Team'].fillna('default', inplace=True)
    data_preprocessing['Kerapihan'].fillna('default', inplace=True)
    data_preprocessing['Kategori Durasi Projek'].fillna('default', inplace=True)
    data_preprocessing['Kategori Anggaran Projek'].fillna('default', inplace=True)

    # Melakukan fit_transform menggunakan LabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(data_preprocessing['Nilai Performansi KHS'])
    mlb.fit_transform(data_preprocessing['Alker/Salker'])
    mlb.fit_transform(data_preprocessing['Stok Material'])
    mlb.fit_transform(data_preprocessing['Jumlah Team'])
    mlb.fit_transform(data_preprocessing['Kerapihan'])
    mlb.fit_transform(data_preprocessing['Kategori Durasi Projek'])
    mlb.fit_transform(data_preprocessing['Kategori Anggaran Projek'])

    data_preprocessing['Nilai Performansi KHS']=data_preprocessing['Nilai Performansi KHS'].str.split(',\s*')
    data_preprocessing['Alker/Salker']=data_preprocessing['Alker/Salker'].str.split(',\s*')
    data_preprocessing['Stok Material']=data_preprocessing['Stok Material'].str.split(',\s*')
    data_preprocessing['Jumlah Team']=data_preprocessing['Jumlah Team'].str.split(',\s*')
    data_preprocessing['Kerapihan']=data_preprocessing['Kerapihan'].str.split(',\s*')
    data_preprocessing['Kategori Durasi Projek']=data_preprocessing['Kategori Durasi Projek'].str.split(',\s*')
    data_preprocessing['Kategori Anggaran Projek']=data_preprocessing['Kategori Anggaran Projek'].str.split(',\s*')

    mlb = MultiLabelBinarizer()

    mlb.fit(data_preprocessing['Nilai Performansi KHS'])
    mlb.fit(data_preprocessing['Alker/Salker'])
    mlb.fit(data_preprocessing['Stok Material'])
    mlb.fit(data_preprocessing['Jumlah Team'])
    mlb.fit(data_preprocessing['Kerapihan'])
    mlb.fit(data_preprocessing['Kategori Durasi Projek'])
    mlb.fit(data_preprocessing['Kategori Anggaran Projek'])

    mlb.classes_

    data_preprocessing['Nilai Performansi KHS'].explode().unique()
    data_preprocessing['Alker/Salker'].explode().unique()
    data_preprocessing['Stok Material'].explode().unique()
    data_preprocessing['Jumlah Team'].explode().unique()
    data_preprocessing['Kerapihan'].explode().unique()
    data_preprocessing['Kategori Durasi Projek'].explode().unique()
    data_preprocessing['Kategori Anggaran Projek'].explode().unique()

    mlb.transform(data_preprocessing['Nilai Performansi KHS'])
    mlb.transform(data_preprocessing['Alker/Salker'])
    mlb.transform(data_preprocessing['Stok Material'])
    mlb.transform(data_preprocessing['Jumlah Team'])
    mlb.transform(data_preprocessing['Kerapihan'])
    mlb.transform(data_preprocessing['Kategori Durasi Projek'])
    mlb.transform(data_preprocessing['Kategori Anggaran Projek'])

    a1 = pd.DataFrame(mlb.fit_transform(data_preprocessing['Nilai Performansi KHS']), columns=mlb.classes_)
    a2 = pd.DataFrame(mlb.fit_transform(data_preprocessing['Alker/Salker']), columns=mlb.classes_)
    a3 = pd.DataFrame(mlb.fit_transform(data_preprocessing['Stok Material']), columns=mlb.classes_)
    a4 = pd.DataFrame(mlb.fit_transform(data_preprocessing['Jumlah Team']), columns=mlb.classes_)
    a5 = pd.DataFrame(mlb.fit_transform(data_preprocessing['Kerapihan']), columns=mlb.classes_)
    a6 = pd.DataFrame(mlb.fit_transform(data_preprocessing['Kategori Durasi Projek']), columns=mlb.classes_)
    a7 = pd.DataFrame(mlb.fit_transform(data_preprocessing['Kategori Anggaran Projek']), columns=mlb.classes_)

    binarizer=pd.concat([data_preprocessing, a1, a2, a3, a4, a5, a6, a7], axis=1)

    multilabel_binarizer = binarizer.drop(['Nilai Performansi KHS','Alker/Salker','Stok Material','Jumlah Team','Kerapihan','Kategori Durasi Projek','Kategori Anggaran Projek'], axis=1)

    #SIMPAN FILE BINARIZER DATA
    path_direktori = r"D:\xampp\htdocs\skripsiku\api\data"
    multilabel_binarizer_csv = "binarizer_data.csv"
    path_file_csv = os.path.join(path_direktori, multilabel_binarizer_csv)
    multilabel_binarizer.to_csv(path_file_csv, index=False)

    #KLASIFIKASI
    binarizer_labelling =pd.concat([multilabel_binarizer, labelling], axis='columns')
    data_mitra = binarizer_labelling.drop(['Name of Vendor','Short Text','PO Date','Delivery Date','Doc Date GR','pengerjaan cepat','pengerjaan lambat', 'anggaran sesuai',	'anggaran tidak sesuai'], axis=1)

    #SIMPAN FILE TELKOM MITRA
    path_direktori = r"D:\xampp\htdocs\skripsiku\api\data"
    data_mitra_csv = "telkom_mitra.csv"
    path_file_csv = os.path.join(path_direktori, data_mitra_csv)
    data_mitra.to_csv(path_file_csv, index=False)
    
    #SOFT VOTING CLASSIFIER
    # Memisahkan fitur (X) dan label (y)
    X = data_mitra.drop("Label", axis=1).values
    y = data_mitra['Label'].values

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Define three classification models
    xgb_clf = XGBClassifier(objective='multi:softmax', num_class=4)
    adaboost_clf = AdaBoostClassifier()
    gb_clf = GradientBoostingClassifier()

    # Train the models on the training data
    xgb_clf.fit(X_train, y_train)
    adaboost_clf.fit(X_train, y_train)
    gb_clf.fit(X_train, y_train)

    # Define the ensemble model for soft voting
    classifiers = [xgb_clf, adaboost_clf, gb_clf]

    # Function for soft voting with specified weights
    def soft_voting(classifiers, X, weights):
        num_classes = len(np.unique(y))
        weighted_probabilities = np.zeros((len(X), num_classes))

        for i, clf in enumerate(classifiers):
            probabilities = clf.predict_proba(X)
            weighted_probabilities += probabilities * weights[i]

        final_predictions = np.argmax(weighted_probabilities, axis=1)

        return final_predictions

    # Objective function for the Firefly algorithm
    def objective_function_firefly(weights):
        predictions = soft_voting(classifiers, X_test, weights)
        return -accuracy_score(y_test, predictions)  # Invert accuracy for maximization

    # Customized Firefly algorithm parameters
    num_fireflies = 40
    max_generation_firefly = 230
    alpha = 0.35
    beta = 0.6

    # Lower and upper bounds for Firefly weights
    lower_bound_firefly = 0.0
    upper_bound_firefly = 1.0

    best_accuracy_firefly = 0.0
    best_weights_firefly = None

    # Perform optimization with the Firefly algorithm
    for _ in range(10):
        initial_weights = np.random.uniform(lower_bound_firefly, upper_bound_firefly, len(classifiers))
        result_firefly = minimize(objective_function_firefly, initial_weights, method='Nelder-Mead', options={'maxiter': max_generation_firefly})
        if -result_firefly.fun > best_accuracy_firefly:
            best_accuracy_firefly = -result_firefly.fun
            best_weights_firefly = result_firefly.x

    # Normalize the weights so they sum to 1
    best_weights_firefly /= np.sum(best_weights_firefly)

    # Use the best weights for prediction
    final_predictions = soft_voting(classifiers, X_test, best_weights_firefly)
    
    #PREDIKSI DATA BARU
    # DATA TRANSFORMATION
    # Mengubah kolom tanggal awal dan tanggal akhir menjadi tipe data datetime
    prediksi_data['PO Date'] = pd.to_datetime(prediksi_data['PO Date'])
    prediksi_data['Delivery Date'] = pd.to_datetime(prediksi_data['Delivery Date'])

    # Menghitung durasi hari antara tanggal awal dan tanggal akhir
    prediksi_data['Durasi Kontrak'] = (
        prediksi_data['Delivery Date'] - prediksi_data['PO Date']).dt.days

    # Mengubah kolom tanggal awal dan tanggal akhir menjadi tipe data datetime
    prediksi_data['PO Date'] = pd.to_datetime(prediksi_data['PO Date'])
    prediksi_data['Doc Date GR'] = pd.to_datetime(prediksi_data['Doc Date GR'])

    # Menghitung durasi hari antara tanggal awal dan tanggal akhir
    prediksi_data['Durasi Penyelesaian'] = (
        prediksi_data['Doc Date GR'] - prediksi_data['PO Date']).dt.days

    # Mengonversi kolom 'Local_Amount' ke tipe data string
    prediksi_data['Local Amount'] = prediksi_data['Local Amount'].astype(str)

    # Menghilangkan koma dari kolom 'Local_Amount'
    prediksi_data['Local Amount'] = prediksi_data['Local Amount'].str.replace(
        ',', '').astype(float)

    # Multiply 'jumlah_projek' with 'nilai_projek' and store the result in a new column 'total_nilai'
    prediksi_data['nilai projek per LoP'] = prediksi_data['Jumlah Projek'] * \
        prediksi_data['Local Amount']

    # Calculate the average 'nilai projek per LoP' per banyaknya baris data
    average_per_banyaknya_baris_data = prediksi_data['nilai projek per LoP'].mean()

    # Create a new column 'Sesuai' based on the condition
    prediksi_data['Kategori Anggaran Projek'] = prediksi_data['nilai projek per LoP'].apply(
        lambda x: 'anggaran sesuai' if x <= average_per_banyaknya_baris_data else 'anggaran tidak sesuai')

    # Create a new column 'Kecepatan' based on the condition
    prediksi_data['Kategori Durasi Projek'] = ''
    for index, row in prediksi_data.iterrows():
        if row['Durasi Penyelesaian'] <= row['Durasi Kontrak']:
            prediksi_data.loc[index, 'Kategori Durasi Projek'] = 'pengerjaan cepat'
        else:
            prediksi_data.loc[index,
                            'Kategori Durasi Projek'] = 'pengerjaan lambat'

    prediksi_data['Nilai Performansi KHS'] = prediksi_data['Nilai Performansi KHS'].str.lower()

    #SIMPAN FILE 
    path_direktori = r"D:\xampp\htdocs\skripsiku\api\data"
    prediksi_data_csv = "olah data prediksi.csv"
    path_file_csv = os.path.join(path_direktori, prediksi_data_csv)
    prediksi_data.to_csv(path_file_csv, index=False)

    data_preprocessing2 = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\olah data prediksi.csv')

    #MULTILABEL BINARIZER
    # Mengganti nilai NaN dengan nilai default atau nilai yang sesuai dengan konteks data Anda
    data_preprocessing2['Nilai Performansi KHS'].fillna('default', inplace=True)
    data_preprocessing2['Alker/Salker'].fillna('default', inplace=True)
    data_preprocessing2['Stok Material'].fillna('default', inplace=True)
    data_preprocessing2['Jumlah Team'].fillna('default', inplace=True)
    data_preprocessing2['Kerapihan'].fillna('default', inplace=True)
    data_preprocessing2['Kategori Durasi Projek'].fillna('default', inplace=True)
    data_preprocessing2['Kategori Anggaran Projek'].fillna('default', inplace=True)

    # Melakukan fit_transform menggunakan LabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(data_preprocessing2['Nilai Performansi KHS'])
    mlb.fit_transform(data_preprocessing2['Alker/Salker'])
    mlb.fit_transform(data_preprocessing2['Stok Material'])
    mlb.fit_transform(data_preprocessing2['Jumlah Team'])
    mlb.fit_transform(data_preprocessing2['Kerapihan'])
    mlb.fit_transform(data_preprocessing2['Kategori Durasi Projek'])
    mlb.fit_transform(data_preprocessing2['Kategori Anggaran Projek'])

    data_preprocessing2['Nilai Performansi KHS']=data_preprocessing2['Nilai Performansi KHS'].str.split(',\s*')
    data_preprocessing2['Alker/Salker']=data_preprocessing2['Alker/Salker'].str.split(',\s*')
    data_preprocessing2['Stok Material']=data_preprocessing2['Stok Material'].str.split(',\s*')
    data_preprocessing2['Jumlah Team']=data_preprocessing2['Jumlah Team'].str.split(',\s*')
    data_preprocessing2['Kerapihan']=data_preprocessing2['Kerapihan'].str.split(',\s*')
    data_preprocessing2['Kategori Durasi Projek']=data_preprocessing2['Kategori Durasi Projek'].str.split(',\s*')
    data_preprocessing2['Kategori Anggaran Projek']=data_preprocessing2['Kategori Anggaran Projek'].str.split(',\s*')

    mlb = MultiLabelBinarizer()

    mlb.fit(data_preprocessing2['Nilai Performansi KHS'])
    mlb.fit(data_preprocessing2['Alker/Salker'])
    mlb.fit(data_preprocessing2['Stok Material'])
    mlb.fit(data_preprocessing2['Jumlah Team'])
    mlb.fit(data_preprocessing2['Kerapihan'])
    mlb.fit(data_preprocessing2['Kategori Durasi Projek'])
    mlb.fit(data_preprocessing2['Kategori Anggaran Projek'])

    mlb.classes_

    data_preprocessing2['Nilai Performansi KHS'].explode().unique()
    data_preprocessing2['Alker/Salker'].explode().unique()
    data_preprocessing2['Stok Material'].explode().unique()
    data_preprocessing2['Jumlah Team'].explode().unique()
    data_preprocessing2['Kerapihan'].explode().unique()
    data_preprocessing2['Kategori Durasi Projek'].explode().unique()
    data_preprocessing2['Kategori Anggaran Projek'].explode().unique()

    mlb.transform(data_preprocessing2['Nilai Performansi KHS'])
    mlb.transform(data_preprocessing2['Alker/Salker'])
    mlb.transform(data_preprocessing2['Stok Material'])
    mlb.transform(data_preprocessing2['Jumlah Team'])
    mlb.transform(data_preprocessing2['Kerapihan'])
    mlb.transform(data_preprocessing2['Kategori Durasi Projek'])
    mlb.transform(data_preprocessing2['Kategori Anggaran Projek'])

    a1 = pd.DataFrame(mlb.fit_transform(data_preprocessing2['Nilai Performansi KHS']), columns=mlb.classes_)
    a2 = pd.DataFrame(mlb.fit_transform(data_preprocessing2['Alker/Salker']), columns=mlb.classes_)
    a3 = pd.DataFrame(mlb.fit_transform(data_preprocessing2['Stok Material']), columns=mlb.classes_)
    a4 = pd.DataFrame(mlb.fit_transform(data_preprocessing2['Jumlah Team']), columns=mlb.classes_)
    a5 = pd.DataFrame(mlb.fit_transform(data_preprocessing2['Kerapihan']), columns=mlb.classes_)
    a6 = pd.DataFrame(mlb.fit_transform(data_preprocessing2['Kategori Durasi Projek']), columns=mlb.classes_)
    a7 = pd.DataFrame(mlb.fit_transform(data_preprocessing2['Kategori Anggaran Projek']), columns=mlb.classes_)

    binarizer=pd.concat([data_preprocessing2, a1, a2, a3, a4, a5, a6, a7], axis=1)

    multilabel_binarizer = binarizer.drop(['Nilai Performansi KHS','Alker/Salker','Stok Material','Jumlah Team','Kerapihan','Kategori Durasi Projek','Kategori Anggaran Projek'], axis=1)

    #SIMPAN FILE BINARIZER DATA
    path_direktori = r"D:\xampp\htdocs\skripsiku\api\data"
    multilabel_binarizer_csv = "binarizer_data_prediksi.csv"
    path_file_csv = os.path.join(path_direktori, multilabel_binarizer_csv)
    multilabel_binarizer.to_csv(path_file_csv, index=False)

    #KLASIFIKASI
    data_mitra2 = multilabel_binarizer.drop(['Name of Vendor','Short Text','PO Date','Delivery Date','Doc Date GR','pengerjaan cepat','pengerjaan lambat', 'anggaran sesuai',	'anggaran tidak sesuai'], axis=1)
    
    #SIMPAN FILE TELKOM MITRA
    path_direktori = r"D:\xampp\htdocs\skripsiku\api\data"
    data_mitra2_csv = "prediksi data.csv"
    path_file_csv = os.path.join(path_direktori, data_mitra2_csv)
    data_mitra2.to_csv(path_file_csv, index=False)
    
    # Melakukan prediksi menggunakan soft voting
    hasil_prediksi = soft_voting(classifiers, data_mitra2, best_weights_firefly)

    # Tambahkan hasil prediksi ke DataFrame data baru
    data_mitra2['Label Soft Voting'] = hasil_prediksi

    #SIMPAN FILE TELKOM MITRA
    path_direktori = r"D:\xampp\htdocs\skripsiku\api\data"
    hasil_prediksi_csv = "prediksi data baru.csv"
    path_file_csv = os.path.join(path_direktori, prediksi_data_csv)
    data_mitra2.to_csv(path_file_csv, index=False)
    
    # Tambahkan kolom "Label Soft Voting" ke DataFrame
    prediksi_data['Label Soft Voting'] = hasil_prediksi
    
    #SIMPAN FILE TELKOM MITRA
    path_direktori = r"D:\xampp\htdocs\skripsiku\api\data"
    prediksi_data_csv = "data prediksi mitra.csv"
    path_file_csv = os.path.join(path_direktori, prediksi_data_csv)
    prediksi_data.to_csv(path_file_csv, index=False)
    
    #HASIL KLASIFIKASI SEMUA MITRA
    # Membaca dataset
    data_klasifikasi = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\data prediksi mitra.csv')

    soft_voting = VotingClassifier(estimators=[('xgb', xgb_clf), ('adaboost', adaboost_clf), ('gb', gb_clf)], voting='soft')

    # Encode nama vendor
    encoder_vendor = LabelEncoder()
    data_klasifikasi['Encoded Name of Vendor'] = encoder_vendor.fit_transform(data_klasifikasi['Name of Vendor'])
    # Simpan pemetaan dari encoded label ke nama asli
    inverse_label_mapping = {encoded_label: original_label for original_label, encoded_label in zip(data_klasifikasi['Name of Vendor'], data_klasifikasi['Encoded Name of Vendor'])}

    # Hapus baris dengan nilai NaN di kolom 'Label Soft Voting'
    data_klasifikasi.dropna(subset=['Label Soft Voting'], inplace=True)

    # Menyiapkan data latih
    X_train = data_klasifikasi[['Encoded Name of Vendor']]
    y_train = data_klasifikasi['Label Soft Voting']

    # Melatih voting classifier
    soft_voting.fit(X_train, y_train)

    # Grouping dan unstacking jumlah label per vendor
    vendor_label_counts = data_klasifikasi.groupby(['Encoded Name of Vendor', 'Label Soft Voting']).size().unstack(fill_value=0)

    # Ganti index yang di-encode dengan nama asli vendor
    vendor_label_counts.rename(index=inverse_label_mapping, inplace=True)

    # Konversi DataFrame ke dictionary
    vendor_label_counts_dict = vendor_label_counts.to_dict('index')

    # Menghitung total label
    totals = {
        "label_0": int(vendor_label_counts.get(0, pd.Series()).sum()),
        "label_1": int(vendor_label_counts.get(1, pd.Series()).sum()),
        "label_2": int(vendor_label_counts.get(2, pd.Series()).sum()),
        "label_3": int(vendor_label_counts.get(3, pd.Series()).sum())
    }

    # Buat dictionary dengan semua informasi yang ingin disimpan
    current_year = datetime.now().year
    data_to_save = {
        "vendor_label_counts": vendor_label_counts_dict,
        "totals": totals
    }

    # Menyimpan ke dalam file JSON
    path_to_save = os.path.join('d:\\xampp\\htdocs\\skripsiku\\api\\data', f'data_klasifikasi_mitra_{current_year}.json')
    with open(path_to_save, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)
    
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

    current_year = datetime.now().year
     
    # Menyimpan ke dalam file JSON
    path_to_save = f"d:\\xampp\\htdocs\\skripsiku\\api\\data\\hasil_klasifikasi_mitra_baiksekali_{current_year}.json"
    with open(path_to_save, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)

    #HASIL KLASIFIKASI MITRA BAIK
    # Membaca dataset
    data_klasifikasi = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\data prediksi mitra.csv')

    # Menghapus baris dengan nilai NaN di 'Label Soft Voting'
    data_klasifikasi.dropna(subset=['Label Soft Voting'], inplace=True)

    # Membuat filter untuk data dengan label 0
    data_label0 = data_klasifikasi[data_klasifikasi['Label Soft Voting'] == 1]

    # Menghitung jumlah vendor dengan label 0
    vendor_label0_counts = data_label0['Name of Vendor'].value_counts()

    # Konversi Series ke dictionary
    vendor_label0_counts_dict = vendor_label0_counts.to_dict()

    # Buat dictionary dengan informasi yang ingin disimpan
    data_to_save = {
        "vendor_label0_counts": vendor_label0_counts_dict
    }

    current_year = datetime.now().year
     
    # Menyimpan ke dalam file JSON
    path_to_save = f"d:\\xampp\\htdocs\\skripsiku\\api\\data\\hasil_klasifikasi_mitra_baik_{current_year}.json"
    with open(path_to_save, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)
        
    #HASIL KLASIFIKASI KELAS CUKUP
    # Membaca dataset
    data_klasifikasi = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\data prediksi mitra.csv')

    # Menghapus baris dengan nilai NaN di 'Label Soft Voting'
    data_klasifikasi.dropna(subset=['Label Soft Voting'], inplace=True)

    # Membuat filter untuk data dengan label 0
    data_label0 = data_klasifikasi[data_klasifikasi['Label Soft Voting'] == 2]

    # Menghitung jumlah vendor dengan label 0
    vendor_label0_counts = data_label0['Name of Vendor'].value_counts()

    # Konversi Series ke dictionary
    vendor_label0_counts_dict = vendor_label0_counts.to_dict()

    # Buat dictionary dengan informasi yang ingin disimpan
    data_to_save = {
        "vendor_label0_counts": vendor_label0_counts_dict
    }

    current_year = datetime.now().year
     
    # Menyimpan ke dalam file JSON
    path_to_save = f"d:\\xampp\\htdocs\\skripsiku\\api\\data\\hasil_klasifikasi_mitra_cukup_{current_year}.json"
    with open(path_to_save, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)
    
    #HASIL KLASIFIKASI MITRA BURUK
    # Membaca dataset
    data_klasifikasi = pd.read_csv('d:\\xampp\\htdocs\\skripsiku\\api\\data\\data prediksi mitra.csv')

    # Menghapus baris dengan nilai NaN di 'Label Soft Voting'
    data_klasifikasi.dropna(subset=['Label Soft Voting'], inplace=True)

    # Membuat filter untuk data dengan label 0
    data_label0 = data_klasifikasi[data_klasifikasi['Label Soft Voting'] == 3]

    # Menghitung jumlah vendor dengan label 0
    vendor_label0_counts = data_label0['Name of Vendor'].value_counts()

    # Konversi Series ke dictionary
    vendor_label0_counts_dict = vendor_label0_counts.to_dict()

    # Buat dictionary dengan informasi yang ingin disimpan
    data_to_save = {
        "vendor_label0_counts": vendor_label0_counts_dict
    }

    current_year = datetime.now().year
     
    # Menyimpan ke dalam file JSON
    path_to_save = f"d:\\xampp\\htdocs\\skripsiku\\api\\data\\hasil_klasifikasi_mitra_buruk_{current_year}.json"
    with open(path_to_save, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)
    return {}
 








