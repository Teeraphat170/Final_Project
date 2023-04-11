import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from datetime import datetime

# ForTest_Prediction
# TestX = pd.read_csv('ReadCSV_OneFile.csv')
# TestX.drop(TestX.columns[0], axis=1, inplace=True)
# # print(TestX)
# model = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale.csv')
# # model = pd.read_csv('CSV_XLSX2/Ready_For_Model_Scale.csv')

# model = model.iloc[: , 1:]

# TestX = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale_Sliding_Windows.csv')
# TestX = TestX.iloc[: , 1:]

def predict(TestX):
    # time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    time = datetime.today().strftime('%H:%M:%S')
    # filename = filename
    df = TestX
    # print(df)

    #supervised Random
    # load_clf = pickle.load(open('Model/New_Model_NoScale_V2.pkl', 'rb'))
    # load_clf = pickle.load(open('Model/New_Model_V3.pkl', 'rb'))
    # load_clf = pickle.load(open('Model/MT_clf.pkl', 'rb')) old model
    # load_clf = pickle.load(open('Model2/Random_noScale.pkl', 'rb')) #####undetectable
    # load_clf = pickle.load(open('Model2/Random_Scale.pkl', 'rb'))
    # load_clf = pickle.load(open('Model2/Random_NewOk_Scale.pkl', 'rb')) #####undetectable
    # load_clf = pickle.load(open('Model2/Random_NewOk_NoScale.pkl', 'rb')) #####undetectable
    # load_clf = pickle.load(open('Model3/Random_Scale.pkl', 'rb'))
    load_clf = pickle.load(open('Model3/Random_NoScale.pkl', 'rb')) #---------------------ใช้อันนี้เป็นหลัก---------------------#
    # load_clf = pickle.load(open('Model3/Random_NoScale_2.pkl', 'rb'))
    # load_clf = pickle.load(open('Model3/Random_NoScale_fs06.pkl', 'rb'))
    # load_clf = pickle.load(open('Model3/Random_NoScale_fs04.pkl', 'rb'))
    # load_clf = pickle.load(open('Model3/Random_NoScale_fs05.pkl', 'rb'))
    # load_clf = pickle.load(open('Model3/Random_NoScale_SameFeatureScale.pkl', 'rb'))

    #unsupervised k-mean
    # load_clf = pickle.load(open("Model/k-mean.pkl", "rb"))
    # load_clf = pickle.load(open("Model/k-mean_NoScale.pkl", "rb")) # OK = 0 , NG = 1
    # load_clf = pickle.load(open("Model/k-mean_SlidingWindow.pkl", "rb")) # OK = 0 , NG = 1
    # load_clf = pickle.load(open("Model/k-mean_SlidingWindow_NewOk.pkl", "rb")) # OK = 0 , NG = 1
    # load_clf = pickle.load(open("Model2/k-mean_Scale.pkl", "rb")) # OK = 1 , NG = 0
    # load_clf = pickle.load(open("Model2/k-mean_NoScale.pkl", "rb")) # OK = 1 , NG = 0 #---------------------ใช้อันนี้เป็นหลัก---------------------#
    # load_clf = pickle.load(open("Model2/k-mean_NewOk_NoScale.pkl", "rb")) # OK = 0 , NG = 1 #####undetectable
    # load_clf = pickle.load(open("Model2/k-mean_NewOk_Scale.pkl", "rb")) # OK = 0 , NG = 1 #####undetectable
    

    #unsupervised IsolationForest
    # load_clf = pickle.load(open("Model2/IsolationForest.pkl", "rb")) 
    # load_clf = pickle.load(open("Model2/IsolationForest_1.pkl", "rb")) #####Not Good
    # load_clf = pickle.load(open("Model2/IsolationForest_2.pkl", "rb")) #####undetectable
    # load_clf = pickle.load(open("Model2/IsolationForest_3.pkl", "rb")) 
    # load_clf = pickle.load(open("Model2/IsolationForest_4.pkl", "rb")) 
    # load_clf = pickle.load(open("Model2/IsolationForest_6.pkl", "rb"))
    # load_clf = pickle.load(open("Model2/IsolationForest_7.pkl", "rb")) #---------------------ใช้อันนี้เป็นหลัก---------------------#
    # load_clf = pickle.load(open("Model2/IsolationForest_8.pkl", "rb")) #---------------------ใช้อันนี้เป็นหลัก_1---------------------#
    # load_clf = pickle.load(open("Model2/IsolationForest_sliding.pkl", "rb")) #---------------------ใช้อันนี้เป็นหลัก_2---------------------#
    # load_clf = pickle.load(open("Model2/IsolationForest_scale.pkl", "rb"))
    # load_clf = pickle.load(open("TestModel/IsolationForest_Test1.pkl", "rb")) 
    # load_clf = pickle.load(open("TestModel/IsolationForest_Test3.pkl", "rb"))
    # iso = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.12),max_features=1.0, 
    #                       bootstrap=False, n_jobs=-1, random_state=42, verbose=0).fit(model) 
    # iso = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.17),max_features=1.0, 
    #                   bootstrap=False, n_jobs=-1, random_state=42, verbose=0).fit(model)
    # load_clf = iso

    #unsupervised one_class_svm
    # load_clf = pickle.load(open("Model2/one_class_svm1.pkl", "rb")) #####undetectable
    # load_clf = pickle.load(open("Model2/one_class_svm2.pkl", "rb")) #####undetectable
    # load_clf = pickle.load(open("Model2/one_class_svm3.pkl", "rb")) #####undetectable
    # load_clf = pickle.load(open("Model2/one_class_svm5.pkl", "rb"))
    # load_clf = pickle.load(open("Model2/one_class_svm7.pkl", "rb"))
    # load_clf = pickle.load(open("Model2/one_class_svm8.pkl", "rb")) #---------------------ใช้อันนี้เป็นหลัก---------------------#
    # load_clf = pickle.load(open("Model2/one_class_svm_sliding.pkl", "rb")) #---------------------ใช้อันนี้เป็นหลัก_2---------------------#
    # one_class_svm = OneClassSVM(nu = 0.01, kernel = 'rbf', gamma = 0.0000001).fit(model) #default
    # one_class_svm = OneClassSVM(kernel='rbf', degree=5, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=False, 
    #                             cache_size=200, verbose=False, max_iter=0).fit(model)
    # one_class_svm = OneClassSVM(kernel='rbf', degree=5, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=False, 
    #                             cache_size=200, verbose=False, max_iter=0).fit(model)
    # load_clf = one_class_svm

    #unsupervised LOF
    # load_clf = pickle.load(open("Model2/LOF.pkl", "rb")) #####undetectable
    # load_clf = pickle.load(open("Model2/LOF1.pkl", "rb")) #####undetectable
    # clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    # load_clf = pickle.load(open("Model2/LOF3.pkl", "rb"))
    # load_clf = pickle.load(open("Model2/LOF4.pkl", "rb"))
    # load_clf = pickle.load(open("Model2/LOF5.pkl", "rb")) #---------------------ใช้อันนี้เป็นหลัก_1---------------------#
    # load_clf = pickle.load(open("Model2/LOF_sliding.pkl", "rb")) #---------------------ใช้อันนี้เป็นหลัก_2---------------------#
    # clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True, algorithm='auto').fit(model) #default
    # clf = LocalOutlierFactor(n_neighbors=5, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, 
    #                          contamination=0.0001, novelty=True, n_jobs=None).fit(model)
    # clf = LocalOutlierFactor(n_neighbors=15, algorithm='auto', leaf_size=30, metric='minkowski', p=2, 
    #                      metric_params=None, contamination=0.19, novelty=True, n_jobs=None).fit(model)
    # load_clf = clf

    #Supervised #Random
    prediction = load_clf.predict(df) 
    prediction_proba = load_clf.predict_proba(df) 

    #Unsupervised #K-mean
    # prediction = load_clf.predict(df) 

    # IsolationForest
    # prediction = load_clf.predict(df) 
    # prediction_proba = load_clf.decision_function(df)

    # one_class_svm
    # prediction = load_clf.predict(df)

    #LOF
    # prediction = load_clf.predict(df)
    # if prediction[0] == -1:
        # prediction[0] = 1
    # else:
        # prediction[0] = -1

    # Set threshold
    # if prediction_proba[0][0] > 0.51:  #[0][0] = NG  #Random
    # if prediction[0] == -1: #IsolationForest
    # if prediction[0] == -1: #LOF
    # if prediction[0] == -1: #one_class_svm
    # if prediction[0] == 0: #k-mean
    if prediction[0] == 0 and prediction_proba[0][0] > 0.51:
        OKNG = 'NG'
        # OKNG = 'OK'
    else:
        OKNG = 'OK'
        # OKNG = 'NG'

    # return OKNG,time
    # return OKNG,time,prediction #Unsupervised
    # return OKNG,time,prediction_proba #Supervised
    return OKNG,time,prediction_proba,prediction #Supervised And IsolationForest

# ForTest_Prediction
# predict(TestX)
# print(predict(TestX))