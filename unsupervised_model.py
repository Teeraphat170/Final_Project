import pandas as pd
import pickle
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_classification
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

start = time.time()
# model = pd.read_csv('allfile.csv')
# model = pd.read_csv('CSV_XLSX/TotalFile.csv')
# model = pd.read_csv('CSV_XLSX/ReadCSV_AllFile_SlidingWindow.csv')
# model = pd.read_csv('CSV_XLSX/ReadCSV_AllFile_SlidingWindow_NewOk.csv')
# model = pd.read_excel('CSV_XLSX/ReadCSV_AllFile_V3.xlsx')
# model = pd.read_excel('CSV_XLSX/ReadCSV_AllFile_NoScale.xlsx')
# model = pd.read_excel('CSV_XLSX/ReadCSV_NewOk.xlsx')
# model = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale.csv')
model = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale_Sliding_Windows.csv')
model = model.iloc[: , 1:]
# test = model

#IsolationForest
# # iso = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, 
#                       bootstrap=False, n_jobs=None, random_state=None, verbose=0, warm_start=False).fit(model)
print("IsolationForest")
default_iso = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, 
                           bootstrap=False, n_jobs=None, random_state=None, verbose=0, warm_start=False).fit(model)

default_prediction = default_iso.predict(model)
default_prediction = [1 if i==-1 else 0 for i in default_prediction]
print("IsolationForest_default")
# print(default_prediction)

iso = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.17),max_features=1.0, 
                      bootstrap=False, n_jobs=-1, random_state=42, verbose=0).fit(model)
# pickle.dump(iso, open("Model2/IsolationForest_sliding.pkl", "wb"))
prediction = iso.predict(model)
prediction = [1 if i==-1 else 0 for i in prediction]
print("IsolationForest_tuning")
# print(prediction)

print("Anomaly(default):",default_prediction.count(1),"::","Anomaly(tuning):",prediction.count(1)) #Anomaly
print("Normal(default):",default_prediction.count(0),"::","Normal(tuning):",prediction.count(0)) 

#Kmean
print("")
print("k-mean")
default_kmean =KMeans(n_clusters=2, max_iter=300, tol=0.0001, verbose=0, 
                      random_state=None, copy_x=True, algorithm='lloyd').fit(model)
default_kmean = default_kmean.predict(model)
default_kmean = [0 if i==1 else 1 for i in default_kmean]
print("k-mean_default")
# print(default_kmean)

kmean = KMeans(n_clusters=2).fit(model)
# pickle.dump(kmean, open("Model2/kmean_sliding.pkl", "wb"))
prediction = kmean.predict(model)
prediction = [0 if i==1 else 1 for i in prediction]
print("k-mean_tuning")
# print(prediction)

print("Anomaly(default):",default_kmean.count(0),"::","Anomaly(tuning):",prediction.count(1)) #Anomaly
print("Normal(default):",default_kmean.count(1),"::","Normal(tuning):",prediction.count(0)) 

#One-Class SVM
print("")
print("one_class_svm")
default_one_class_svm = OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, 
                            cache_size=200, verbose=False, max_iter=-1).fit(model)
default_one_class_svm = default_one_class_svm.predict(model)
default_one_class_svm = [1 if i==-1 else 0 for i in default_one_class_svm]
print("one_class_svm_default")
# print(default_one_class_svm)

one_class_svm = OneClassSVM(kernel='rbf', degree=5, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=False, 
                                cache_size=200, verbose=False, max_iter=0).fit(model)
# pickle.dump(one_class_svm, open("Model2/one_class_svm_sliding.pkl", "wb"))
prediction = one_class_svm.predict(model)
prediction = [1 if i==-1 else 0 for i in prediction]
print("one_class_svm_tuning")
# print(prediction)

print("Anomaly(default):",default_one_class_svm.count(1),"::","Anomaly(tuning):",prediction.count(1)) #Anomaly
print("Normal(default):",default_one_class_svm.count(0),"::","Normal(tuning):",prediction.count(0))  

#LOF
print("")
print("LocalOutlierFactor")
clf = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, 
                         metric_params=None, contamination='auto', novelty=True, n_jobs=None).fit(model)
default_lof = clf.predict(model)
default_lof = [1 if i==-1 else 0 for i in default_lof]
# print("default_lof")
# print(default_lof)

clf = LocalOutlierFactor(n_neighbors=15, algorithm='auto', leaf_size=30, metric='minkowski', p=2, 
                         metric_params=None, contamination=0.19, novelty=True, n_jobs=None).fit(model)
# pickle.dump(clf, open("Model2/LOF_sliding.pkl", "wb"))
prediction = clf.predict(model)
prediction = [1 if i==-1 else 0 for i in prediction]
# print("tuning_lof")
# print(prediction)

print("Anomaly(default):",default_lof.count(1),"::","Anomaly(tuning):",prediction.count(1)) #Anomaly
print("Normal(default):",default_lof.count(0),"::","Normal(tuning):",prediction.count(0)) 

# # X_scores = clf.negative_outlier_factor_
# print(X_scores)


# #save model
# # pickle.dump(kmean, open("Model/k-mean_SlidingWindow_NewOk.pkl", "wb"))
# # pickle.dump(kmean, open("Model2/k-mean_Scale.pkl", "wb"))
# # pickle.dump(kmean, open("Model2/k-mean_NewOk_NoScale.pkl", "wb"))
# pickle.dump(kmean, open("Model2/k-mean_NewOk_Scale.pkl", "wb"))
# pickle.dump(iso, open("Model2/IsolationForest_1.pkl", "wb"))
# pickle.dump(isolation_model, open("Model2/IsolationForest_2.pkl", "wb"))
# pickle.dump(iso, open("Model2/IsolationForest_3.pkl", "wb"))
# pickle.dump(iso, open("Model2/IsolationForest_5.pkl", "wb"))
# pickle.dump(iso, open("Model2/IsolationForest_6.pkl", "wb"))
# pickle.dump(iso, open("Model2/IsolationForest_7.pkl", "wb"))
# pickle.dump(iso, open("Model2/IsolationForest_8.pkl", "wb"))
# pickle.dump(iso, open("Model2/IsolationForest_scale.pkl", "wb"))
# pickle.dump(one_class_svm, open("Model2/one_class_svm1.pkl", "wb"))
# pickle.dump(one_class_svm, open("Model2/one_class_svm2.pkl", "wb"))
# pickle.dump(one_class_svm, open("Model2/one_class_svm7.pkl", "wb"))
# pickle.dump(one_class_svm, open("Model2/one_class_svm8.pkl", "wb"))
# pickle.dump(LOF, open("Model2/LOF4.pkl", "wb"))
# pickle.dump(clf, open("Model2/LOF5.pkl", "wb"))
# pickle.dump(clf, open("Model2/LOF5.pkl", "wb"))

end = time.time()
print(end - start)



import pickle
pickle.dump(one_class_svm, open("one_class_svm1.pkl", "wb"))