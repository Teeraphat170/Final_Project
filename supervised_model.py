import pandas as pd
# import ReadCSV_AllFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# model = pd.read_csv('CSV_XLSX/New_allfile_NoScale.csv')
# model = pd.read_csv('CSV/XLSX/New_allfile_V3.csv')
# model = pd.read_excel('CSV_XLSX/ReadCSV_NewOk.xlsx')


# model = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale_fs04.csv')
# model = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale_fs05.csv')
# model = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale.csv')
# model = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale_fs06.csv')
# model = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale_SameFeatureScale.csv')
model = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale_Sliding_Windows.csv')
# print(model)
model = model.iloc[: , 1:] #เอา column ที่เกินออก
# print(model)
# model = ReadCSV_AllFile.xx
# fromanotherfile = pd.read_csv('CSV_XLSX2/Ready_For_Model_Scale.csv')
# NewOk
# data = {#เอาแค่ column name and Scale
#                     "Mean2": [420, 380, 390],
#                     "Std1": [50, 40, 45],
#                     "Mean1": [50, 40, 45],
#                     "Std2": [50, 40, 45],
#                     "Median1": [50, 40, 45],
#                     "Std3": [50, 40, 45],
#                     "Kurtosis1": [50, 40, 45],
#                     "Kurtosis4": [50, 40, 45]
#                 } 
# data = {###เอาแค่ column name and No Scale
#                   "Std3": [420, 380, 390],
#                   "Std2": [420, 380, 390],
#                   "Mean2": [420, 380, 390],
#                   "Std1": [420, 380, 390],
#                   "PToP1": [420, 380, 390],
#                   "PToP4": [420, 380, 390],
#                   "PToP2": [420, 380, 390],
#                   "Std4": [420, 380, 390],
#                   "Kurtosis1": [420, 380, 390],
#                   "Kurtosis4": [420, 380, 390]
#                 } 
# # fromanotherfile = pd.DataFrame(data)
# # onefile = list(scaled_df)
# onefile = list(model)
# manyfile = list(fromanotherfile)      
# # Find Feature Importance
# newresult = pd.DataFrame()
# for i in range(len(manyfile)):
#     for j in range(len(onefile)):
#         if onefile[j] == manyfile[i]:
#             newresult = pd.concat([newresult, model[onefile[j]]], axis=1)

# model = newresult
# print(model)
# xs = [1] * len(model)
# # count = xs.count(1)
# # print(count)
# jud = pd.DataFrame(xs, columns=['Result'])

# Old Data
# jud = pd.read_excel(r"E:\Test\Result_OK_NG.xlsx")
# # jud = pd.read_excel(r"C:\Users\Fourth\Demo_machine_dataset\Result.xlsx")
# jud = pd.DataFrame(jud['Result'])
# print(jud)
# # jud = jud.dropna()
# # print(jud)
# # jud.to_excel("CSV_XLSX/Result_OK_NG.xlsx")
# df = model.copy()

# # # Separating X and y
# X = df
# Y = jud

# # Build random forest model
# x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 7)
# rf = RandomForestClassifier(random_state = 7,n_estimators=250)
# rf.fit(x_train,y_train.values.ravel())

# # confusion_matrix & classification_report
# y_pred = rf.predict(x_test)
# cm = confusion_matrix(y_test,y_pred)
# print('Confusion matrix: \n',cm)
# print('Classification report: \n',classification_report(y_test,y_pred))

# Saving the model
import pickle
# pickle.dump(rf, open('Model2/Random_Scale_V2.pkl', 'wb'))
# pickle.dump(rf, open('Model2/New_Model_V3.pkl', 'wb'))
# pickle.dump(rf, open('Model2/Random_Scale.pkl', 'wb'))
# pickle.dump(rf, open('Model2/Random_noScale.pkl', 'wb'))
# pickle.dump(rf, open('Model2/Random_NewOk_Scale.pkl', 'wb'))
# pickle.dump(rf, open('Model2/Random_NewOk_NoScale.pkl', 'wb'))
# pickle.dump(rf, open('Model3/Random_Scale.pkl', 'wb'))
# pickle.dump(rf, open('Model3/Random_NoScale.pkl', 'wb'))
# pickle.dump(rf, open('Model3/Random_NoScale_2.pkl', 'wb'))
# pickle.dump(rf, open('Model3/Random_NoScale_fs06.pkl', 'wb'))
# pickle.dump(rf, open('Model3/Random_NoScale_fs04.pkl', 'wb'))
# pickle.dump(rf, open('Model3/Random_NoScale_fs05.pkl', 'wb'))

load_clf = pickle.load(open('Model3/Random_NoScale.pkl', 'rb'))

prediction = load_clf.predict(model) 
# prediction_proba = load_clf.predict_proba(model) 
default_prediction = [1 if i==1 else 0 for i in prediction]
print(default_prediction.count(1),default_prediction.count(0))

# print(prediction_proba)