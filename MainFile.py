
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statistics
import TestPredict
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis
import time
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

start = time.time()
# TestX = pd.read_csv(r'E:\Test\Sliding_Window\sliding_window_1.csv')
# TestX = pd.read_csv(r'C:\Users\Fourth\OneDrive - Srinakharinwirot University\เดสก์ท็อป\ฝึกงาน\Demo_machine_dataset\SWU_DATA_00000030.csv')TotalFile98_9C
# TestX = pd.read_csv(r'C:\Users\Fourth\OneDrive - Srinakharinwirot University\เดสก์ท็อป\ฝึกงาน\MTClassification\CSV_XLSX\TotalFile.csv')
# TestX = pd.read_csv(r'C:\Users\Fourth\OneDrive - Srinakharinwirot University\เดสก์ท็อป\Work\ปี 4\Final Project\MTClassification\CSV_XLSX\TotalFile46_47.csv')
# TestX = pd.read_csv(r'C:\Users\Fourth\OneDrive - Srinakharinwirot University\เดสก์ท็อป\Work\ปี 4\Final Project\MTClassification\CSV_XLSX\TotalFile35_36.csv')
# TestX = pd.read_csv(r'C:\Users\Fourth\OneDrive - Srinakharinwirot University\เดสก์ท็อป\Work\ปี 4\Final Project\MTClassification\CSV_XLSX\TotalFile36_37.csv')
# TestX = pd.read_csv(r'C:\Users\Fourth\OneDrive - Srinakharinwirot University\เดสก์ท็อป\Work\ปี 4\Final Project\MTClassification\CSV_XLSX\TotalFile6D_73.csv')
# TestX = pd.read_csv(r'C:\Users\Fourth\OneDrive - Srinakharinwirot University\เดสก์ท็อป\Work\ปี 4\Final Project\MTClassification\CSV_XLSX\TotalFile98_9C.csv')
# TestX = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale_Sliding_Windows.csv')
TestX = pd.read_csv('CSV_XLSX2/All134.csv')


def ReadCSV(df):
    # filename = filename
    TestX = df
    TestX = TestX.iloc[: , 1:] 
    # print(TestX)

    ###### Delete Unwants Row and Columns
    ###### ใช้ SWU_DATA_00000030.csv
    # TestX.reset_index(inplace=True)
    # TestX = TestX.drop(TestX.index[0])
    # TestX = TestX.drop(TestX.index[0])
    # TestX = TestX.drop(TestX.columns[0], axis=1)
    # TestX = TestX.drop(TestX.columns[0], axis=1)
    # for columns_name in TestX:
    #     TestX[columns_name] = TestX[columns_name].astype(float)

    ##### ใช้ sliding_window_1.csv
    # TestX.reset_index(inplace=True)
    # TestX = TestX.drop(TestX.columns[0], axis=1)
    # TestX = TestX.drop(TestX.columns[0], axis=1)
    # TestX = TestX.drop(TestX.columns[0], axis=1)
    # TestX = TestX.drop(TestX.columns[0], axis=1)
    # for columns_name in TestX:
    #     TestX[columns_name] = TestX[columns_name].astype(float)
    # print(TestX)

    ###### Sliding Windows
    First,Last = 0,390
    # i = 1
    # for x in range(len(TestX)): Not Work
    #     # print(x)
    #     if x == len(TestX) : break
    # while i <= 10:
    TotalX = pd.DataFrame()
    while Last <= len(TestX): 
    # while Last <= 600:     
        Position = TestX.iloc[First:Last]
        # First = First + 2
        # Last = Last + 2
        # print(Position)
        #For Test
        # i = i + 1

        # Scale Data
        # scalar = MinMaxScaler()
        # # scalar = StandardScaler()
        # scaled_data = scalar.fit_transform(Position)
        # scaled_newdf = pd.DataFrame(data = scaled_data,columns=TestX.columns[:])

        # If not use scale data
        scaled_newdf = Position

        Mean = pd.DataFrame()
        # empty = []
        # m = 0
        for x in scaled_newdf:
        # for x in TestX:  
            total = scaled_newdf[x].mean()
            df2 = pd.DataFrame([total])
            Mean = pd.concat([Mean, df2], axis=1)
            # m = m + 1
            # empty.append(m)
        # Mean.columns = (f'Mean{m}' for m in empty )
        Mean.columns = ['Mean1','Mean2','Mean3','Mean4','Mean5',
                        'Mean6','Mean7','Mean8','Mean9']
        # print(Mean)
        Median = pd.DataFrame()
        # empty = []
        # m = 0
        for x in scaled_newdf:
        # for x in TestX:
            total = scaled_newdf[x].median()
            df2 = pd.DataFrame([total])
            Median = pd.concat([Median, df2], axis=1)
            # m = m + 1
            # empty.append(m)
        # Median.columns = (f'Median{m}' for m in empty )
        Median.columns = ['Median1','Median2','Median3','Median4','Median5',
                          'Median6','Median7','Median8','Median9']
        # print(Median)
        Std = pd.DataFrame()
        # empty = []
        # m = 0
        for x in scaled_newdf:
        # for x in TestX:
            total = scaled_newdf[x].std()
            df2 = pd.DataFrame([total])
            Std = pd.concat([Std, df2], axis=1)
            # m = m + 1
            # empty.append(m)
        # Std.columns = (f'Std{m}' for m in empty )
        Std.columns = ['Std1','Std2','Std3','Std4',
                       'Std5','Std6','Std7','Std8','Std9']
        # print(Std)
        Mode = pd.DataFrame()
        # empty = []
        # m = 0
        for x in scaled_newdf:
        # for x in TestX:
            total = statistics.mode(scaled_newdf[x])
            df2 = pd.DataFrame([total])
            Mode = pd.concat([Mode, df2], axis=1)
            # m = m + 1
            # empty.append(m)
        # Mode.columns = (f'Mode{m}' for m in empty )
        Mode.columns = ['Mode1','Mode2','Mode3','Mode4','Mode5',
                        'Mode6','Mode7','Mode8','Mode9']
        # print(Mode)
        Kurt = pd.DataFrame()
        # empty = []
        # m = 0
        for x in scaled_newdf:
        # for x in TestX:
            total = kurtosis(scaled_newdf[x],bias=False)
            df2 = pd.DataFrame([total])
            Kurt = pd.concat([Kurt, df2], axis=1)
            # m = m + 1
            # empty.append(m)
        # Kurt.columns = (f'Kurtosis{m}' for m in empty )
        Kurt.columns = ['Kurtosis1','Kurtosis2','Kurtosis3',
                        'Kurtosis4','Kurtosis5','Kurtosis6',
                        'Kurtosis7','Kurtosis8','Kurtosis9'] # New way
        # print(Kurt)
        PtoP = pd.DataFrame()
        # empty = []
        # m = 0
        for x in scaled_newdf:
        # for x in TestX:
            total = scaled_newdf[x].max() + scaled_newdf[x].min()
            df2 = pd.DataFrame([total])
            PtoP = pd.concat([PtoP, df2], axis=1)
            # m = m + 1
            # empty.append(m)
        # PtoP.columns = (f'PToP{m}' for m in empty )
        PtoP.columns = ['PToP1','PToP2','PToP3','PToP4',
                        'PToP5','PToP6','PToP7','PToP8','PToP9']
        # print(PtoP)
        RMS = pd.DataFrame() 
        # empty = []
        # m = 0
        for x in scaled_newdf:
        # for x in TestX:
            c = TestX[x]
            da1 = c.iloc[[0]]
            total = np.sqrt((da1**2).sum() / len(scaled_newdf[x]))
            df2 = pd.DataFrame([total])
            RMS = pd.concat([RMS, df2], axis=1)
            # m = m + 1
            # empty.append(m)
        # RMS.columns = (f'RMS{m}' for m in empty )
        RMS.columns = ['RMS1','RMS2','RMS3','RMS4',
                       'RMS5','RMS6','RMS7','RMS8','RMS9']
        # print(RMS)
        result = pd.concat([Mean, Median], axis=1)
        result = pd.concat([result, Std], axis=1)
        result = pd.concat([result, Mode], axis=1)
        result = pd.concat([result, Kurt], axis=1)
        result = pd.concat([result, PtoP], axis=1)
        result = pd.concat([result, RMS], axis=1)

        # scalar = MinMaxScaler()
        # # scalar = StandardScaler()
        # scaled_data = scalar.fit_transform(result)
        # scaled_newdf = pd.DataFrame(data = scaled_data,columns=result.columns[:])
        # scaled_newdf.to_csv('ccccccccccccccccccccccccccccc.csv')

        # print(result)
        # print(scaled_newdf)

        # File that has the same feature
        # fromanotherfile = pd.read_csv('CSV_XLSX/allfile.csv')
        # fromanotherfile = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale_fs04.csv')
        # fromanotherfile = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale_fs05.csv')
        # fromanotherfile = pd.read_csv('CSV_XLSX2/Ready_For_Model_NoScale.csv')
        # fromanotherfile = pd.read_csv('CSV_XLSX2/Ready_For_Model_Scale.csv')

        # data = {###เอาแค่ column name and No Scale fs06
        #           "PtoP1": [420, 380, 390],
        #           "Kurtosis1": [420, 380, 390],
        #           "PtoP4": [420, 380, 390],
        #           "Std4": [420, 380, 390],
        #           "Kurtosis4": [420, 380, 390],
        #           "Std1": [420, 380, 390]
        #         } 

        # fromanotherfile = pd.DataFrame(data)
        # onefile = list(scaled_df)
        # onefile = list(result)
        # # print(onefile)
        # manyfile = list(fromanotherfile)
        # # print(manyfile)

        # # Find Feature Importance
        # newresult = pd.DataFrame()
        # for i in range(len(manyfile)):
        #     for j in range(len(onefile)):
        #         if onefile[j] == manyfile[i]:
        #             newresult = pd.concat([newresult, result[onefile[j]]], axis=1)
        
        # result = scaled_newdf

        # print(newresult)
        newresult = result[['Std3','Std2','Mean2','Std1','PToP1','PToP4','PToP2','Std4','Kurtosis1','Kurtosis4']]
        # okng,timeX,prediction = TestPredict.predict(newresult) #Unsupervised
        # okng,timeX,prediction_proba = TestPredict.predict(newresult) #Supervised
        okng,timeX,prediction_proba,prediction = TestPredict.predict(newresult) #Supervised And IsolationForest

        # print(newresult)
        data = {"Ok_NG":okng,
                "Time":timeX,
                "prediction_proba_0":prediction_proba[0][0],
                "prediction_proba_1":prediction_proba[0][1],
                "prediction":prediction
                }
        df = pd.DataFrame(data)
        # TotalX = pd.concat([TotalX,df], axis=0)
        # print(First,Last,okng,timeX,prediction) #Unsupervised
        # print(First,Last,okng,timeX,prediction_proba) #Supervised
        print(First,Last,okng,timeX,prediction_proba,prediction) #Supervised And IsolationForest

        First = First + 5
        Last = Last + 5

    return df
    # return filename

##### For_Testing
CSVReader = ReadCSV(TestX)
# print(CSVReader)

end = time.time()
print(end - start)

# CSVReader.to_csv('CSV_XLSX2/Result_0_1_1.csv')
# print(CSVReader)
