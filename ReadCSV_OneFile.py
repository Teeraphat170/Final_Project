import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

# TestX = pd.read_csv(r'E:\Test\Sliding_Window\sliding_window_1.csv')
# TestX = pd.read_csv(r'C:\Users\Fourth\OneDrive - Srinakharinwirot University\เดสก์ท็อป\ฝึกงาน\Demo_machine_dataset\SWU_DATA_00000030.csv')
# filenameF = 'filename'


#Delete Unwants Row and Columns
def ReadCSV(df,filename):
    filename = filename
    TestX = df
    # ใช้ SWU_DATA_00000030.csv
    # TestX.reset_index(inplace=True)
    # TestX = TestX.drop(TestX.index[0])
    # TestX = TestX.drop(TestX.index[0])
    # TestX = TestX.drop(TestX.columns[0], axis=1)
    # TestX = TestX.drop(TestX.columns[0], axis=1)
    # for columns_name in TestX:
    #     TestX[columns_name] = TestX[columns_name].astype(float)

    # ใช้ sliding_window_1.csv
    TestX.reset_index(inplace=True)
    TestX = TestX.drop(TestX.columns[0], axis=1)
    TestX = TestX.drop(TestX.columns[0], axis=1)
    TestX = TestX.drop(TestX.columns[0], axis=1)
    TestX = TestX.drop(TestX.columns[0], axis=1)
    for columns_name in TestX:
        TestX[columns_name] = TestX[columns_name].astype(float)
    # print(TestX)

    # Scale Data
    scalar = MinMaxScaler()
    # scalar = StandardScaler()
    scaled_data = scalar.fit_transform(TestX)
    scaled_newdf = pd.DataFrame(data = scaled_data,columns=TestX.columns[:])
    
    # If not use scale data
    # scaled_newdf = TestX

    Mean = pd.DataFrame()
    empty = []
    m = 0
    for x in scaled_newdf:
    # for x in TestX:  
        total = scaled_newdf[x].mean()
        df2 = pd.DataFrame([total])
        Mean = pd.concat([Mean, df2], axis=1)
        m = m + 1
        empty.append(m)
    Mean.columns = (f'Mean{m}' for m in empty )
    # print(Mean)
    Median = pd.DataFrame()
    empty = []
    m = 0
    for x in scaled_newdf:
    # for x in TestX:
        total = scaled_newdf[x].median()
        df2 = pd.DataFrame([total])
        Median = pd.concat([Median, df2], axis=1)
        m = m + 1
        empty.append(m)
    Median.columns = (f'Median{m}' for m in empty )
    # print(Median)
    Std = pd.DataFrame()
    empty = []
    m = 0
    for x in scaled_newdf:
    # for x in TestX:
        total = scaled_newdf[x].std()
        df2 = pd.DataFrame([total])
        Std = pd.concat([Std, df2], axis=1)
        m = m + 1
        empty.append(m)
    Std.columns = (f'Std{m}' for m in empty )
    # print(Std)
    Mode = pd.DataFrame()
    empty = []
    m = 0
    for x in scaled_newdf:
    # for x in TestX:
        total = statistics.mode(scaled_newdf[x])
        df2 = pd.DataFrame([total])
        Mode = pd.concat([Mode, df2], axis=1)
        m = m + 1
        empty.append(m)
    Mode.columns = (f'Mode{m}' for m in empty )
    # print(Mode)
    Kurt = pd.DataFrame()
    empty = []
    m = 0
    for x in scaled_newdf:
    # for x in TestX:
        total = kurtosis(scaled_newdf[x],bias=False)
        df2 = pd.DataFrame([total])
        Kurt = pd.concat([Kurt, df2], axis=1)
        m = m + 1
        empty.append(m)
    Kurt.columns = (f'Kurtosis{m}' for m in empty )
    # print(Kurt)
    PtoP = pd.DataFrame()
    empty = []
    m = 0
    for x in scaled_newdf:
    # for x in TestX:
        total = scaled_newdf[x].max() + scaled_newdf[x].min()
        df2 = pd.DataFrame([total])
        PtoP = pd.concat([PtoP, df2], axis=1)
        m = m + 1
        empty.append(m)
    PtoP.columns = (f'PToP{m}' for m in empty )
    # print(PtoP)
    RMS = pd.DataFrame() 
    empty = []
    m = 0
    for x in scaled_newdf:
    # for x in TestX:
        c = TestX[x]
        da1 = c.iloc[[0]]
        total = np.sqrt((da1**2).sum() / len(scaled_newdf[x]))
        df2 = pd.DataFrame([total])
        RMS = pd.concat([RMS, df2], axis=1)
        m = m + 1
        empty.append(m)
    RMS.columns = (f'RMS{m}' for m in empty )
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
    fromanotherfile = pd.read_csv('CSV_XLSX/New_allfile_V2.csv')
    # onefile = list(scaled_df)
    onefile = list(result)
    manyfile = list(fromanotherfile)

    # Find Feature Importance
    newresult = pd.DataFrame()
    for i in range(len(manyfile)):
        for j in range(len(onefile)):
            if onefile[j] == manyfile[i]:
                newresult = pd.concat([newresult, result[onefile[j]]], axis=1)

    return filename,newresult
    # return filename

# For_Testing
# CSVReader = ReadCSV(TestX,filenameF)
# print(CSVReader)
# CSVReader.to_csv('CSV_XLSX/ReadCSV_OneFile.csv')
# print(CSVReader)
