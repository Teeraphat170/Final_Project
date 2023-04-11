import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os
import glob
from scipy.stats import kurtosis
from scipy.signal import find_peaks
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

path = r"C:\Users\Fourth\OneDrive - Srinakharinwirot University\เดสก์ท็อป\New folder"  
# path = r"C:\Users\Fourth\OneDrive - Srinakharinwirot University\เดสก์ท็อป\ฝึกงาน\NewOK"
# path = r"E:\Test\New Dataset"
csv_files = glob.glob(path + "/*.csv")
print(csv_files)

def readfile():
    TestX = pd.read_csv(f)
    # TestX=TestX.drop(TestX.index[0])
    # TestX.reset_index(inplace=True)
    # del TestX['level_0']
    # del TestX['level_1']
    # TestX=TestX.drop(TestX.index[0])
    # TestX['level_2'] = TestX['level_2'].astype(float)
    # TestX['level_3'] = TestX['level_3'].astype(float)
    # TestX['level_4'] = TestX['level_4'].astype(float)
    # TestX['level_5'] = TestX['level_5'].astype(float)
    # TestX['[LOGGING]'] = TestX['[LOGGING]'].astype(float)
    # TestX['RD81DL96_1'] = TestX['RD81DL96_1'].astype(float)
    # TestX['2'] = TestX['2'].astype(float)
    # TestX['3'] = TestX['3'].astype(float)
    # TestX['4'] = TestX['4'].astype(float)

    TestX = TestX.drop(columns=['[LOGGING]', 'RD81DL96_1'])
    TestX = TestX.drop(TestX.index[0])
    TestX = TestX.drop(TestX.index[0])

    TestX['2'] = TestX['2'].astype(float)
    TestX['3'] = TestX['3'].astype(float)
    TestX['4'] = TestX['4'].astype(float)
    TestX['Unnamed: 5'] = TestX['Unnamed: 5'].astype(float)
    TestX['Unnamed: 6'] = TestX['Unnamed: 6'].astype(float)
    TestX['Unnamed: 7'] = TestX['Unnamed: 7'].astype(float)
    TestX['Unnamed: 8'] = TestX['Unnamed: 8'].astype(float)
    TestX['Unnamed: 9'] = TestX['Unnamed: 9'].astype(float)
    TestX['Unnamed: 10'] = TestX['Unnamed: 10'].astype(float)
    TestX['Unnamed: 11'] = TestX['Unnamed: 11'].astype(float)
    # TestX = TestX.set_index(0)
    return TestX

TotalFile = pd.DataFrame()
for f in csv_files:

    TestX = readfile()
    # print(TestX)

    TotalFile = pd.concat([TotalFile, TestX], axis=0)

# print(TotalFile)
TotalFile.to_csv("CSV_XLSX2/All134_with_label.csv")
print(TotalFile)
# print()
# print(TotalFile.iloc[1:4 , -1:]) #result 0,1
# print(TotalFile.iloc[1:4 , :-1]) #ยกเว้น 0,1