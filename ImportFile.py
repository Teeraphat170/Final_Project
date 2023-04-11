import glob
import os
import pandas as pd
from time import time,sleep
# import ReadCSV_OneFile
import MainFile
# import TestPredict
# import ConvertToCsv
# import SendEmail
# import SendLine
# import ToFireBase
# import ToGoogleSheet
from sqlalchemy import false
from datetime import datetime

# oldFileName = ''
# i = 0

def filepath():
    # folder_path = r"C:\Users\Fourth\OneDrive - Srinakharinwirot University\เดสก์ท็อป\ฝึกงาน\TestFolder" 
    folder_path = r"E:\Test\TestFolder\SWU_DATA"
    files_path = os.path.join(folder_path, '*')
    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
    return files,folder_path

# def filename():
#     files,folder_path = filepath()
#     if(len(os.listdir(folder_path)) == 0):
#       pass
#     else:
#       filename = os.path.basename(files[0]).split(".")[0]
#       return filename

def input():
    # global i
    # global oldFileName

    files,folder_path = filepath()
    
    # newFileName = files[0]
    if(len(os.listdir(folder_path)) == 0):
      pass
    elif(len(os.listdir(folder_path)) == 1):
      
      # ------------------------------------------
      try:
        df = pd.read_csv(files[0])
        return df
      except:
        pass
      # ------------------------------------------ 

  # ------------------------------------------ 
      # if i == 0:
      #   df = pd.read_csv(files[0])
        # i = i + 1
      #   return df
    # else:
    #   newFileName = files[0]
    #   if(oldFileName == newFileName):
    #     return false
    #   oldFileName = newFileName
    #   try:
    #     if files[0] != files[1]:
    #       df = pd.read_csv(files[0])
    #       return df
    #     else :
    #       print("Don't have rescent files")      
    #   except:
    #     pass

while True:
    sleep(1)
    df = input()
    # filenamex = filename()
    files,folder_path = filepath()
    try:

      # Feature Importance
      # namefile,newresult = NewRead_OneFile.ReadCSV(df,filenamex)
      namefile,newresult = MainFile.ReadCSV(df)
      # print(namefile,newresult)

      # Prediction
      # namefileX,okng,timeX,prediction_proba = TestPredict.predict(newresult,namefile)
      # print(namefile,timeX,okng + " " + " Probability(OK) : " + str(prediction_proba[0][1]) + " " +" Probability(NG) : " + str(prediction_proba[0][0]))

      ### Convert To CSV File
      # CSVReader = ConvertToCsv.tocsv(namefileX,okng,timeX,prediction_proba)
      ### print(CSVReader)

      ### Send Email 
      # email_sender = SendEmail.send_email(namefileX,okng,timeX,prediction_proba)
      ### print(email_sender)

      ### Send To Line
      # line_sender,line_sender1 = SendLine.send_line(namefileX,okng,timeX,prediction_proba)
      ### print(line_sender,line_sender1)

      ### To Firebase
      # To_Firebase = ToFireBase.ToFirebase(namefileX,okng,timeX,prediction_proba)
      ### print(To_Firebase)

      ### To Google Sheets
      # To_googlesheets = ToGoogleSheet.googlesheets(namefileX,okng,timeX,prediction_proba)

      ### Delete File
      os.remove(files[0])
      
    except:
      print("No File in Directory or Erroe Something")
      continue

