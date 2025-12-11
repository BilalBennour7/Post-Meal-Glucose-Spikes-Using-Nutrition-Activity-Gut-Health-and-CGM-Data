import pandas as pd 
import glob
import os 

macro_files= glob.glob("CGMacros-*.csv")

macro_list=[]

for file in macro_files:
    df=pd.read_csv(file)
    subject_id= int(file.split("-")[1].split(".")[0])
    df["subject"]=subject_id
    macro_list.append(df)

all_macro=pd.concat(macro_list, ignore_index=True)

bio= pd.read_csv("bio.csv")
gut_health=pd.read_csv("gut_health_test.csv")


final=(all_macro
       .merge(bio, on="subject", how="left")
       .merge(gut_health, on="subject", how="left"))

final.to_csv("final_dataset.csv", index=False)


