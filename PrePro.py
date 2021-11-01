import nltk
import json
import pandas as pd
import jsonlines
from nltk.corpus.europarl_raw import english
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.probability import FreqDist
from collections import Counter
nltk.download()




#from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

data = []
for line in open("C:/Users/Gianluca/Desktop/Università/Master UniMi/IR/data.jsonl", "r"):
    data.append(json.loads(line))

#Build another list with only ID and tokenize text
#using dicts
stop_words = set(stopwords.words('english'))
table=str.maketrans('','',string.punctuation)

PreP_data=[]
print("Starting...")
for j in range(len(data)): # in range(len(a_list))
    case = {}
    case["ID"]=data[j]["id"]
    case['decision_date']=data[j]['decision_date']
    case["court"]=data[j]["court"]["name"]
    case["jurisdiction"]=data[j]["jurisdiction"]["name_long"]
    print(f"working on doc.id={j}")
    for a in data[j]["casebody"]["data"]["opinions"]:
        input_str=a["text"]
        input_str=input_str.lower()
        #input_str=input_str.translate(str.maketrans("",""))
        #input_str=input_str.translate(string.maketrans("",""), string.punctuation)
        input_str=input_str.strip()
        #input_str=[w.translate(table) for w in input_str]
        tokens = word_tokenize(input_str)
        case["Type_opinion"]=a["type"]
        case["result"] = [i for i in tokens if not i in stop_words]
        case["result"] = [i for i in case["result"] if not i in table]
        case["result"] = [i for i in case["result"] if i.isalpha()]
        PreP_data.append(case)
    print(f"doc.id={j} has been completed")
Process_data=pd.DataFrame(PreP_data)
print("Process Completed!")
Process_data.to_csv("data_process.csv", sep=';')
data_test=pd.read_csv("data_process.csv",sep=";")
textfile = open("Prep_data.txt", "w")
for element in PreP_data:
    textfile.write(element)
textfile.close()

import json
with open('data.json', 'w') as f:
    json.dump(PreP_data, f)

with open("data.json") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()

#fix a multiple rows with same ID
"C:/Users/Gianluca/Desktop/Università/Master UniMi/IR/data.jsonl"
Narcotics= ["cannabis", "cocaine", "methamphetamine", "smart drugs", "marijuana", "mdma", "lsd", "ketamina", "heroin", "fentanyl"]
Weapons=["gun", "knife", "weapon", "firearm", "rifle", "carabine", "shotgun", "assaults rifle", "sword", "blunt objects"]
Investigation=["male", "female", "man", "woman", "girl", "boy","gang", "mafia", "serial killer", "rape", "thefts", "recidivism", "arrest", "ethnicity", "gender", "robbery", "cybercrime"]
all_wrd=Narcotics + Weapons + Investigation
##dict for ethicities and gender
#convert to df jsonObject, clean from punctuation

#get from new df word count matrix, filter it >1 keep the ID-opinion
data_procs=pd.read_csv("data_process.csv",sep=";")

Count_words = Counter(data_procs.iloc[0]["result"].strip('][').replace("'","").split(', '))

CW_data=data_procs
CW_data = CW_data["result"].applymap(lambda x: FreqDist(x))

list_freq=[]
for i in range(len(data_procs)):#range(len(data_procs)):
    dict_freq=Counter(data_procs.iloc[i]["result"].strip('][').replace("'","").split(', '))
    dict_freq['Unnamed: 0'] = i
    list_freq.append(dict_freq)

import json
with open('freq_data.json', 'w') as f:
    json.dump(list_freq, f)

with open("freq_data.json") as jsonFile:
    Freq_json = json.load(jsonFile)
    jsonFile.close()

freq_data=pd.DataFrame(list_freq)
freq_data=pd.DataFrame.from_dict(Freq_json)
freq_data=pd.DataFrame.from_records(Freq_json)
Freq_df=pd.DataFrame()
#df_2add=pd.DataFrame(columns=['A','B','C'])

with open("freq_data.json") as jsonFile:
    Freq_json = json.load(jsonFile)
    jsonFile.close()
for j in range(len(Freq_json)):
    for k,v in list(Freq_json[j].items()):
        if k != "Unnamed: 0":
            if v <= 50:
                del Freq_json[j][k]

with open("freq_data.json") as jsonFile:
    Freq_json = json.load(jsonFile)
    jsonFile.close()
for j in range(len(Freq_json)):
    for k,v in list(Freq_json[j].items()):
        if k != "Unnamed: 0":
            if k not in all_wrd:
                del Freq_json[j][k]


filter = [i for i in Freq_json if not (len(i) <= 1)]
filter=pd.DataFrame(filter)
filter=filter.fillna(0)
#filter.to_csv("filtered_df.csv", sep=';')
data_procs=pd.read_csv("data_process.csv",sep=";")
result = pd.merge(filter, data_procs, how="left", on=["Unnamed: 0"])

#result.to_csv("join_df.csv", sep=';')
#result.to_csv("join_df_keyw.csv", sep=';')
#result=pd.read_csv("join_df_keyw.csv",sep=";", index_col=0)

#plotly TS using date
result["decision_date"]=pd.to_datetime(result["decision_date"], format='%Y-%m-%d')
not_presents=['smart drugs', 'LSD', 'carabine', 'cybercrime', 'serial killer', 'KETAMINA', 'MDMA', 'assaults rifle', 'blunt objects']
all_wrd_pre=  list(set(all_wrd) - set(not_presents))
import plotly.express as px
import psutil
lines = result.plot.line(x='decision_date', y=all_wrd_pre)

import matplotlib.pyplot as plt
plt.close("all")
plt.figure()
result.plot(x="decision_date", y=all_wrd_pre)
result.plot(x="decision_date", y="cocaine")
plt.figure()
result.plot(x="decision_date", y=["rifle", "cocaine","woman"], kind="bar")






data = []
for line in open("C:/Users/Gianluca/Desktop/Università/Master UniMi/IR/data.jsonl", "r"):
    data.append(json.loads(line))

#Build another list with only ID and tokenize text
#using dicts
stop_words = set(stopwords.words('english'))
table=str.maketrans('','',string.punctuation)

PreP_data=[]
print("Starting...")
for j in range(len(data)): # in range(len(a_list))
    case = {}
    for a in data[j]["casebody"]["data"]["opinions"]:
        input_str=a["text"]
        input_str=input_str.lower()
        #input_str=input_str.translate(str.maketrans("",""))
        #input_str=input_str.translate(string.maketrans("",""), string.punctuation)
        input_str=input_str.strip()
        #input_str=[w.translate(table) for w in input_str]
        #tokens = word_tokenize(input_str)
        case["Type_opinion"]=a["type"]
        case["result"] = [i for i in tokens if not i in stop_words]
        case["result"] = [i for i in case["result"] if not i in table]
        case["result"] = [i for i in case["result"] if i.isalpha()]
        PreP_data.append(case)
    print(f"doc.id={j} has been completed")
Process_data=pd.DataFrame(PreP_data)
print("Process Completed!")


##pure freq approach oversampling from recent documents---

    #{i:words.count(i["results"]) for i in set(i["result"])}

#Apply tf-idf for relevant term of the lists macro topics
##Develop Word2Vec Embedding to compare with simple frequency_based approach and get words similar to the one in the list (basic classification is possible here)
#TS analyst using data values from original df, group by opinions and other categorical variables presents in the original text (get trends)
#if presents also apply to the similar term not present in the list
#get similar from Word2Vec embedding or whatever
#Correlation matrix-->single term of the list, group term of lists--->move to enlarged list if possible.


