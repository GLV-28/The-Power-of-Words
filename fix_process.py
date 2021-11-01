import nltk
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import jsonlines
from nltk.corpus.europarl_raw import english
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.probability import FreqDist
from collections import Counter
import ast
from  tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense,Flatten,Embedding
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy import stats

Narcotics= ["cannabis", "cocaine", "methamphetamine", "smart drugs", "marijuana", "mdma", "lsd", "ketamina", "heroin", "fentanyl"]
Weapons=["gun", "knife", "weapon", "firearm", "rifle", "carabine", "shotgun", "assaults rifle", "sword", "blunt objects"]
Investigation=["male", "female", "man", "woman", "girl", "boy","gang", "mafia", "serial killer", "rape", "thefts", "recidivism", "arrest", "ethnicity", "gender", "robbery", "cybercrime"]

all_wrd=Narcotics + Weapons + Investigation
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
result["decision_year"]=result["decision_date"].dt.year
not_presents=['smart drugs', 'LSD', 'carabine', 'cybercrime', 'serial killer', 'ketamina', 'MDMA', 'assaults rifle', 'blunt objects']
all_wrd_pre=  list(set(all_wrd) - set(not_presents))

all_wrd_pre_narcos=  list(set(Narcotics) - set(not_presents))
all_wrd_pre_wep=  list(set(Weapons) - set(not_presents))
all_wrd_pre_process=  list(set(Investigation) - set(not_presents))

result["Narcos"] = result[all_wrd_pre_narcos].sum(axis=1)
result["Weapons"] = result[all_wrd_pre_wep].sum(axis=1)
result["Investigation"] = result[all_wrd_pre_process].sum(axis=1)

all_cats= ["Narcos", "Weapons", "Investigation"]

all_comp = all_wrd_pre + all_cats

plt.figure()
result.groupby('decision_year')[all_cats].nunique().plot(kind='line')
#all trend
plt.figure()
result[(result['decision_year'] >= 1990) & (result['decision_year'] <= 2020)].groupby('decision_year')[all_cats].nunique().plot(kind='line')
#specific year-range
result.groupby('decision_year')[all_cats].nunique().plot(kind='line')


#filtered
result[ all_wrd_pre ] = result[all_wrd_pre].div(result[ all_wrd_pre ].sum(axis=1), axis=0)
result.groupby('decision_year')[all_wrd_pre].nunique().plot(kind='line')
result.groupby('decision_date')[all_wrd_pre].nunique().plot(kind='line')
result.groupby('decision_date')["weapon", "firearm","gun", "knife", "rifle", "shotgun", "sword"].nunique().plot(kind='line')

plt.figure()
result["Type_opinion"].plot.bar(stacked=True)

result[["Freq",'decision_year']].groupby('decision_year').sum("Freq").plot(kind='bar')
plt.figure()
result["Freq"]=1
plt.close("all")
plt.figure()
result.plot(x="decision_date", y=all_wrd_pre, kind="line")
result.plot(x="decision_year", y=all_wrd_pre, kind="line")


plt.figure()
result.groupby('decision_year')[all_wrd_pre_narcos].nunique().plot(kind='line')
plt.figure()
result.groupby('decision_year')[all_wrd_pre_wep].nunique().plot(kind='line')
plt.figure()
result.groupby('decision_year')[all_wrd_pre_process].nunique().plot(kind='line')


plt.figure()
result[(result['decision_year'] >= 1990) & (result['decision_year'] <= 2020)].groupby('decision_year')[all_wrd_pre_narcos].nunique().plot(kind='line')


plt.figure()
result[(result['decision_year'] >= 1990) & (result['decision_year'] <= 2020)].groupby('decision_year')[all_wrd_pre_wep].nunique().plot(kind='line')

plt.figure()
result[(result['decision_year'] >= 1990) & (result['decision_year'] <= 2020)].groupby('decision_year')[all_wrd_pre_process].nunique().plot(kind='line')


plt.figure()
result.groupby('decision_year')[all_wrd_pre_narcos].nunique().plot(kind='line')



result.groupby('Type_opinion').nunique()
##5 possible values
plt.figure()
result.plot(x="decision_date", y=["cocaine", "cannabis", "lsd"], kind="line")


plt.figure()
result.plot(x="decision_year", y=["cocaine", "cannabis", "lsd"], kind="line")


plt.figure()
result.plot(x="Type_opinion",kind='bar')



plt.figure()
result.groupby('decision_year')['mafia'].nunique().plot(kind='line')

plt.figure()
result.groupby('decision_year').nunique().plot(kind='line')

plt.figure()
result.groupby('Type_opinion')['mafia', 'cocaine', 'cannabis', 'rifle'].nunique().plot(kind='bar')
plt.show()
##Oversampling of majority, kinda expected
plt.figure()
result.groupby(['Type_opinion']).size().to_frame().plot(kind='bar',stacked=True,legend=False)

plt.figure()
result.groupby('Type_opinion')['lsd','heorin','cannabis'].nunique().plot(kind='bar')


word_corr=result[all_wrd_pre].corr()

plt.figure()
sn.heatmap(word_corr, annot=True)

word_corr_narcos=result[all_wrd_pre_narcos].corr()
word_corr_process=result[all_wrd_pre_process].corr()
word_corr_wep=result[all_wrd_pre_wep].corr()

word_corr_cat=result[["Narcos", "Weapons", "Investigation"]].corr()
plt.figure()
sn.heatmap(word_corr_cat, annot=True)

result_c=pd.get_dummies(result.court, prefix='Court')
result_t=pd.get_dummies(result.Type_opinion, prefix='Type_opinion')
result_ct=pd.concat([result_c,result_t], axis=1)
word_corr_CT=result_ct.corr()
plt.figure()
sn.heatmap(word_corr_CT, annot=True)



plt.figure()
sn.heatmap(word_corr_narcos, annot=True)

plt.figure()
sn.heatmap(word_corr_wep, annot=True)

plt.figure()
sn.heatmap(word_corr_process, annot=True)

word_corr_narcos_grp=result.groupby('Type_opinion')[all_wrd_pre_narcos].corr()
plt.figure()
sn.heatmap(word_corr_narcos_grp, annot=True)
#confirmed syno cannabis-marijuana

word_corr_wep_grp=result.groupby('Type_opinion')[all_wrd_pre_wep].corr()
plt.figure()
sn.heatmap(word_corr_wep_grp, annot=True)
#stronger with rehearing or concurrence rifle,gun, shotgun, weapon
#più concurrence quando si usano termini più generici a quanto pare
#gruppando per dissent--->female positivo corr con male-->riferimenti ai generi nelle opinioni dissenso
#reharing male positivo, theft invece con arrest
#in altri reati ciò non succede.
word_corr_inv_grp=result.groupby('Type_opinion')[all_wrd_pre_process].corr()

word_corr_inv_grp=result.groupby('decision_year')[all_wrd_pre_process].corr()

plt.figure()
sn.heatmap(word_corr_inv_grp, annot=True)
#rehearing theft, male and arrest, dissent male-female and dissent-theft female

word_corr_cats=result.groupby('Type_opinion')[all_cats].corr()
plt.figure()
sn.heatmap(word_corr_cats, annot=True)
#majority investigation weapons--->all'interno della majority terms inv e wea sono correlazione.

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
#costruire corpus usando dataset, estrarre e fare sia count matrix che idf matrix dei termini in lista.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform()


idf_matrix=TfidfTransformer().fit_transform(result[all_comp])
idf_df=pd.DataFrame.sparse.from_spmatrix(idf_matrix)
idf_df.columns=result[all_comp].columns
idf_df["decision_year"]=result["decision_year"]
idf_df["Type_opinion"]=result["Type_opinion"]

plt.figure()
idf_df.groupby('decision_year')['mafia'].nunique().plot(kind='line')


plt.figure()
idf_df.groupby('decision_year').nunique().plot(kind='line')


plt.figure()
idf_df.groupby('decision_year')[all_wrd_pre_narcos].nunique().plot(kind='line')

plt.figure()
idf_df.groupby('decision_year')[all_wrd_pre_wep].nunique().plot(kind='line')

plt.figure()
idf_df.groupby('decision_year')[all_wrd_pre_process].nunique().plot(kind='line')

plt.figure()
idf_df.groupby('decision_year')[all_cats].nunique().plot(kind='line')


plt.figure()
idf_df[(idf_df['Type_opinion'] == "majority")].groupby('decision_year')[all_wrd_pre_process].nunique().plot(kind='line')

plt.figure()
idf_df[(idf_df['Type_opinion'] == "dissent")].groupby('decision_year')[all_wrd_pre_process].nunique().plot(kind='line')

plt.figure()
idf_df[(idf_df['Type_opinion'] == "dissent")].groupby('decision_year')["mafia"].nunique().plot(kind='line')

plt.figure()
idf_df[(idf_df['Type_opinion'] == "majority")].groupby('decision_year')["mafia"].nunique().plot(kind='line')


plt.figure()
idf_df.groupby('Type_opinion')['mafia', 'cocaine', 'cannabis', 'rifle'].nunique().plot(kind='bar')


result.to_csv("result_data.csv", sep=";")

idf_df.to_csv("idf_data.csv", sep=";")


Narcotics= ["cannabis", "cocaine", "methamphetamine", "smart drugs", "marijuana", "mdma", "lsd", "ketamina", "heroin", "fentanyl"]
Weapons=["gun", "knife", "weapon", "firearm", "rifle", "carabine", "shotgun", "assaults rifle", "sword", "blunt objects"]
Investigation=["male", "female", "man", "woman", "girl", "boy","gang", "mafia", "serial killer", "rape", "thefts", "recidivism", "arrest", "ethnicity", "gender", "robbery", "cybercrime"]
all_wrd=Narcotics + Weapons + Investigation
not_presents=['smart drugs', 'LSD', 'carabine', 'cybercrime', 'serial killer', 'ketamina', 'Mdma', 'assaults rifle', 'blunt objects']
all_wrd_pre=  list(set(all_wrd) - set(not_presents))
result=pd.read_csv("result_data.csv",sep=";")
opinions = result["Type_opinion"]
single_str_opinion=result["result"]


generalist=["weapon", "firearm","gender", "recidivism", "arrest", "ethnicity"]
specific=["gun", "knife", "rifle", "shotgun", "sword","male", "female", "rape", "thefts"]


plt.figure()
result.groupby('Type_opinion')[generalist].nunique().plot(kind='bar')
plt.show()

result.groupby('court')[generalist].nunique().plot(kind='bar')

result.groupby('court').nunique().plot(kind='bar')


plt.figure()
result.groupby('Type_opinion')[specific].nunique().plot(kind='bar')
plt.show()



testgp = pd.DataFrame(result.groupby('Type_opinion')[all_wrd_pre].nunique())
testgp = pd.DataFrame(result.groupby('court')[all_wrd_pre].nunique())



testgp = testgp.div(testgp.sum(axis=1), axis=0)
a = result['Type_opinion'].unique()

result['Unnamed: 0'] = 1
fig1, ax1 = plt.subplots()
plot=ax1.pie(result.groupby('Type_opinion')['Unnamed: 0'].sum(), labels=a)

testgp.plot(kind='bar')


plt.figure()
testgp[generalist].plot(kind='bar')
testgp[specific].plot(kind='bar')
plt.show()

from skbio import DistanceMatrix
years=result['decision_year'].unique().tolist()
years = sorted(years)
result=pd.read_csv("result_data.csv",sep=";")
result=result.sort_values(by=['decision_year'])
#result_word=result[all_wrd_pre]
chunk_years = []
for i in years:
    c = result.loc[result.decision_year == i]
    c = pd.DataFrame(c)
    c = c[all_wrd_pre]
    #c = c.corr(method='pearson')
    chunk_years.append(c)


from skbio.stats.distance import mantel
#result=result.sort_values(by=['decision_year'])
#coeff, p_value = mantel(chunk_years[0],chunk_years[1])
#round(coeff, 4)
# d=chunk_years[0].corrwith(chunk_years[1], axis=0)
# d= dcor(chunk_years[0],chunk_years[1])
# d= dcor(chunk_years[1],chunk_years[2])
# d= dcor(chunk_years[5],chunk_years[6])

from scipy.spatial import distance_matrix
list_dist_y = []
for i in range(len(chunk_years)):
        c = distance_matrix(chunk_years[i],chunk_years[i + 1])
        list_dist_y.append(c)



import numpy as np
#49864/184 divide in chunk size of n rows
result = result.iloc[2:]
result = result[:-2]
result=result.sort_values(by=['decision_year'])
#result = result.drop(result.tail(2).index, inplace = True)
df_list_ordered=np.array_split(result[all_wrd_pre], 1662)
df_list_years=np.array_split(result['decision_year'], 1662)

list_range_y = []
for i in df_list_years:
    max_value = i.max()
    min_value = i.min()
    range_p = str(min_value) + '-' + str(max_value)
    list_range_y.append(range_p)


d1 = DistanceMatrix(df_list_ordered[0])
coeff, p_value = mantel(df_list_ordered[0],df_list_ordered[1])
round(coeff, 4)

from sklearn.cross_decomposition import CCA
cca = CCA(n_components=2)
df_list_ordered_haf=np.array_split(result[all_wrd_pre], 2)
#FIT ALL
cca.fit(df_list_ordered_haf[0],df_list_ordered_haf[1])

df_list_ordered=np.array_split(result[all_wrd_pre], 2)

X_c, Y_c = cca.transform(df_list_ordered[0],df_list_ordered[1])

cca.fit(df_list_ordered[1],df_list_ordered[2])
result_pre=result[all_wrd_pre]
cca.fit(result_pre[2],result_pre[3])



cca.score(df_list_ordered[0],df_list_ordered[1])
cca.score(df_list_ordered[1],df_list_ordered[2])
cca.score(df_list_ordered[2],df_list_ordered[3])

list_dist_y = []
for i in range(len((df_list_ordered))):
    if i != max(df_list_ordered):
        c = cca.score(df_list_ordered[i],df_list_ordered[i + 1])
        list_dist_y.append(c)
    else:
        c=cca.score(df_list_ordered[i],df_list_ordered[i])
        list_dist_y.append(c)



from sklearn.preprocessing import StandardScaler
list_dist_y = []
to_range = len((df_list_ordered))
for i in range(to_range):
    X=df_list_ordered[i]
    Y=df_list_ordered[i + 1]
    #X_mc=(X - X.mean()) / (X.std())
    #Y_mc=(Y - Y.mean()) / (Y.std())
    ca=CCA()
    X_mc=pd.DataFrame(StandardScaler().fit_transform(X))
    Y_mc =pd.DataFrame(StandardScaler().fit_transform(Y))
    ca.fit(X_mc,Y_mc)
    X_c,Y_c=ca.transform(X_mc,Y_mc)
    c= np.corrcoef(X_c[:,0],Y_c[:,0])[0,1]
        #c = cca.score(df_list_ordered[i],df_list_ordered[i + 1])
    list_dist_y.append(c)

import matplotlib.pyplot as plt
plt.plot((range(len((df_list_ordered)) - 1 )), score_array)
plt.show()

score_array = np.array(list_dist_y)
list_range_y_red=list_range_y[:-1]

plt.plot(list_range_y_red, score_array)
plt.show()
#tends towards 1 overtime, cause we have more data points from the same year.
plt.plot(list_range_y_red[1:10], score_array[1:10])
plt.show()

plt.plot(list_range_y_red[1:20], score_array[1:20])
plt.show()

plt.plot(list_range_y_red[20:38], score_array[20:38])
plt.show()

###

plt.plot(list_range_y_red[38:58], score_array[38:58])
plt.show()


plt.plot(list_range_y_red[138:158], score_array[138:158])
plt.show()
plt.plot(list_range_y_red[1:22], score_array[1:22])
plt.show()


#organize in to arg data
cca = CCA(n_components=2)

cca.fit(chunk_years[5],chunk_years[3])


from sklearn.preprocessing import StandardScaler
list_dist_y = []
to_range = len((df_list_ordered))
for i in range(to_range):
    X=df_list_ordered[i]
    Y=df_list_ordered[i + 1]
    #X_mc=(X - X.mean()) / (X.std())
    #Y_mc=(Y - Y.mean()) / (Y.std())
    ca=CCA(n_components=2)
    X_mc=pd.DataFrame(StandardScaler().fit_transform(X))
    Y_mc =pd.DataFrame(StandardScaler().fit_transform(Y))
    ca.fit(X_mc,Y_mc)
    X_c,Y_c=ca.transform(X_mc,Y_mc)
    c= np.corrcoef(X_c[:,0],Y_c[:,0])[0,1]
        #c = cca.score(df_list_ordered[i],df_list_ordered[i + 1])
    list_dist_y.append(c)


narcos=np.array_split(result[all_wrd_pre_narcos], 1662)
list_dist_y_narcos = []
to_range = len(narcos)
for i in range(to_range):
    X=narcos[i]
    Y=narcos[i + 1]
    #X_mc=(X - X.mean()) / (X.std())
    #Y_mc=(Y - Y.mean()) / (Y.std())
    ca=CCA(n_components=1)
    X_mc=pd.DataFrame(StandardScaler().fit_transform(X))
    Y_mc =pd.DataFrame(StandardScaler().fit_transform(Y))
    ca.fit(X_mc,Y_mc)
    X_c,Y_c=ca.transform(X_mc,Y_mc)
    c= np.corrcoef(X_c[:,0],Y_c[:,0])[0,1]
        #c = cca.score(df_list_ordered[i],df_list_ordered[i + 1])
    list_dist_y_narcos.append(c)



narcos=result[all_wrd_pre]
list_dist_y_narcos_wep = []
to_range = len(narcos)
#for i in range(to_range):

X=narcos[all_wrd_pre_narcos]
Y=narcos[all_wrd_pre_wep]
    #X_mc=(X - X.mean()) / (X.std())
    #Y_mc=(Y - Y.mean()) / (Y.std())
ca=CCA(n_components=1)
X_mc=pd.DataFrame(StandardScaler().fit_transform(X))
Y_mc =pd.DataFrame(StandardScaler().fit_transform(Y))
ca.fit(X_mc,Y_mc)
X_c,Y_c=ca.transform(X_mc,Y_mc)
c= np.corrcoef(X_c[:,0],Y_c[:,0])[0,1]
        #c = cca.score(df_list_ordered[i],df_list_ordered[i + 1])
list_dist_y_narcos_wep.append(c)




list_dist_y_narcos_pro=[]
X=narcos[all_wrd_pre_narcos]
Y=narcos[all_wrd_pre_process]
    #X_mc=(X - X.mean()) / (X.std())
    #Y_mc=(Y - Y.mean()) / (Y.std())
ca=CCA(n_components=1)
X_mc=pd.DataFrame(StandardScaler().fit_transform(X))
Y_mc =pd.DataFrame(StandardScaler().fit_transform(Y))
ca.fit(X_mc,Y_mc)
X_c,Y_c=ca.transform(X_mc,Y_mc)
c= np.corrcoef(X_c[:,0],Y_c[:,0])[0,1]
        #c = cca.score(df_list_ordered[i],df_list_ordered[i + 1])
list_dist_y_narcos_pro.append(c)



list_dist_y_pro_wep=[]
X=narcos[all_wrd_pre_wep]
Y=narcos[all_wrd_pre_process]
    #X_mc=(X - X.mean()) / (X.std())
    #Y_mc=(Y - Y.mean()) / (Y.std())
ca=CCA(n_components=1)
X_mc=pd.DataFrame(StandardScaler().fit_transform(X))
Y_mc =pd.DataFrame(StandardScaler().fit_transform(Y))
ca.fit(X_mc,Y_mc)
X_c,Y_c=ca.transform(X_mc,Y_mc)
c= np.corrcoef(X_c[:,0],Y_c[:,0])[0,1]
        #c = cca.score(df_list_ordered[i],df_list_ordered[i + 1])
list_dist_y_pro_wep.append(c)


plt.figure()
testgp[["weapon", "firearm","gun", "knife", "rifle", "shotgun", "sword"]].plot(kind='bar')
plt.show()

plt.figure()
testgp[["gender", "male","female", "man", "woman"]].plot(kind='bar')
plt.show()



plt.figure()
testgp[["gender", "male","female", "man", "woman","recidivism", "arrest", "ethnicity"]].plot(kind='bar')
plt.show()
#
# testgp["sum"] = testgp.sum(axis=1)
# # for i in testgp:
# #     print(i)
# #     for x in i:
# #         x = (x /testgp.sum(axis=1)) * 100
# #
# #     i = ( i /testgp.sum(axis=1)) * 100
#
#
# for i in range(testgp.shape[0]): #iterate over rows
#     for j in range(testgp.shape[1]): #iterate over columns
#         print( testgp.at[i,j])
#         testgp.at[i,j]  = (testgp.at[i,j] * 100)/testgp["sum"][i]
#         #get cell value
# def ratio(df):
#     for rowIndex, row in df.iterrows(): #iterate over rows
#          for columnIndex, value in row.items():
#              value = value * 100 / row.sum()
#     return df
#
# testgp = testgp.applymap( lambda x:  (x * 100 /testgp["sum"][1]))
# testgp.applymap( lambda x:  x * 100/testgp["sum"] )
#def ratio(item):
#    return item/testgp.sum(axis=1)
#testgp .applymap(ratio).head()
#testgp = testgp .applymap(ratio)
#testgp = testgp.apply( x/testgp.sum(axis=1))


##testgp = testgp.applymap( lambda x:  x /(testgp.sum(axis=1)) * 100,axis=1,result_type='expand')


plt.figure()
result.groupby('Type_opinion')["weapon", "firearm","gun", "knife", "rifle", "shotgun", "sword"].nunique().plot(kind='bar')
plt.show()
###check generic--->dissent majority--->specific
##proportionate_check_ratio grouped by category---plot it
for j in range(len(single_str_opinion)):
    single_str_opinion[j]=ast.literal_eval(single_str_opinion[j])
    single_str_opinion[j]=[i for i in single_str_opinion[j] if i in all_wrd]
    single_str_opinion[j]=' '.join(single_str_opinion[j])

single_str_opinion=single_str_opinion.values.tolist()


multiclass_dict_int={'concurrence':1,
 'concurring-in-part-and-dissenting-in-part':2,
 'dissent':3,
 'majority':4,
 'rehearing':5}

opinions=opinions.replace(multiclass_dict_int, inplace=True)

vocab_size = 31#len(max(single_str_opinion,key=len))
encoded_opi= [one_hot(o,vocab_size) for o in single_str_opinion]
max_lg=31#vocab_size
padded_opi = pad_sequences(encoded_opi,maxlen=max_lg, padding='post')
#non fattibile causa memoria
#droppare le cose

embeded_vector_size = 5

model = Sequential()
model.add(Embedding(vocab_size, embeded_vector_size,name="embedding", input_length=max_lg))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='softmax'))
X = padded_opi
y = opinions
y=y.to_numpy()
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(X, dummy_y, epochs=3, verbose=0)


#cor of single terms towards towards macro categories--->3 vectors
#single element value with single term from category
#avg() and kept a single number for each vectors
#good with stable cor, not so good with high var
#group cor by decision year




import numpy as np
from scipy.spatial.distance import pdist, squareform

def dcov(X, Y):
    """Computes the distance covariance between matrices X and Y.
    """
    n = X.shape[0]
    XY = np.multiply(X, Y)
    cov = np.sqrt(XY.sum()) / n
    return cov


def dvar(X):
    """Computes the distance variance of a matrix X.
    """
    return np.sqrt(np.sum(X ** 2 / X.shape[0] ** 2))


def cent_dist(X):
    """Computes the pairwise euclidean distance between rows of X and centers
     each cell of the distance matrix with row mean, column mean, and grand mean.
    """
    M = squareform(pdist(X))    # distance matrix
    rmean = M.mean(axis=1)
    cmean = M.mean(axis=0)
    gmean = rmean.mean()
    R = np.tile(rmean, (M.shape[0], 1)).transpose()
    C = np.tile(cmean, (M.shape[1], 1))
    G = np.tile(gmean, M.shape)
    CM = M - R - C + G
    return CM


def dcor(X, Y):
    """Computes the distance correlation between two matrices X and Y.
    X and Y must have the same number of rows.
    >>> X = np.matrix('1;2;3;4;5')
    >>> Y = np.matrix('1;2;9;4;4')
    >>> dcor(X, Y)
    0.76267624241686649
    """
    assert X.shape[0] == Y.shape[0]

    A = cent_dist(X)
    B = cent_dist(Y)

    dcov_AB = dcov(A, B)
    dvar_A = dvar(A)
    dvar_B = dvar(B)

    dcor = 0.0
    if dvar_A > 0.0 and dvar_B > 0.0:
        dcor = dcov_AB / np.sqrt(dvar_A * dvar_B)

    return dcor
