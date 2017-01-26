#!/usr/bin/python3
# -*- coding:utf-8 -*-

import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt #Package à installer avec tkinter (python-tk) too 
from tsfresh.feature_extraction import MinimalFeatureExtractionSettings
import tsfresh as ts
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

##################################Declaration de vars##############################################

#path relatif du repository des data
path='ds/'

#dataframe et list de class et id temps (variable demandé dans le doc)
master_df=pd.DataFrame()
class_names=["Working at Computer","Standing Up, Walking and Going up\down stairs","Standing","Walking","Going Up\Down Stairs","Walking and Talking with Someone","Talking while Standing"]
id=0
y=[]

#dictionnaire de metrics pour chaque algo de train (variable perso)
algo_metrics={}

##############################Déclaration de fonctions de preprocessing des données################

#fonction qui read les fichier et met les données en dataframe(quest 1,2)
def csv_to_dataframe(path):
	files=glob.glob(os.path.join(path,'*.csv'))
	list_=[]
	frame=pd.DataFrame([],columns= ["step", "x", "y", "z","Class"])
	for infile in files:
		df=pd.read_csv(infile,names = ["step", "x", "y", "z","Class"])
		list_.append(df)
	frame=pd.concat(list_)

	#drop la column "step" et les rows avec Class=0
	frame=frame.drop(labels="step",axis=1)
	frame=frame[frame.Class != 0]
	return frame



#fonction qui plot un graph pour chaque class sur le fichier 1.csv(quest 3)
def class_plot():
	df=pd.read_csv("ds/1.csv",names = ["step", "x", "y", "z","Class"])
	df=df.drop(labels="step",axis=1)
	df=df[df.Class != 0]
	for i in range(len(class_names)):
		df_=df[df.Class==(i+1)]
		plt.plot(np.arange(len(df_)),list(df_.x))
		plt.plot(np.arange(len(df_)),list(df_.y))
		plt.plot(np.arange(len(df_)),list(df_.z))
		plt.suptitle(class_names[i], fontsize=14)
		plt.legend(['x', 'y', 'z'], loc='upper left')
		plt.show()

#fonction qui divide les df_class en partie de tailles length because np.array_split fait n'importe quoi
def splitDataFrameIntoSmaller(df, chunkSize): 
	listOfDf = list()
	numberChunks = len(df) // chunkSize + 1
	for i in range(numberChunks):
		listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
	return listOfDf
    
#fonction qui prepare les données sous un format supporté par tsfresh(quest 4)
def preparation(df,id,y,length=100):
	rdf=pd.DataFrame()
	classes= class_names
	#classes_nb= [ i for i in range(1,8)] #sert à rien
	for cls in classes:
		print(cls)
		df_class=df[df.Class==classes.index(cls)+1]
		df_i=splitDataFrameIntoSmaller(df_class,length)
		for df_ in df_i:
			y.append(cls)
			df_=df_.drop(labels="Class",axis=1)
			df_.insert(3,'id',id)
			id+=1
			rdf=rdf.append(df_)
		print("fin")
	return rdf,id,y

#fonction qui generate les features à partir du dataframe temp master_df(quest 5)	
def get_features(master_df):
	features = ts.extract_features(master_df, column_id="id",feature_extraction_settings = MinimalFeatureExtractionSettings())
	return features
	
#split les features temps en set de train et teste pour l'apprentissage(quest 6)
def split_train_test(feature,y,test=0.25):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=0)
	return X_train, X_test, y_train, y_test

##############################Déclaration de fonctions des algorithme de train#####################
#A demandé au moins 3 algo mais i guess ça fait pas de mal d'en faire genre 5...

#fonction de l'algorithme d'apprentissage KNN(quest 7)
def Knn_algorithm(X_train,y_train,X_test):
	clf = KNeighborsClassifier(n_neighbors=len(class_names))
	clf.fit(X_train, y_train)	
	y_predict = clf.predict(X_test)
	return clf, y_predict

#fonction de l'algorithme d'apprentissage DecisionTree(quest 7)
def DecisionTree_algorithm(X_train,y_train,X_test):
	clf = DecisionTreeClassifier()
	clf.fit(X_train,y_train)
	y_predict=clf.predict(X_test)
	return clf, y_predict
	
#fonction de l'algorithme d'apprentissage RandomForest(quest 7)
def RandomForest_algorithm(X_train,y_train,X_test):
	clf = RandomForestClassifier()
	clf.fit(X_train,y_train)
	y_predict=clf.predict(X_test)
	return clf,y_predict


#fonction de l'algorithme d'apprentissage AdaBoostClassifier(quest 7)
def AdaBoost_algorithm(X_train,y_train,X_test):
	clf= AdaBoostClassifier()
	clf.fit(X_train,y_train)
	y_predict=clf.predict(X_test)
	return clf,y_predict
	
#fonction de l'algorithme d'apprentissage GradientBoostingClassifier(quest 7)
def GradientBoosting_algorithm(X_train,y_train,X_test):
	clf= GradientBoostingClassifier()
	clf.fit(X_train,y_train)
	y_predict=clf.predict(X_test)
	return cld, y_predict
	
###################################################################################################
#kinda testing random code

##recup les données dans un dataframe pandas
df=csv_to_dataframe(path)

##affiche les graphe 2D de chaque class sur le fichier 1.csv
#class_plot()

##Transforme les données en un format acceptable par tsfresh
master_df,id,y=preparation(df,id,y,100)

##Extrait les features des données master_df
X=get_features(master_df)

##Diviser les données en train et test sets
X_train, X_test, y_train, y_test=split_train_test(X,y,0.2)

##tester KNN sur nos données
clf,y_predict = Knn_algorithm(X_train,y_train,X_test)

##tester les mesure de accuracy etc pour KNN
precision, recall, fscore, support = score(y_test, y_predict)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

############################# Si on a le temps...maybe interface  ###############################
