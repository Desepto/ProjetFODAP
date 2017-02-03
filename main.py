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
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
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
algo_metrics['precision']=[]
algo_metrics['recall']=[]
algo_metrics['fscore']=[]
algo_metrics['support']=[]
algo_metrics['log_loss']=[]
algo_metrics['confusion_matrix']=[]

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

#même fonction mais c'est celle d'Eya 
def splitDataFrame(df, length): 
	list_df = list()
	rest = len(df)%length
	if rest==0:
		numberOfParts = len(df) // length 
	else:
		numberOfParts = len(df) // length+1
		for i in range(numberOfParts):
			list_df.append(df[i*length:(i+1)*length])
	return list_df
    
#fonction qui prepare les données sous un format supporté par tsfresh(quest 4)
def preparation(df,id,y,length=100):
	rdf=pd.DataFrame()
	classes= class_names
	print("preprocessing...")
	for cls in classes:
		print("===> C"+str(classes.index(cls)+1)+": "+cls)
		df_class=df[df.Class==classes.index(cls)+1]
		df_i=splitDataFrame(df_class,length) #splitDataFrameIntoSmaller(df_class,length) #fct d'Eya because ça à l'air plus logique quand même
		for df_ in df_i:
			y.append(cls)
			df_=df_.drop(labels="Class",axis=1)
			df_.insert(3,'id',id)
			id+=1
			rdf=rdf.append(df_)
	return rdf,id,y

#fonction qui generate les features à partir du dataframe temp master_df(quest 5)	
def get_features(master_df):
	features = ts.extract_features(master_df, column_id="id",feature_extraction_settings = MinimalFeatureExtractionSettings())
	return features
	
#split les features temps en set de train et teste pour l'apprentissage(quest 6)
def split_train_test(feature,y,test=0.25):
	X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=test, random_state=0)
	return X_train, X_test, y_train, y_test

##############################Déclaration de fonctions des algorithme de train#####################
#A demandé au moins 3 algo mais i guess ça fait pas de mal d'en faire genre 5...

#fonction de l'algorithme d'apprentissage KNN(quest 7)
def Knn_algorithm(X_train,y_train,X_test,y_test):
	clf = KNeighborsClassifier(n_neighbors=len(class_names))
	clf.fit(X_train, y_train)	
	y_predict = clf.predict(X_test)
	clf_probs = clf.predict_proba(X_test)
	
	get_metrics(y_test,y_predict,clf_probs)
	
#fonction de l'algorithme d'apprentissage DecisionTree(quest 7)
def DecisionTree_algorithm(X_train,y_train,X_test,y_test):
	clf = DecisionTreeClassifier()
	clf.fit(X_train,y_train)
	y_predict=clf.predict(X_test)
	clf_probs = clf.predict_proba(X_test)
	
	get_metrics(y_test,y_predict,clf_probs)
	
#fonction de l'algorithme d'apprentissage RandomForest(quest 7)
def RandomForest_algorithm(X_train,y_train,X_test,y_test):
	clf = RandomForestClassifier()
	clf.fit(X_train,y_train)
	y_predict=clf.predict(X_test)
	clf_probs = clf.predict_proba(X_test)
	
	get_metrics(y_test,y_predict,clf_probs)

#fonction de l'algorithme d'apprentissage AdaBoostClassifier(quest 7)
def AdaBoost_algorithm(X_train,y_train,X_test,y_test):
	clf= AdaBoostClassifier()
	clf.fit(X_train,y_train)
	y_predict=clf.predict(X_test)
	clf_probs = clf.predict_proba(X_test)
	
	get_metrics(y_test,y_predict,clf_probs)

#fonction de l'algorithme d'apprentissage GradientBoostingClassifier(quest 7)
def GradientBoosting_algorithm(X_train,y_train,X_test,y_test):
	clf= GradientBoostingClassifier()
	clf.fit(X_train,y_train)
	y_predict=clf.predict(X_test)
	clf_probs = clf.predict_proba(X_test)

	get_metrics(y_test,y_predict,clf_probs)

##############################Declaration de fonction du déroulement du script#####################

def get_metrics(y_test,y_predict,clf_probs):
	a,b,c,support =score(y_test, y_predict)
	algo_metrics['precision'].append(accuracy_score(y_test, y_predict))
	algo_metrics['recall'].append(recall_score(y_test, y_predict,average='macro'))
	algo_metrics['fscore'].append(f1_score(y_test, y_predict,average='macro'))
	algo_metrics['support'].append(support)
	algo_metrics['log_loss'].append(log_loss(y_test,clf_probs))
	algo_metrics['confusion_matrix'].append(confusion_matrix(y_test,y_predict))
	
def data_and_algorithm_exe():
	id=0
	y=[]
	##recup les données dans un dataframe pandas
	df=csv_to_dataframe(path)
	
	#Transforme les données en un format acceptable par tsfresh
	master_df,id,y=preparation(df,id,y,100)

	#Extrait les features des données master_df
	os.system('clear')
	X=pd.DataFrame()
	X=get_features(master_df)

	#Diviser les données en train et test sets (20% test)
	X_train, X_test, y_train, y_test=split_train_test(X,y,0.2)

	os.system('clear')
	print("Training...")
	#tester tous les algos sur nos données
	Knn_algorithm(X_train,y_train,X_test,y_test)
	DecisionTree_algorithm(X_train,y_train,X_test,y_test)
	RandomForest_algorithm(X_train,y_train,X_test,y_test)
	AdaBoost_algorithm(X_train,y_train,X_test,y_test)
	GradientBoosting_algorithm(X_train,y_train,X_test,y_test)
	
def comparaison(metric):
	algorithm = ['KNNeighbors', 'DecisionTree', 'RandomForest', 'AdaBoost', 'GradientBoostring']
	data = algo_metrics[metric]
	if(metric!="confusion_matrix" and metric!="support"):
		pos = np.arange(len(algorithm))
		width = 1.0     # gives histogram aspect to the bar diagram

		ax = plt.axes()
		ax.set_xticks(pos + (width / 2))
		ax.set_xticklabels(algorithm,rotation=10, rotation_mode="anchor", ha="right")
		plt.suptitle(metric, fontsize=14)
		plt.bar(pos,data,width)
		plt.show()
	else:
		for algo in algorithm:
			print(algo)
			print(data[algorithm.index(algo)])
			print("********")

def script():
	choice=0
	metrics_names =['precision','recall','fscore','support','log_loss','confusion_matrix']
	while choice==0:
		os.system('clear')
		print("*****************FODAP PROJECT*****************")
		print("***********************************************")
		print("1-Show plot of classes.")
		print("2-Load data and execute training algorithms.")
		print("3-exit.")
		print("***********************************************")
		
		#Demander le choix à l'utilisateur
		print("Choose what to execute first")
		while choice not in [1,2,3]:
			choice=int(input())
		
		os.system('clear')
		if(choice==3):
			print("Bye!")
		elif(choice==1):
			print("***** Classes Plot ******")
			#affiche les graphe 2D de chaque class sur le fichier 1.csv
			class_plot()
			print("press 0 to return")
			print("press whatever number to quite")
			choice=int(input())
		elif(choice==2):
			#charge les données, retreive les features et train sur les 5 algos
			data_and_algorithm_exe()
			while choice != 7 and choice != 0:
				os.system('clear')
				print("Traning done !")
				print("**************")
				print("Metrics comparing plots:")
				print("1- Precision.")
				print("2- Recall")
				print("3- Fscore")
				print("4- Support")
				print("5- Log_loss")
				print("6- Confusion matrix")
				print("-")
				print("press 0 to return")
				print("press 7 to quite")
				choice=int(input())
				if(choice!=0 and choice!=7):
					os.system('clear')
					comparaison(metrics_names[choice-1])
					if(choice==6 or choice==4):
						print("-")
						print("press 0 to return")
						print("press 8 to choose another metric")
						print("press 7 to quite")
						choice=int(input())
						
	
###################################################################################################
#kinda testing random code

script()

###################################################################################################
