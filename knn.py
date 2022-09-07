#-------------------------------------------------------------------------------
# Name:        knn
# Purpose:
#
# Author:      Cowbra
#
# Created:     18/05/2022
# Copyright:   (c) Cowbra 2022
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt


def min_max_scaling(x,min_x,max_x):
    return (x-min_x) / (max_x - min_x)



def normalization(dataframe):
    for column in dataframe:
        if column != "Classe":
            X_min = df[column].min()
            X_max = df[column].max()
            diff = X_max - X_min
            df[column] = df[column].apply(lambda x : min_max_scaling(x, X_min, X_max))




def euclidean_distance(ind_1, ind_2):
    ''' Calcule la distance euclidienne entre 2 individus'''
    ind_2 = ind_2[:-1] if len(ind_2)==11 else ind_2
    distance = [(a - b)**2 for a, b in zip(ind_1[:-1], ind_2)]
    distance = np.sqrt(sum(distance))
    return distance


def operate(k, individu):
    dataset = sorted(learning_df.values.tolist(), key = lambda x : euclidean_distance(x,individu))[:k]
    result=[0,0]
    for i in dataset:
        if i[10]==0:result[0]+=1
        else:result[1]+=1

    return np.argmax(result)


def knn(k,ind):
    individu_resultat = operate(k,ind)
    return 0 if individu_resultat ==0 else 1


def accuracy(k,show_matrix=False):
    accu_matrix = [[0,0],[0,0]]
    for ind in test_dataset:
        knn_result = knn(k,ind)
        if ind[10]==0:
            if knn_result  ==0:
                accu_matrix[0][0]+=1
            else :
                accu_matrix[0][1]+=1
        else :
            if knn_result ==1:
                accu_matrix[1][1]+=1
            else :
                accu_matrix[1][0]+=1
    if show_matrix:
        sns.heatmap(accu_matrix,square=True,annot=True,fmt='d',cbar=False)
        plt.show()

    return np.round((accu_matrix[0][0] + accu_matrix[1][1])/len(test_dataset),4)*100



def choose_best_k():
    result = [accuracy(i) for i in range(100)]
    return np.argmax(result),result[np.argmax(result)]

def best_k():
    result = choose_best_k()
    print(f"best k is: {result[0]}; accuracy : {result[1]} %")



def plot_correlation(data):
    rcParams['figure.figsize'] = 15, 20
    fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.show()
    fig.savefig('correlation.png')


if __name__ == '__main__':
    df2 = pd.read_csv('preTest.txt', sep=";", header=None)
    df = pd.read_csv('data.txt', sep=";", header=None)
    df.columns = ["a", "b", "c", "d", "e", "f", "g", "h", "i" ,"j","Classe"]
    df2.columns = ["a", "b", "c", "d", "e", "f", "g", "h", "i" ,"j","Classe"]

    df = pd.concat([df,df2] , ignore_index=True)
    df_class0 = df[df['Classe']== 0 ]
    df_class1 = df[df['Classe']== 1 ]

    print("objet classifiés en tant que '0': ",len(df_class0))
    print("objet classifiés en tant que '1': ",len(df_class1))

    #plot_correlation(df)
    #sns.countplot(x='Classe',data=df,label="Nombre")
    """
    colonnes=["a","b","c","d","e","f","g","h","i","j","Classe"]
    fig = plt.figure()
    sns.pairplot(data=df[colonnes],hue="Classe")
    plt.show()
    fig.savefig('2.png')
    """


    #Normalisation des données :
    normalization(df)

    learning_df = pd.concat([df_class0.sample(frac=0.6),df_class1.sample(frac=0.6)] , ignore_index=True)
    test_df = df.drop(learning_df.index)
    test_dataset = test_df.values.tolist()



    #best_k()
    print(accuracy(45))
