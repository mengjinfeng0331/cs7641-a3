from sklearn.cluster import KMeans
import load_data,os
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import util, pandas as pd
from sklearn.preprocessing import OneHotEncoder

SEED = 100
import plot
OUTPUT_DIR = 'kmeans'
if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

def kmeans(dataset, n_clusters,n_init,max_iter,metric):
    if dataset =='creditcard':
        X,y = load_data.load_creditcard_data()
    else:
        X,y = load_data.load_cancer_data()
        
    kmeans = KMeans(n_clusters=n_clusters,
                    n_init = n_init,
                    max_iter = max_iter,
                    random_state=SEED).fit(X)
    n_samples = X.shape[0]
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels,metric=metric)
    y_pred = kmeans.predict(X)
    cluster_acc = util.cluster_acc(y, y_pred)
    score = kmeans.score(X)
    
    return {'inertia':round(kmeans.inertia_/n_samples,1), 
            'silh':round(silhouette_avg,3),
            'cluster_acc': cluster_acc,
            'score': score
            }

dataset=['creditcard','cancer']
n_init = 10
max_iter = 300
n_clusters_list = [2,3,4,5,6,7,8,9,10,15,20,25]    
def part1():
    ''' run cluster on the original dataset
    
    '''
    df = pd.DataFrame()
    index = 0
    #metrics = ['manhattan', 'euclidean','cosine','l1','l2','cosine','cityblock']
    metrics = ['euclidean']
    for data in dataset:
        print(data)
        for n_cluster in n_clusters_list:
            for metric in metrics:
                results = kmeans(data, n_cluster, n_init, max_iter,metric)
                df.loc[index,'data'] = data
                df.loc[index,'metric'] = metric
                df.loc[index,'n_cluster'] = n_cluster
                for key, value in results.items():
                    df.loc[index, key] =value
                index +=1
    print(df)
    
    plot.plot_cluster_metrics(df, OUTPUT_DIR+os.sep + 'kmeans_metrics.png')
    return df
     
def part3():
    ''' run cluster on the dimension-reduction dataset
    
    '''
    def kmeans(path,n_clusters):
        if 'creditcard' in path:
            X,y = load_data.load_creditcard_data(path)
        else:
            X,y = load_data.load_cancer_data(path)
            
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=SEED).fit(X)
        n_samples = X.shape[0]
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        y_pred = kmeans.predict(X)
        cluster_acc = util.cluster_acc(y, y_pred)
        score = kmeans.score(X)
        
        return {'inertia':round(kmeans.inertia_/n_samples,1), 
                'silh':round(silhouette_avg,3),
                'cluster_acc': cluster_acc,
                'score': score
                }
    df = pd.DataFrame()
    index = 0
    alg_list= ['PCA','ICA','RP','RFE']
    
    for alg in alg_list:
        for data in dataset:
            for n_cluster in n_clusters_list:
                data_file = alg +os.sep + data+'.csv'
#                print(data_file)
                results = kmeans(data_file, n_cluster)
#                print(results)
                df.loc[index, 'alg'] = alg
                df.loc[index, 'data'] = data
                df.loc[index, 'n_cluster'] = n_cluster
                for key, value in results.items():
                    df.loc[index, key] = value
                index +=1
    return df



def part5():
    ''' run cluster on the dimension-reduction dataset
    
    '''
    def kmeans(path,n_clusters=10):
        if 'creditcard' in path:
            X,y = load_data.load_creditcard_data(path)
        else:
            X,y = load_data.load_cancer_data(path)
            
        model = KMeans(n_clusters=n_clusters,
                        random_state=SEED).fit(X)
        
        cluster_labels = model.fit_predict(X)
        cluster_labels = cluster_labels.reshape(-1,1)
        enc = OneHotEncoder()
        enc.fit(cluster_labels)
        X_predict = enc.transform(cluster_labels)
        X_predict = pd.DataFrame(X_predict.toarray())
        output_file = path.split('.csv')[0] + '_Kmeans_part5.csv'
        X_predict['label'] = y
        X_predict.to_csv(output_file)
        
    alg_list= ['PCA','ICA','RP','RFE']
    
    for alg in alg_list:
        for data in dataset:
            data_file = alg +os.sep + data+'.csv'
            print(data_file)
            results = kmeans(data_file, n_clusters=10)
            
            
df1 = part1()
df1['alg']='original'

df3 = part3()
df = pd.concat([df1,df3])
plot.plot_part3(df,OUTPUT_DIR+os.sep+'kmeans-part3.PNG')

part5()