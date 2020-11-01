from sklearn import mixture
import load_data,os
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import util, pandas as pd
import plot
from sklearn.preprocessing import OneHotEncoder

SEED = 1
OUTPUT_DIR = 'EM'
if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
def gmm(dataset, n_components,n_init,max_iter, metric):
    if dataset =='creditcard':
        X,y = load_data.load_creditcard_data()
    else:
        X,y = load_data.load_cancer_data()
        
    model = mixture.GaussianMixture(n_components=n_components,
                    n_init = n_init,
                    max_iter = max_iter,
                    random_state=SEED).fit(X)
    
    cluster_labels = model.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels, metric=metric)
    y_pred = model.predict(X)
    cluster_acc = util.cluster_acc(y, y_pred)
    
#    return {'inertia':round(model.inertia_,1), 'silh':round(silhouette_avg,3)}
    return {'silh':round(silhouette_avg,3), 
            'cluster_acc': cluster_acc,
            'aic': model.aic(X),
            'bic':model.bic(X)
            }

dataset=['creditcard','cancer']
n_init = 10
max_iter = 300
n_components_list =[2,3,4,5,6,7,8,9,10,15,20,25]
index = 0
df = pd.DataFrame()
metrics = ['euclidean']

def part1():
    ''' run cluster on the original dataset
    
    '''
    index=0
    for data in dataset:
        print(data)
        for n_cluster in n_components_list:
            for metric in metrics:
                results = gmm(data, n_cluster, n_init, max_iter, metric)
                df.loc[index,'data'] = data
                df.loc[index,'n_cluster'] = n_cluster
                df.loc[index,'metric'] = metric
                for key, value in results.items():
                    df.loc[index, key] =value
                index +=1
    plot.plot_cluster_metrics(df, OUTPUT_DIR+os.sep + 'gmm_metrics.png')
    return df

def plot_aic_bic(metrics_df, output_file):
    f, axes = plt.subplots(1, 2, figsize=(10,5))
    metrics_df['n_cluster'] = metrics_df['n_cluster'].astype(int)
    
    credit_df = metrics_df[metrics_df['data']=='creditcard']
    sns.lineplot(y="aic", x= "n_cluster", data=credit_df,label='aic',marker='o', ax=axes[0])
    sns.lineplot(y="bic", x= "n_cluster", data=credit_df,label='bic',marker='o', ax=axes[0]).set(title='CreditCard',ylabel='score')

    cancer_df = metrics_df[metrics_df['data']=='cancer']
    sns.lineplot(y="aic", x= "n_cluster", data=cancer_df,label='aic',marker='o', ax=axes[1])
    sns.lineplot(y="bic", x= "n_cluster", data=cancer_df,label='bic',marker='o', ax=axes[1]).set(title='Cancer',ylabel='score')
    f.savefig(output_file)
    
#plot.plot_cluster_metrics(df, OUTPUT_DIR+os.sep + 'gmm_metrics.png')
#plot_aic_bic(df, OUTPUT_DIR+os.sep + 'gmm_metrics-2.png')


def part3():
    ''' run cluster on the dimension-reduction dataset
    
    '''
    def GMM(path,n_clusters):
        if 'creditcard' in path:
            X,y = load_data.load_creditcard_data(path)
        else:
            X,y = load_data.load_cancer_data(path)
            
        model = mixture.GaussianMixture(n_components=n_clusters,
                        random_state=SEED).fit(X)
        
        cluster_labels = model.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        y_pred = model.predict(X)
        cluster_acc = util.cluster_acc(y, y_pred)
        
        return {'silh':round(silhouette_avg,3), 
                'cluster_acc': cluster_acc,
                'aic': model.aic(X),
                'bic':model.bic(X)
                }
    df = pd.DataFrame()
    index = 0
    alg_list= ['PCA','ICA','RP','RFE']
    
    for alg in alg_list:
        for data in dataset:
            for n_cluster in n_components_list:
                data_file = alg +os.sep + data+'.csv'
                results = GMM(data_file, n_cluster)
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
    def GMM(path,n_clusters=10):
        if 'creditcard' in path:
            X,y = load_data.load_creditcard_data(path)
        else:
            X,y = load_data.load_cancer_data(path)
            
        model = mixture.GaussianMixture(n_components=n_clusters,
                        random_state=SEED).fit(X)
        
        cluster_labels = model.fit_predict(X)
        cluster_labels = cluster_labels.reshape(-1,1)
        enc = OneHotEncoder()
        enc.fit(cluster_labels)
        X_predict = enc.transform(cluster_labels)
        X_predict = pd.DataFrame(X_predict.toarray())
        output_file = path.split('.csv')[0] + '_EM_part5.csv'
        X_predict['label'] = y
        X_predict.to_csv(output_file)
        
    alg_list= ['PCA','ICA','RP','RFE']
    
    for alg in alg_list:
        for data in dataset:
            data_file = alg +os.sep + data+'.csv'
            print(data_file)
            results = GMM(data_file, n_clusters=10)

df1 = part1()
df1['alg']='original'

df3 = part3()
df = pd.concat([df1,df3])

plot.plot_part3(df,OUTPUT_DIR+os.sep+'EM-part3.PNG')

part5()