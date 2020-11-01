from sklearn import mixture
import load_data,os,plot
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import FastICA
from sklearn import random_projection
import util, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
SEED = 1
OUTPUT_DIR = 'ICA'
if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

def ica(dataset, n_components,save_to_file=False):
    if dataset =='creditcard':
        X,y = load_data.load_creditcard_data()
    else:
        X,y = load_data.load_cancer_data()
    
    model = FastICA(n_components=n_components, random_state=SEED).fit(X)
    n_samples = X.shape[0]
    ## average log-likelihood of all samples
    X_fitted = model.transform(X)
    X_inverse = model.inverse_transform(X_fitted)
    dist = np.linalg.norm(X - X_inverse)/n_samples
    kurt = pd.DataFrame(X_fitted)
    kurt = kurt.kurt(axis=0)
    kurt= kurt.abs().mean()
    if save_to_file:
        X_fitted = pd.DataFrame(X_fitted)
        X_fitted['label']= y.values
        X_fitted.to_csv(OUTPUT_DIR+os.sep+dataset+'.csv')    
    return dist, kurt

dataset=['creditcard','cancer']
n_init = 10
max_iter = 300
n_components_list = range(1,15)
df = pd.DataFrame()
index=0

for data in dataset:
    print(data)
    for n_cluster in n_components_list:
        dist, kurt = ica(data, n_cluster)
        df.loc[index,'data'] = data
        df.loc[index,'n_cluster']= int(n_cluster)
#        df.loc[index, 'variance']= variance[index%len(n_components_list)]
#        df.loc[index, 'cumsum']= cumsum[index%len(n_components_list)]
        df.loc[index, 'reconstruction_error']= dist
        df.loc[index, 'kurt']= kurt
        index+=1
df['n_cluster'] =df['n_cluster'].astype(int)

def plot_cumsum(df, title, output_file):
    f, axes = plt.subplots(1, 2, figsize=(10,5))
    
#    df['n_cluster'] = df['n_cluster'].astype(int)
    f.tight_layout()

    credit_df = df[df['data']=='creditcard']
    sns.lineplot(y="kurt", x= "n_cluster", data=credit_df,label='kurt',marker='o', ax=axes[0], legend=False)
    axes[0].set_ylabel('kurt')
    
    ax2 = axes[0].twinx()
    sns.lineplot(y="reconstruction_error", x= "n_cluster", data=credit_df,label='reconstruction_error',legend=False,color='orange',marker='o', ax=ax2)
    axes[0].set(title='CreditCard')
    axes[0].figure.legend()
    ax2.set_ylabel('error')

    plt.subplots_adjust(wspace = 0.3)
    cancer_df = df[df['data']=='cancer']
#    print(cancer_df)
    sns.lineplot(y="kurt", x= "n_cluster", data=cancer_df,label='kurt',marker='o', ax=axes[1],legend=False)
    ax2 = axes[1].twinx()
    sns.lineplot(y="reconstruction_error", x= "n_cluster", data=cancer_df,label='reconstruction_error',color='orange',legend=False,marker='o', ax=ax2).set(title='Cancer',ylabel='score')
#    axes[1].set_ylim(0,1)
    axes[1].set(title='Cancer')
    ax2.set_ylabel('error')
    
    f.savefig(output_file)
    
plot_cumsum(df, 'ICA',OUTPUT_DIR +os.sep+'ICA.png')
ica('creditcard', 6,save_to_file=True)
ica('cancer', 8,save_to_file=True)

