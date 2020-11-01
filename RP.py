from sklearn import mixture
import load_data,os
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.random_projection import GaussianRandomProjection
import util, pandas as pd
import seaborn as sns
SEED = 10
OUTPUT_DIR = 'RP'
if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

def rp(dataset, n_components, save_to_file=False, seed = 1):
    if dataset =='creditcard':
        X,y = load_data.load_creditcard_data()
    else:
        X,y = load_data.load_cancer_data()
    model = GaussianRandomProjection(n_components=n_components, random_state=seed)
    model.fit(X)
    ## average log-likelihood of all samples
    X_fitted = model.transform(X)
    kurt = pd.DataFrame(X_fitted)
    kurt = kurt.kurt(axis=0)
    kurt= kurt.abs().mean()
    reconstruction_error =util.reconstructionError(model, X)
    if save_to_file:
        X_fitted = pd.DataFrame(X_fitted)
        X_fitted['label']= y.values
        X_fitted.to_csv(OUTPUT_DIR+os.sep+dataset+'.csv')  
    return  kurt, reconstruction_error

dataset=['creditcard','cancer']
n_init = 10
max_iter = 300
n_components_list = range(2,11)    
index=1
df = pd.DataFrame()
seed_list = range(30)
for seed in seed_list :
    for data in dataset:
        print(data)
        for n_cluster in n_components_list:
            kurt, error = rp(data, n_cluster,seed=seed)
            df.loc[index,'seeed'] = seed
            df.loc[index,'data'] = data
            df.loc[index,'n_cluster']= int(n_cluster)
    #        df.loc[index, 'variance']= variance[index%len(n_components_list)]
    #        df.loc[index, 'cumsum']= cumsum[index%len(n_components_list)]
            df.loc[index, 'reconstruction_error']= error
            df.loc[index, 'kurt']= kurt
            index+=1
df['n_cluster'] =df['n_cluster'].astype(int) 

def plot_cumsum(df, title, output_file):
    f, axes = plt.subplots(1, 2, figsize=(10,5))
    
#    df['n_cluster'] = df['n_cluster'].astype(int)
    
    credit_df = df[df['data']=='creditcard']
    sns.lineplot(y="kurt", x= "n_cluster", data=credit_df,label='kurt',marker='o', ax=axes[0], legend=False)
#    ax2 = axes[0].twinx()

    sns.lineplot(y="reconstruction_error", x= "n_cluster", data=credit_df,color='orange',legend=False,label='reconstruction_error',marker='o', ax=axes[0]).set(title='CreditCard',ylabel='error')
    plt.subplots_adjust(wspace = 0.4)
    axes[0].figure.legend()

    cancer_df = df[df['data']=='cancer']
#    print(cancer_df)
    sns.lineplot(y="kurt", x= "n_cluster", data=cancer_df,label='kurt',marker='o', legend=False,ax=axes[1])
    ax2 = axes[1].twinx()

    sns.lineplot(y="reconstruction_error", x= "n_cluster", data=cancer_df,color='orange',legend=False,label='reconstruction_error',marker='o', ax=axes[1]).set(title='Cancer',ylabel='error')
#    axes[1].set_ylim(0,1)
    
    f.savefig(output_file)
    
plot_cumsum(df, 'RP',OUTPUT_DIR +os.sep+'RP.png')
rp('creditcard', 9,save_to_file=True)
rp('cancer', 8 ,save_to_file=True)

