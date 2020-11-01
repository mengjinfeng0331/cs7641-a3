from sklearn import mixture
import load_data,os
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import util, pandas as pd

SEED = 100
OUTPUT_DIR = 'RFE'
if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

def rfe(dataset, n_components, save_to_file=False):
    if dataset =='creditcard':
        X,y = load_data.load_creditcard_data()
    else:
        X,y = load_data.load_cancer_data()

    estimator = SVR(kernel="linear")
    model = RFE(estimator,n_features_to_select=n_components, step=1)
    model = model.fit(X, y)
    
    n_samples = X.shape[0]
    
    X_fitted = model.transform(X)
    kurt = pd.DataFrame(X_fitted)
    kurt = kurt.kurt(axis=0)
    kurt= kurt.abs().mean()
    X_inverse = model.inverse_transform(X_fitted)
    reconstruction_error = np.linalg.norm(X - X_inverse)/n_samples
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

for data in dataset:
    print(data)
    for n_cluster in n_components_list:
        kurt, error = rfe(data, n_cluster)
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
    sns.lineplot(y="kurt", x= "n_cluster", data=credit_df,label='kurt',marker='o', ax=axes[0])
    sns.lineplot(y="reconstruction_error", x= "n_cluster", data=credit_df,label='reconstruction_error',marker='o', ax=axes[0]).set(title='CreditCard',ylabel='score')
#    axes[0].set_ylim(0,1)

    cancer_df = df[df['data']=='cancer']
#    print(cancer_df)
    sns.lineplot(y="kurt", x= "n_cluster", data=cancer_df,label='kurt',marker='o', ax=axes[1])
    sns.lineplot(y="reconstruction_error", x= "n_cluster", data=cancer_df,label='reconstruction_error',marker='o', ax=axes[1]).set(title='Cancer',ylabel='score')
#    axes[1].set_ylim(0,1)
    
    f.savefig(output_file)
    
plot_cumsum(df, 'RFE',OUTPUT_DIR +os.sep+'RFE.png')
rfe('creditcard', 9,save_to_file=True)
rfe('cancer', 8 ,save_to_file=True)
