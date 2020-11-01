from sklearn import mixture
import load_data,os
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import util, pandas as pd
import plot

SEED = 1
OUTPUT_DIR = 'PCA'
if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

def pca(dataset, n_components, save_to_file=False):
    if dataset =='creditcard':
        X,y = load_data.load_creditcard_data()
    else:
        X,y = load_data.load_cancer_data()
        
    model = PCA(n_components=n_components,
                    random_state=SEED).fit(X)
    n_samples = X.shape[0]
    ## average log-likelihood of all samples
    score  = model.score(X)
    variance = model.explained_variance_ratio_
    cumsum = np.cumsum(model.explained_variance_ratio_)
    X_fitted = model.transform(X)
    X_inverse = model.inverse_transform(X_fitted)
    dist = np.linalg.norm(X - X_inverse)/n_samples
    if save_to_file:
        X_fitted = pd.DataFrame(X_fitted)
        X_fitted['label']= y.values
        X_fitted.to_csv(OUTPUT_DIR+os.sep+dataset+'.csv')
    return score, variance, cumsum, dist

dataset=['creditcard','cancer']
n_init = 10
max_iter = 300
n_components_list = range(1,15)    
df = pd.DataFrame()
index=0
for data in dataset:
    print(data)
    for n_cluster in n_components_list:
        score,variance,cumsum ,dist = pca(data, n_cluster)
        df.loc[index,'data'] = data
        df.loc[index,'n_cluster']= int(n_cluster)
        df.loc[index, 'score']= score
        df.loc[index, 'variance']= variance[index%len(n_components_list)]
        df.loc[index, 'cumsum']= cumsum[index%len(n_components_list)]
        df.loc[index, 'reconstruction_error']= dist
        index+=1
df['n_cluster'] =df['n_cluster'].astype(int)
plot.plot_cumsum(df, 'PCA',OUTPUT_DIR +os.sep+'PCA.png')

pca('creditcard', 8, save_to_file=True)
pca('cancer', 8, save_to_file=True)
