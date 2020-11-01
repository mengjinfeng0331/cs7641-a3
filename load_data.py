import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import plot,os

creditcard_file = 'data/creditcard_undersample.csv'
cancer_file = 'data/cancer.csv'

def load_creditcard_data(path=None):
    """
    Loads the creditcard Classification Data Set
    :param filename: path to csv file
    :return: X (data) and y (labels)
    """
    if path is None:
        df = pd.read_csv(creditcard_file, index_col=0)
        y = df['Class']
        X = df.drop('Class', axis=1)
    else:
        df = pd.read_csv(path, index_col=0)
        y = df['label']
        X = df.drop('label', axis=1)
    return X, y

def load_cancer_data(path=None):
    """
    Loads the creditcard Classification Data Set
    :param filename: path to csv file
    :return: X (data) and y (labels)
    """
    if path is None:
        df = pd.read_csv(cancer_file)
        drop_list = ['perimeter_mean',
                      'radius_mean',
                      'compactness_mean',
                      'concave points_mean',
                      'radius_se','perimeter_se',
                      'radius_worst',
                      'perimeter_worst',
                      'compactness_worst',
                      'concave points_worst',
                      'compactness_se',
                      'concave points_se',
                      'texture_worst',
                      'area_worst',
                      'Unnamed: 32',
                      'id',
                      ]
        df = df.drop(drop_list,axis = 1) 
        y = df['diagnosis']
        mapping = {'B':0, 'M':1}
        y = y.replace(mapping)
        std_scaler= StandardScaler()
        X_df = df.drop('diagnosis', axis=1)
        X = std_scaler.fit_transform(X_df)
        X = pd.DataFrame(X, index=X_df.index, columns=X_df.columns)
    else:
        df = pd.read_csv(path,index_col=0)
        y = df['label']
        X = df.drop('label', axis=1)
    return X, y

if '__main__' == __name__:
    OUTPUT_DIR= 'results'
    X,y = load_creditcard_data()
    plot.plot_data_heatmap(X,y,'creditcard_heatmap',OUTPUT_DIR+os.sep+ 'credit.png')
    X,y = load_cancer_data()
    plot.plot_data_heatmap(X,y,'cancer_heatmap',OUTPUT_DIR+os.sep+ 'cancer.png')    