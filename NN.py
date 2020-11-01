# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
import load_data
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import backend as K
import time,os, plot
import pandas as pd
OUTPUT_DIR = 'NN'
if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def nn(X, y, act='relu',validation_split=0.33, epoch=50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
    model = Sequential()
    model.add(Dense(20, input_dim=X.shape[1], activation=act))
    model.add(Dense(5, activation=act))
    model.add(Dense(1, activation='sigmoid'))
#    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',auc])
    start = time.time()
    history = model.fit(X_train, y_train, epochs=epoch,validation_split=validation_split, batch_size=10,verbose=0)
    metrics = model.evaluate(X_test, y_test)
    end  = time.time()
    durartion = end-start
    return model, history.history, metrics, durartion

def part4():
    df = pd.DataFrame()
    index=0
    history_dict = {}
    alg_list= ['PCA','ICA','RP','RFE','original']
    for alg in alg_list:
        if alg =='original':
            X, y= load_data.load_creditcard_data()
        else:
            data_file = alg +os.sep + 'creditcard.csv'
            X, y= load_data.load_creditcard_data(data_file)
        model, history, metrics, duration = nn(X,y,act='relu',epoch=100 )
#        print(model.metrics_names)
#        print(metrics)
        df.loc[index,'alg'] = alg
        df.loc[index,'loss'] = metrics[0]
        df.loc[index,'acc'] = metrics[1]
        df.loc[index,'auc'] = metrics[2]
        df.loc[index,'duration'] = duration
        index+=1
        history_dict[alg] =  history
    df = df.round(3)
    return df, history_dict
        
df,history_dict =part4()
df.to_csv(OUTPUT_DIR+os.sep+'NN_df.csv')
plot.plot_part4_history(history_dict, OUTPUT_DIR+os.sep+'NN_training_acc.png')
plot.plot_part4_test_acc(df, OUTPUT_DIR+os.sep+'NN_test.png')
      
