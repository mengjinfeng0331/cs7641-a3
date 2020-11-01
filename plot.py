import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

plt.tight_layout()
#sns.set_theme()
sns.set_style("darkgrid")
OUTPUT_DIR = 'results'

def plot_data_heatmap(X,y,title, output_file):
    df = X.copy()
    df['label']=y
    corr = df.corr()
    plt.figure()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
        )
    ax.figure.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right')
#    ax.savefig(output_file)
    ax.set_title(title)
    plt.savefig(output_file)

def plot_cluster_metrics(metrics_df, output_file):
    f, axes = plt.subplots(1, 2, figsize=(10,5))
    metrics_df['n_cluster'] = metrics_df['n_cluster'].astype(int)
    
    credit_df = metrics_df[metrics_df['data']=='creditcard']
    sns.lineplot(y="silh", x= "n_cluster", data=credit_df,label='silh_score',marker='o', ax=axes[0])
    ax = sns.lineplot(y="cluster_acc", x= "n_cluster", data=credit_df,label='cluster_accuracy',marker='o', ax=axes[0]).set(title='CreditCard',ylabel='score')
#    plt.ylim(0,1)
    
    cancer_df = metrics_df[metrics_df['data']=='cancer']
    sns.lineplot(y="silh", x= "n_cluster", data=cancer_df,label='silh_score',marker='o', ax=axes[1])
    sns.lineplot(y="cluster_acc", x= "n_cluster", data=cancer_df,label='cluster_accuracy',marker='o', ax=axes[1]).set(title='Cancer',ylabel='score')
#    plt.ylim(0,1)
    
    f.savefig(output_file)
    
def plot_cumsum(df, title, output_file):
    f, axes = plt.subplots(1, 2, figsize=(10,5))
    
#    df['n_cluster'] = df['n_cluster'].astype(int)
    
    credit_df = df[df['data']=='creditcard']
    sns.lineplot(y="cumsum", x= "n_cluster", data=credit_df,label='variance(cumsum)',marker='o', ax=axes[0])
    sns.lineplot(y="reconstruction_error", x= "n_cluster", data=credit_df,label='reconstruction_error',marker='o', ax=axes[0]).set(title='CreditCard',ylabel='score')
    axes[0].set_ylim(0,1)

    cancer_df = df[df['data']=='cancer']
#    print(cancer_df)
    sns.lineplot(y="cumsum", x= "n_cluster", data=cancer_df,label='variance(cumsum)',marker='o', ax=axes[1])
    sns.lineplot(y="reconstruction_error", x= "n_cluster", data=cancer_df,label='reconstruction_error',marker='o', ax=axes[1]).set(title='Cancer',ylabel='score')
    axes[1].set_ylim(0,1)
    
    f.savefig(output_file)
    
def plot_part3(df, output_file):
    df.n_cluster = df.n_cluster.astype(int)
    f, axes = plt.subplots(2, 2, figsize=(10,10))
    
    alg_list = ['PCA','ICA','RP','RFE','original']
    data_list = ['creditcard','cancer']
    for i, data in enumerate(data_list):
        for alg in alg_list:
            sub_df = df[df['data']== data]
            axes[0][i].plot('n_cluster','silh','o-',data=sub_df[sub_df['alg']==alg], label=alg)
            axes[1][i].plot('n_cluster','cluster_acc','o-',data=sub_df[sub_df['alg']==alg], label=alg)
            
        axes[0][i].legend()
        axes[0][i].set_xlabel('n_cluster')
        axes[0][i].set_ylabel('silh score')
        axes[0][i].set_title(data+'_silh')
        axes[1][i].legend()
        axes[1][i].set_xlabel('n_cluster')
        axes[1][i].set_ylabel('accuracy')
        axes[1][i].set_title(data+'_accuracy') 
    f.savefig(output_file)
    
    
def plot_part4_test_acc(df, output_file):
#    plt.figure()
#    plt.bar('alg','acc',data=df)
#    plt.savefig(output_file)
    plt.figure()
    plt.plot('acc','auc','o', data=df)
    for i, row in df.iterrows():
        plt.annotate(row['alg'], (row['acc'], row['auc']), fontsize=15)
    plt.xlabel('test accuracy')
    plt.ylabel('test AUC')
    plt.title('NN Performance on test data')
    plt.savefig(output_file)
    
    
def plot_part4_history(history_dict, output_file):
    f, axes = plt.subplots(1, 2, figsize=(10,5))
    
    for key, history in history_dict.items():
        axes[0].plot(history['val_acc'],label=key)
    axes[0].legend()
    axes[0].set_title('Validation Accuracy during training')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('accuracy')
    
    for key, history in history_dict.items():
        axes[1].plot(history['val_loss'],label=key)
    axes[1].legend()
    axes[1].set_title('Validation loss during training')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')    
    f.savefig(output_file)
    
def plot_confusion_matrix(cm, classes,
                        normalize=True,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    sns.heatmap(cm, cmap=cmap)
#    sns.heatmap(cm, cmap=cmap,annot=True)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#    plt.savefig('test.png')
