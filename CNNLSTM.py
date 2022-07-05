import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *
from keras import callbacks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
def process(path):
        #importing the dataset
        data = pd.read_csv(path)
        print(data.isna().sum())
        data.rename(columns={'Result': 'target'}, inplace=True)

        data['target'] = data['target'].map({-1:0, 1:1})
        data['target'].unique()
        X = data.iloc[:,0:30].values.astype(int)
        y = data.iloc[:,30].values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.seed(7))
        model = Sequential()

        model.add(Dense(40, activation='relu',
                  kernel_initializer='uniform',input_dim=X.shape[1]))
        model.add(Dense(30, activation='relu',
                  kernel_initializer='uniform'))
        model.add(Dense(20, activation='relu',
                  kernel_initializer='uniform'))
        model.add(Dense(10, activation='relu',
                  kernel_initializer='uniform'))
        model.add(Dense(1,  activation='sigmoid', 
                  kernel_initializer='uniform'))

        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        print(model.summary())
        es_cb = callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=5)
        history = model.fit(X_train, y_train, batch_size=64, epochs=128, verbose=2)
        model.save('results/CNNLSTM.h5');
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train'],loc='upper left')
        plt.savefig('results/CNNLSTM Loss.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()

        #train and validation accuracy
        plt.plot(history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train'],loc='upper left')
        plt.savefig('results/CNNLSTM Accuracy.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()

       
process("data.csv")
        
        
