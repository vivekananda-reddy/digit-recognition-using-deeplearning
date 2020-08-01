import numpy as np
import pandas as pd





def create_train_test_set():
    df_train=pd.read_csv("mnist_train.csv")
    df_test=pd.read_csv("mnist_test.csv")

    rows_train, columns_train = df_train.shape
    rows_test, columns_test = df_test.shape



    y_train=df_train['label']
    y_test=df_test['label']


    x_train=df_train.drop(["label"], axis=1)
    x_test=df_test.drop(["label"], axis=1)



    y_train=y_train.to_numpy()
    y_train=y_train.reshape(rows_train,1)
    y_train=np.transpose(y_train)

    x_train=np.transpose(x_train.to_numpy())

    y_test = y_test.to_numpy()
    y_test = y_test.reshape(rows_test, 1)
    y_test = np.transpose(y_test)

    x_test=np.transpose(x_test.to_numpy())


    y_train = np.eye(10)[y_train.reshape(-1)].T #convert digits to column matrices
    y_test = np.eye(10)[y_test.reshape(-1)].T   #convert digits to column matrices



    print("Y_train shape=",y_train.shape)
    print("X_train shape=",x_train.shape)
    print("Y_test shape=",y_test.shape)
    print("X_test shape=",x_test.shape)


    return x_train,y_train,x_test,y_test

create_train_test_set()