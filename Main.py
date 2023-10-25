#encoding=utf-8
import tensorflow.compat.v1 as tf
from keras.layers import Add, Concatenate, BatchNormalization, Input, Dense, LSTM, GRU, merge ,Conv1D,Dropout,Bidirectional,Multiply,MaxPooling1D,Flatten, Permute, Reshape
from keras.models import Model
from keras import backend as K
from attention_utils import get_activations,  get_data,get_data_recurrent
from keras.layers.core import Lambda, RepeatVector
from keras.layers import merge
from keras.layers.core import *
import tensorflow as tf
from keras.models import *
from keras import optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)
    
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def attention_3d_block2(inputs, single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
  #  print ("input:", inputs.shape)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
   # print ("a:", a.shape)
    a_probs = Permute((2, 1))(a)
   # print ("attention weight:", a_probs)
   # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    
    return output_attention_mul

def divideTrainTest(dataset, rate = 0.3): 

    print (len(dataset))
  #  train_size = int(len(dataset) * (1-rate))
    test_size = int(len(dataset) * rate)
  
    train, test = dataset[: -test_size], dataset[-test_size:]#
    print (len(train))
    print (len(test))       
    return train, test

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):     
        a = dataset[i:(i+look_back),1:]  
        dataX.append(a)
    #    X.append(data[i:i + size, :])
        dataY.append(dataset[i + look_back, 0]) 
    
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y

#多维归一化  返回数据和最大最小值
def NormalizeMult(data):
    #normalize 用于反归一化
    data = np.array(data, dtype='float64')
    normalize = np.arange(2*data.shape[1],dtype='float64')

    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    #np.save("./normalize.npy",normalize)
    return  data, normalize

#多维反归一化
def FNormalizeMult(data,normalize):
    data = np.array(data)
    
    print ("ata22",data.shape)
    for i in range(0,(data.shape[1])):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow

    return data




from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calcRMSE(true,pred):
    return np.sqrt(mean_squared_error(true, pred))


# 计算MAE
def calcMAE(true,pred):
    return mean_absolute_error(true, pred)


# 计算MAPE
def calcMAPE(true, pred, epsion = 0.0000000):

    true += epsion
    return np.sum(np.abs((true-pred)/true))/len(true)*100

def calcR2(true, pred):
    return r2_score (true, pred)


# 计算SMAPE
def calcSMAPE(true, pred):
    delim = (np.abs(true)+np.abs(pred))/2.0
    return np.sum(np.abs((true-pred)/delim))/len(true)*100

def mul_cnn(x):
    ## 默认8
    x3 = Conv1D(8, 1, padding = 'same', activation = 'relu')(x)  #, padding = 'same'填充，不填充为valid
    x3 = Dropout(0.2)(x3)
   
    x1 = Conv1D(8, 3, padding = 'same', activation = 'relu')(x)
    x1 = Dropout(0.2)(x1)
   
    x2 = Conv1D(8, 5, padding = 'same', activation = 'relu')(x)
    x2 = Dropout(0.2)(x2)

    x4 = Conv1D(8, 7, padding = 'same', activation = 'relu')(x)
    x4 = Dropout(0.2)(x4)
    y = Add()([x3, x1, x2, x4])
 
    return y
    
    

def model(INPUT_DIMS, lookBack, filters, kernel_size, hidden_dim, lr):
    
    inputs = Input(shape=(lookBack, INPUT_DIMS))
      
    inputs1 = mul_cnn(inputs)
     
    x_shortcut = inputs1
    x= mul_cnn(inputs1)
      
    x= mul_cnn(x)

    res= Add()([x_shortcut, x])
    

    res = Dense(128, activation = 'relu')(res)
    
    
    x5 = Bidirectional(GRU(hidden_dim, return_sequences=True, name='bigru'))(inputs)
    x5 = Dropout(0.2)(x5)
    x5 = attention_3d_block2(x5)
    
    
    x5 = Bidirectional(GRU(hidden_dim, return_sequences=True, name='bigru1'))(x5)
    x5 = Dropout(0.2)(x5)
    x5 = attention_3d_block2(x5)
    
#    print("x5 is", x5.shape)
    
    concat_output = Add()([res, x5]) 
  
    print("concat_output is", concat_output.shape)
    
    

    
#RUl prediction
    
    fc = Dense(128, activation = 'relu')(concat_output) 
    fc = Flatten()(fc) 
    output = Dense(1)(fc) #
  
       
    model = Model(inputs=[inputs], outputs=output)
          
            
    return model
    
                       


def attention_model(data, INPUT_DIMS, lookBack, epochs, batchSize, filters, kernel_size, hidden_dim, lr):
    
    

    train, test = divideTrainTest(data)
    train, normalize = NormalizeMult(train)
    test, normalize1 = NormalizeMult(test)
    
    
    train_X, train_Y = create_dataset(train, TIME_STEPS)
    test_X, test_Y = create_dataset(test, TIME_STEPS)
   
 
    print("trainX shape is", train_X.shape)
    print("trainY shape is", train_Y.shape)
    print("testX shape is", test_X.shape)
    print("testY shape is", test_Y.shape)
    
    print(len(train_X))
    print(train_X.shape,train_Y.shape)

    print(len(test_X))
    print(test_X.shape,test_Y.shape)
     
    
    m = model(INPUT_DIMS, lookBack, filters, kernel_size, hidden_dim, lr)
    m.summary()
    opt = optimizers.Adam(lr=lr)
    m.compile(optimizer= opt, loss='mean_squared_error', metrics=["mean_absolute_percentage_error"])
    
    m.fit([train_X], train_Y, epochs= epochs, batch_size=batchSize)
#     

 
    trainPred = m.predict(train_X)
    testPred = m.predict(test_X)
    trainPred = trainPred.reshape(-1, 1)
    testPred = testPred.reshape(-1, 1)
    
    
    
    trainPred = FNormalizeMult(trainPred, normalize)
    train_Y = train_Y.reshape(-1, 1)
    train_Y = FNormalizeMult(train_Y, normalize)
    
    testPred = FNormalizeMult(testPred, normalize1)
    test_Y = test_Y.reshape(-1, 1)
    test_Y = FNormalizeMult(test_Y, normalize1)
    
    MAE = calcMAE(test_Y, testPred)
    print(MAE)
    RMSE = calcRMSE(test_Y, testPred)
    print(RMSE)
    r2_score = calcR2(test_Y, testPred)
    print(r2_score)
          
    return trainPred, testPred, MAE, RMSE, r2_score

def load_data(filename, columnName):

    df = pd.read_csv(filename)
    df = df.fillna(0)
    ts = df[columnName]
    data = ts.values.reshape(-1, 1).astype("float32")  # (N, 1)
    print("time series shape:", data.shape)
    return data


if __name__ == "__main__":
   #加载数据
    data = pd.read_csv("./OpenStack/RUL.csv") 
     
    print(data.columns)
    print(data.shape)

    INPUTDIMS = 10 #
    
    TIME_STEPS = 20 #固定20
    hidden_dim = 64 #固定 64
     Epochs = 100 #固定 100
    
    
    
    Batch_size = 32 #默认
    Filters = 8    #默认 8
    Kernel_size = 3 #默认1，3,5
    lr = 1e-3  #默认
   

    trainPred, testPred, mae, rmse, r2 = attention_model(data, INPUT_DIMS = INPUTDIMS,
                                                                  lookBack=TIME_STEPS, epochs = Epochs, 
                                                                  batchSize=Batch_size, filters = Filters, 
                                                                  kernel_size = Kernel_size, hidden_dim=hidden_dim, lr=lr)




