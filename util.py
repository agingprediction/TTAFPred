import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import statsmodels.stats.diagnostic
import statsmodels.api as sm


# load data, return pandas format ts and array ts, ts is formatted as a column
def load_data(filename, columnName):

    df = pd.read_csv(filename)
    df = df.fillna(0)
    ts = df[columnName]
    data = ts.values.reshape(-1, 1).astype("float32")  # (N, 1)
    print("time series shape:", data.shape)
    return ts, data


# divide ts as training/testing samples, looBack is lag window
# NOTE: we can generate the samples as RNN format
def createSamples(dataset, lookBack, RNN=True):

    dataX, dataY = [], []
    for i in range(len(dataset) - lookBack):
        sample_X = dataset[i:(i + lookBack), :]
        sample_Y = dataset[i + lookBack, :]
        dataX.append(sample_X)
        dataY.append(sample_Y)
    dataX = np.array(dataX)  # (N, lag, 1)
    dataY = np.array(dataY)  # (N, 1)
    if not RNN:
        dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1]))

    return dataX, dataY


# divide training and testing, default as 7:3
def divideTrainTest(dataset, rate=0.7):

    train_size = int(len(dataset) * rate+1)
    #print ("t", train_size)
    #test_size = len(dataset) - len(train_size)
    train, test = dataset[:train_size], dataset[train_size:]
  #  train, test = dataset[202:train_size], dataset[train_size+86:]
    
    return train, test


# 取不同长度的子序列，以重点为基准点向前搜索出lag区间段的所有子序列
def createVariableDataset(dataset, minLen, maxLen, step):

    dataNum = len(dataset)
    X, Y = [], []

    for i in range(maxLen, len(dataset)):
        for lookBack in range(minLen, maxLen + 1, step):  # 遍历所有长度
            sample_X = dataset[i-lookBack:i]
            sample_Y = dataset[i]
            X.append(sample_X)
            Y.append(sample_Y)
    X = np.array(X)
    Y = np.array(Y)
    X = pad_sequences(X, maxlen=maxLen, dtype='float32')  # 左端补齐
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y


# 将测试用的ground-truth转化为标准形式，与上面的函数一起使用
def transform_groundTruth(vtestY, minLen, maxLen, step):

    lagNum = (maxLen - minLen) // step + 1
    print("lag num is", lagNum)
    truth = []
    for i in range(0, len(vtestY), lagNum):
        truth.append(np.mean(vtestY[i:i + lagNum]))
    return np.array(truth)


# 取不同长度样本并补足成最大长度
def createPaddedDataset(dataset, lookBack, maxLen):

    dataX, dataY = [], []
    for i in range(len(dataset) - lookBack):
        a = dataset[i:(i + lookBack)]
        dataX.append(a)
        dataY.append(dataset[i + lookBack])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    dataX = pad_sequences(dataX, maxlen=maxLen, dtype='float32')  # 左端补齐
    dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
    return dataX, dataY


# 数据对齐
def align(trTrain,trTest,trendWin,resTrain,resTest,resWin):

    empWin = np.empty((trendWin, 1))
    empWin[:] = np.nan

    empWin2 = np.empty((resWin, 1))
    empWin2[:] = np.nan

    # empWinMax = np.empty((varMaxLen))
    # empWinMax[:] = np.nan

    trendPred = np.vstack((empWin, trTrain))
    trendPred = np.vstack((trendPred, empWin))
    trendPred = np.vstack((trendPred, trTest))

    resPred = np.vstack((empWin2, resTrain))
    resPred = np.vstack((resPred, empWin2))
    resPred = np.vstack((resPred, resTest))

    return trendPred, resPred

def align1(trTrain, trTest, trendWin, resTrain, resTest, resWin, resTrain1, resTest1, resWin1):

    empWin = np.empty((trendWin, 1))
    empWin[:] = np.nan

    empWin2 = np.empty((resWin, 1))
    empWin2[:] = np.nan

    # empWinMax = np.empty((varMaxLen))
    # empWinMax[:] = np.nan

    trendPred = np.vstack((empWin, trTrain))
    trendPred = np.vstack((trendPred, empWin))
    trendPred = np.vstack((trendPred, trTest))

    resPred = np.vstack((empWin2, resTrain))
    resPred = np.vstack((resPred, empWin2))
    resPred = np.vstack((resPred, resTest))
    
    resPred1 = np.vstack((empWin2, resTrain1))
    resPred1 = np.vstack((resPred1, empWin2))
    resPred1 = np.vstack((resPred, resTest1))

    return trendPred, resPred


def plot(trainPred, trainY, testPred, testY):
    pred = np.concatenate((trainPred, testPred))
    gtruth = np.concatenate((trainY, testY))
    plt.plot(pred, 'g')
    plt.plot(gtruth, 'r')
    plt.show()


def LBtest(data):
    # lb,p = statsmodels.stats.diagnostic.acorr_ljungbox(residual)
    # print p
    r, q, p = sm.tsa.acf(data, qstat=True)
    data1 = np.c_[range(1, 41), r[1:], q, p]
    table = pd.DataFrame(data1, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))


if __name__ == "__main__":

    ts, data = load_data("./data/IOTdata2.csv", columnName="Mem")
    plt.plot(data)
    plt.show()

    train, test = divideTrainTest(data, 0.75)
    print("train num:", train.shape)
    print("test num:", test.shape)

    trainX, trainY = createSamples(train, 20, RNN=True)
    testX, testY = createSamples(test, 20, RNN=True)
    print("trainX shape is", trainX.shape)
    print("trainY shape is", trainY.shape)
    print("testX shape is", testX.shape)
    print("testY shape is", testY.shape)

    vtrainX, vtrainY = createVariableDataset(train, 10, 20,step=5)
    vtestX, vtestY = createVariableDataset(test, 10, 20, step=5)
    vtestY = transform_groundTruth(vtestY, 10, 20, 5)

    print(testX.shape)
    print(testY.shape)
    print(vtrainX.shape)
    print(vtrainY.shape)
    print(vtestX.shape)
    print(vtestY.shape)







