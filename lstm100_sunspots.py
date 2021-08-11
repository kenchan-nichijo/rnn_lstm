import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from keras.models import Sequential
from keras.layers import Dense,  LSTM
from keras import metrics
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('sunspots.csv', encoding="shift_jis")
L = len(df)
Y = df.iloc[:, 1]  # 最終列のみ抽出
Y = np.array(Y)  # numpy配列に変換
Y = Y.reshape(-1, 1)  # 行列に変換（配列の要素数行×1列）

X = Y[0:L-100, :]  

for i in range(99):
    X = np.concatenate([X,Y[i+1:L-99+i]], axis=1)  # numpy配列を結合

Y = Y[100:L, :]  # 予測対象時刻のデータ

scaler = MinMaxScaler()  # データを0～1の範囲にスケーリングするための関数。
scaler.fit(X)  # スケーリングに使用する最小／最大値を計算
X = scaler.transform(X)  # Xを0～1の範囲にスケーリング

scaler1 = MinMaxScaler()  
scaler1.fit(Y)  
Y = scaler1.transform(Y)  

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))  # 3次元配列に変換

# train, testデータを定義
X_train = X[:24, :, :]
X_test = X[24:, :, :]
Y_train = Y[:24, :]
Y_test = Y[24:, :]

model = Sequential()
model.add(LSTM(50, activation = 'tanh', input_shape = (1,100), recurrent_activation= 'hard_sigmoid'))
model.add(Dense(1))

sgd = optimizers.SGD(lr=0.05) #Stochastic Gradient Descent:確率的勾配降下法

model.compile(loss= 'mean_squared_error', 
              optimizer = sgd
              )

hist = model.fit(X_train, Y_train, 
                 epochs=300,
                 validation_data=(X_test, Y_test),
                 verbose=2,
                 )

Predict = model.predict(X_test)


# オリジナルのスケールに戻す
Y_train = scaler1.inverse_transform(Y_train)

Y_test = scaler1.inverse_transform(Y_test)

Predict = scaler1.inverse_transform(Predict)


t= df.iloc[124:, 0]
Y= scaler1.inverse_transform(Y)

plt.figure(figsize=(15,10))
#plt.plot(df.iloc[:, 0],df.iloc[:, 1],linestyle='--', linewidth=0.5, color='gray',label = 'training')
plt.plot(t,Y_test,  label = 'Test')
plt.plot(t,Predict,  label = 'Prediction')
plt.xlabel("Year")
plt.ylabel("Sunspot Number")
plt.xlim(1875,2020)
plt.xticks([1875, 1900, 1925, 1950, 1975, 2000],
           ["1875", "1900", "1925", "1950","1975","2000"])
plt.ylim(0, 300)
plt.legend(loc='best') #凡例
plt.grid()
plt.show()

#lossを可視化
loss     = hist.history['loss']
val_loss = hist.history['val_loss']

nb_epoch = len(loss)

#plt.plot(range(nb_epoch), loss,  label='loss')
plt.plot(range(nb_epoch), val_loss, label='val_loss(MSE)')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xlim(0,300)
plt.ylim(0,0.10)
plt.show()