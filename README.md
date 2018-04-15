# LSTM-for-Time-Series

It's a model using RNN/LSTM to predict the time series. The core idea of this project is the state of ivx of ETF50 could be classified to two parts. Hence, we give each sample a label -1 or 1 and -1 means ivx would decrease while 1 means ivx will increase. 

During the process, firstly we regularization the data including the train set and test set to fast the convergence. Then we define the parameters of BasicLSTMCell provided by tensorflow and build the framework. Thirdly, we deliver the train set into the model to make the model perform better. At last, we using the test set to calculate the accuracy of the LSTM model and modify the parameters.

We can see training result as following: 
<br/>
![image](https://github.com/richardwang013/LSTM-for-Time-Series/raw/master/ImageStore/result.png)
<br/>
The loss function is under 1 and it couldn't converge to 0. But it's a classifier and the last activate function I use in the LSTM model is tanh so the final result of the prediction is between -1 and 1 while the actual label is 1 or -1 so the maximun average square loss is 2 so the flucturate is just a common sense in this situation when it is always lower than 1.
<br/>
The overall accuracy we get is 0.61 and the accuracy of predicion of 1 is 0.51 as the accuracy of prediction of -1 is 0.72
