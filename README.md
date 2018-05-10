# LSTM-for-Trading-Timing

It's a model using RNN/LSTM to predict the time series. The core idea of this project is the state of ivx of ETF50 could be classified to two parts. Hence, we give each sample a label -1 or 1 and -1 means ivx would decrease while 1 means ivx will increase. 

During the process, firstly we regularization the data including the train set and test set to fast the convergence. Then we define the parameters of BasicLSTMCell provided by tensorflow and build the framework. Thirdly, we deliver the train set into the model to make the model perform better. At last, we using the test set to calculate the accuracy of the LSTM model and modify the parameters.

We can see training result as following: 
<br/>
![image](https://github.com/richardwang013/LSTM-for-Time-Series/raw/master/ImageStore/result.PNG)
<br/>
There are two hidden LSTM layers in the model and each layer have 10 cells. And considering to avoid gradient vanishing, we use the leaky_RELU function as activate function and tanh as the final activate function because we want to get a binary classifier. After 6000 iterations, the loss nearly converge to 0.
<br/>
The overall accuracy we get is 0.58 and the accuracy of predicion of 1 is 0.43 as the accuracy of prediction of -1 is 0.73
