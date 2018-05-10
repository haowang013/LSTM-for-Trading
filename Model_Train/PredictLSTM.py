import lstm as lstm

if __name__ == "__main__":
  y,yhat = lstm.LSTMPredict()
  accuracy1,acc2,acc3 = lstm.PreAccuracy(y,yhat)
  print('the accuracy of the lstm model is: '+str(accuracy1)+'%')
  print('the +1 accuracy is: '+str(acc2)+'%')
  print('the -1 accuracy is: '+str(acc3)+'%')
