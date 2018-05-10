import lstm as lstm 
import time

if __name__ == "__main__":
  begin_time = time.time()
  lstm.TrainLSTM()
  end_time = time.time()
  print("the training precess is finished")
  print("the time consumed is: "+str(end_time-begin_time)+'s')
