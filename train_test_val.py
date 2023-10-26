import pandas as pd 
import os 
import numpy as np
from config import *

p_train = 0.7
p_test = 0.15
p_val = 0.15


datas = pd.DataFrame(os.listdir(path_images))

datas = datas.sample(frac=1).reset_index(drop=True)

train = datas.head(int(len(datas)*(p_train)))
rest = datas.tail(int(len(datas)*(1 - p_train))+1)
test = rest.head(int(len(rest)*(p_test/(p_test +p_val))))
val = rest.tail(int(len(rest)*(p_val/(p_test +p_val))) +1)

train.columns = ["filename"]
test.columns = ["filename"]
val.columns = ["filename"]

train.to_csv('train_df.csv')
test.to_csv('test_df.csv')
val.to_csv('val_df.csv')