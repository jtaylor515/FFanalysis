### CODE CELL ###
from vars import *
### END OF CODE CELL ###

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

data = pd.read_csv("datasets/weekly_scoring.csv")
data=pd.DataFrame(data)
train, test = train_test_split(data, test_size=0.2, random_state=3)

train.to_csv("datasets/train.csv")
test.to_csv("datasets/test.csv")