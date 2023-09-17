import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv("source.csv")
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1), data["target"], test_size=0.1, random_state=42)
X_train.join(y_train).to_csv('train.csv')
X_test.join(y_test).to_csv('debug.csv')