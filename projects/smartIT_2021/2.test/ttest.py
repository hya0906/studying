import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

features=np.load('f.npy')#영상
labels=np.load('l.npy')
Xmusic=np.load('Xmusic.npy')#뮤지컬
ymusic=np.load('ymusic.npy')
X=np.load('X.npy')#기본데이터
y=np.load('y.npy')

X =  np.load(file="X.npy")
y =  np.load(file="Y.npy")


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=60)
print(X)
dt = pd.DataFrame(X)
dt_y = pd.DataFrame(y)
result2 = pd.concat([dt,dt_y], ignore_index=False)
print(result2)
print(dt.head())
plot_data(X_train, y_train, X_test, y_test)
X.scatter()
plt.show()