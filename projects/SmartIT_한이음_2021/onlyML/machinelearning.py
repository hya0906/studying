#패키지
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image
import pandas as pd
import numpy as np
import pydotplus
import os
from tensorflow.keras.models import load_model
from graphviz import Source
import pickle
import joblib

#데이터 로드
a_data = pd.read_csv("./data_2.csv") #chane address
print(a_data)

#데이터 전처리
#    O,X    loud,quit  low,nor,high low,nor,high
#PIR(0,1),마이크(0,1),온도(0,1,2),습도(0,1,2) 순
a_data.result = a_data.result.replace(0,'0000')
a_data.result = a_data.result.replace(1,'0001')
a_data.result = a_data.result.replace(2,'0002')
a_data.result = a_data.result.replace(10, '0010')
a_data.result = a_data.result.replace(11, '0011')
a_data.result = a_data.result.replace(12, '0012')
a_data.result = a_data.result.replace(20, '0020')
a_data.result = a_data.result.replace(21, '0021')
a_data.result = a_data.result.replace(22, '0022')
a_data.result = a_data.result.replace(100, '0100')
a_data.result = a_data.result.replace(101, '0101')
a_data.result = a_data.result.replace(102,'0102')
a_data.result = a_data.result.replace(110, '0110')
a_data.result = a_data.result.replace(111, '0111')
a_data.result = a_data.result.replace(112, '0112')
a_data.result = a_data.result.replace(120, '0120')
a_data.result = a_data.result.replace(121, '0121')
a_data.result = a_data.result.replace(122, '0122')

print(a_data)

#속성과 클래스 분리
X = np.array(pd.DataFrame(a_data, columns=['temp','humid','decibel','PIR']))
y = np.array(pd.DataFrame(a_data, columns=['result'])).tolist()

X_train, X_test, y_train, y_test = train_test_split(X,y)
print('X_train',X_train)
print('X_test',X_test)

#데이터 학습
dt_clf = DecisionTreeClassifier(criterion='entropy')#다중분류
dt_clf = dt_clf.fit(X_train, y_train)
dt_prediction = dt_clf.predict(X_test)
print('ytest: ',y_test,'예측:' ,dt_prediction)
print(dt_clf.score(X_test, y_test))
#accuracy = RMSE(X_test, y_test)
#print(f"정확도: {accuracy}")

#의사결정트리 그래프 표현
feature_names = a_data.columns.tolist()
feature_names = feature_names[1:-1]
target_name = np.array(["0000","0001","0002","0010","0011","0012","0020","0021","0022","0100","0101","0102","0110","0111","0112","0120","0121","0122","1000","1001","1002","1010","1011","1012","1020","1021","1022","1100","1101","1102","1110","1111","1112","1120","1121","1122"])
dt_dot_data = tree.export_graphviz(dt_clf, out_file = None,
                                  feature_names = feature_names,
                                  class_names = target_name,
                                  #filled = True, rounded = True,
                                  #special_characters = False
                                  )
dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)
Image(dt_graph.create_png())

#모델저장
joblib.dump(dt_clf, './demo2_model.pkl') #모델저장
loaded_model = joblib.load('./demo2_model.pkl')

# 모델의 정확도를 계산하고 출력합니다.
score = loaded_model.score(X, y)
print('정확도: {score:.3f}'.format(score=score))
