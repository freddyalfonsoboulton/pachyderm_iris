import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import sys
import os
import shutil

script_name = sys.argv[0]
in_dir = sys.argv[1]
out_dir = sys.argv[2]

feature_names = ['sepal_length','sepal_width','petal_length','petal_width']
target_name = ['species']

train = pd.read_csv(os.path.join(in_dir,'train.csv'))
test = pd.read_csv(os.path.join(in_dir,'test.csv'))

X_train = train[feature_names].values
X_test = test[feature_names].values

labeler = LabelEncoder().fit(train[target_name].values.ravel())
y_train = labeler.transform(train[target_name].values.ravel())
y_test = labeler.transform(test[target_name].values.ravel())

svc_model = SVC(C = 1.0,kernel='rbf',gamma = 'auto')
svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)

pd.DataFrame({'predictions':predictions,'truth':y_test}).to_csv(os.path.join(out_dir,'results.csv'),index = False)

#shutil.copyfile(script_name,os.path.join(out_dir,'model_script.py'))

