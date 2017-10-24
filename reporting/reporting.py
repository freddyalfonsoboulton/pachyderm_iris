import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import OneHotEncoder
import os
import sys

out_dir = sys.argv[2]
in_dir = sys.argv[1]

results = pd.read_csv(os.path.join(in_dir,'results.csv'))

preds = OneHotEncoder(sparse = False).fit_transform(results['predictions'].values.reshape(-1,1))
truth = OneHotEncoder(sparse = False).fit_transform(results['predictions'].values.reshape(-1,1))

mcc_mat = []
for i in range(preds.shape[1]):
    mcc_mat.append(matthews_corrcoef(truth[:,i],preds[:,i]))

plt.bar([0,1,2],mcc_mat,align = 'center')
plt.xticks([0,1,2], ['Setosa','Versicolor','Virginica'])
plt.ylabel('MCC Score')
plt.title("Per-class MCC for Model")
plt.savefig(os.path.join(out_dir,'mcc_plot.png'))
