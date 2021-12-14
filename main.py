import numpy as np
import classification, featureSelection, predictions
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


wdbc = np.genfromtxt('wdbc.data', delimiter=',', dtype='str')
y = wdbc[:,1]
y = (y =='M').astype(int)
X = wdbc[:,2:].astype(float)

# get the columns that have 1  as the ranking
cols1 = featureSelection.recFeatElim(X, y)
cols1 = [index for index, element in enumerate(cols1) if element==1]
# use what is returned
cols2 = featureSelection.kBest(X, y)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)

X_train1 = X_train[:,cols1]
X_train2 = X_train[:,cols2]

X_test1 = X_test[:,cols1]
X_test2 = X_test[:,cols2]

kNmodel = classification.kNeigh(X_train2, y_train)
#logRmodel = classification.logRegress(X_train2, y_train)
clf = classification.rFC(X_train1, y_train)

#predictions.predict(X_train1, y_train, kNmodel)
#predictions.predict(X_test1, y_test, kNmodel)

#predictions.predict(X_train2, y_train, logRmodel)
#predictions.predict(X_test2, y_test, clf)



#metrics.plot_roc_curve(kNmodel, X_test2, y_test)
#plt.show()
metrics.plot_roc_curve(clf, X_test1, y_test)
plt.show()
