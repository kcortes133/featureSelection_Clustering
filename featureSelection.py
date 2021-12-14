from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest

def recFeatElim(X, y):
    estimator = SVR(kernel='linear')
    selector = RFE(estimator)
    selector = selector.fit(X,y)
    return selector.ranking_

def kBest(X, y):
    x_new = SelectKBest(k=10).fit(X,y)
    cols = x_new.get_support(indices=True)
    return cols
