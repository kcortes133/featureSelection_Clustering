from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def kNeigh(X, y):
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(X, y)
    return neigh

def rFC(X,y):
    clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X,y)
    return clf


def logRegress(X, y):
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf
