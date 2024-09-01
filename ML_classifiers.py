# Avoiding warning
import warnings

def warn(*args, **kwargs): pass
warnings.warn = warn
import numpy
# _______________________________
seed =456
numpy.random.seed(seed)

# Essential Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# _____________________________
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, \
    RandomForestClassifier,  \
    AdaBoostClassifier,    \
    GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, \
    confusion_matrix, \
    roc_auc_score, \
    average_precision_score, \
    roc_curve, \
    f1_score, \
    recall_score, matthews_corrcoef, auc,cohen_kappa_score


X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values

'''
### Remove columns there is all zero values.
v = []
for i in range(X.shape[1]):
    if not np.all(X[:, i] == 0):
        v.append(i)

X = X[:, v]

'''
from sklearn.utils import shuffle
X, y = shuffle(X, y)  # Avoiding bias


# Step 06 : Scaling the feature
# ______________________________________________________________________________
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)

# ______________________________________________________________________________
# Step 04 : Encoding y :
# ______________________________________________________________________________
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
# ______________________________________________________________________________


# scikit-learn :
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier,  AdaBoostClassifier,    GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


Classifiers = [
    #SVC(probability=True),

    AdaBoostClassifier(),
    ExtraTreesClassifier(n_estimators=10, min_samples_split=150),
    XGBClassifier(n_estimators=50),
    lgb.LGBMClassifier(n_estimators=10),


    
]


def runClassifiers():
    i=0
    Results = []  # compare algorithms
    #cv = StratifiedKFold(n_splits=8, shuffle=True)
    from sklearn.model_selection import StratifiedKFold,KFold
    cv = KFold(n_splits=10, random_state=None, shuffle=True)
    for classifier, name in zip(Classifiers, Names):
        accuray = []
        auROC= []
        avePrecision = []
        F1_Score = []
        AUC = []
        MCC = []
        Recall = []
        mean_TPR = 0.0
        mean_FPR = np.linspace(0, 1, 100)
        CM = np.array([
            [0, 0],
            [0, 0],
        ], dtype=int)
        print(classifier.__class__.__name__)
        model = classifier
        for (train_index, test_index) in cv.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            model.fit(X_train, y_train)
            # Calculate ROC Curve and Area the Curve
            y_proba = model.predict_proba(X_test)[:, 1]
           

