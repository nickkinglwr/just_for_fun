import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier



# Function to test hard voting
def hVoting(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier(min_samples_leaf=1000, min_samples_split=1000)
    dt2 = DecisionTreeClassifier(min_samples_leaf=1000, min_samples_split=1000)

    knn = KNeighborsClassifier(n_neighbors=1000)
    #knn2 = KNeighborsClassifier(n_neighbors=1000)

    #nn = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(100,100))
    #lr = LogisticRegression()

    preds = [[],[]]

    X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.5) # split training data to so each DT has different results
    #X3, X4, y3, y4 = train_test_split(X_train, y_train, test_size=0.5)

    dt2.fit(X1,y1)
    #lr.fit(X_train,y_train)
    dt.fit(X2,y2)
    #nn.fit(X_train,y_train)
    #nn.fit(X_train, y_train)
    #lr.fit(X_train, y_train)
    preds[0] = dt.predict(X_test)
    preds[1] = dt2.predict(X_test)
   # preds[2] = knn.predict(X_test)
    #preds[3] = nn.predict(X_test)

    avAcc = avPred = avF1 = 0
    for i in range(2): # get average performance of each inner-model
        avAcc += metrics.accuracy_score(y_true=y_test, y_pred=preds[i])
        avPred += metrics.precision_score(y_true=y_test, y_pred=preds[i], pos_label='normal')
        avF1 += metrics.f1_score(y_true=y_test, y_pred=preds[i], pos_label='normal')

    print(avAcc / 2)
    print(avPred / 2)
    print(avF1 / 2)

    normTally = 0
    adNormTally = 0
    realPred = []

    #Implemnt hard voting
    for col in range(len(preds[1])):
        for row in range(len(preds)):
            if preds[row][col] == "normal":
                 normTally += 1
            else:
                adNormTally += 1
        if normTally > adNormTally:
            realPred.append("normal")
        else:
            realPred.append("abnormal")
        normTally = adNormTally = 0

    print("Acc: {0}".format(metrics.accuracy_score(y_true=y_test, y_pred=realPred)))
    print("Prec: {0}".format(
        metrics.precision_score(y_true=y_test, y_pred=realPred, average='binary', pos_label='normal')))
    print("F1: {0}".format(metrics.f1_score(y_true=y_test, y_pred=realPred, pos_label='normal')))



# Function to test soft voting
def sVoting(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier(min_samples_leaf=1000, min_samples_split=1000)

    knn = KNeighborsClassifier(n_neighbors=1000)

    #nn = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(100,100))
    #lr = LogisticRegression()

    preds = [[], []]

    #1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.5)
    #X3, X4, y3, y4 = train_test_split(X_train, y_train, test_size=0.5)

    knn.fit(X_train, y_train)
   # nn.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    #lr.fit(X_train, y_train)
    # nn.fit(X_train, y_train)
    # lr.fit(X_train, y_train)
    preds[0] = dt.predict_proba(X_test)
    #preds[1] = nn.predict_proba(X_test)
    preds[1] = knn.predict_proba(X_test)
    #preds[3] = lr.predict_proba(X_test)

    avAcc = metrics.accuracy_score(y_true=y_test, y_pred=dt.predict(X_test)) + metrics.accuracy_score(y_true=y_test, y_pred=knn.predict(X_test))
    avPred = metrics.precision_score(y_true=y_test, y_pred=dt.predict(X_test), pos_label='normal')  + metrics.precision_score(y_true=y_test, y_pred=knn.predict(X_test), pos_label='normal')
    avF1 = metrics.f1_score(y_true=y_test, y_pred=dt.predict(X_test), pos_label='normal') + metrics.f1_score(y_true=y_test, y_pred=knn.predict(X_test), pos_label='normal')
    print(avAcc / 2)
    print(avPred / 2)
    print(avF1 / 2)

    normTally = 0
    adNormTally = 0
    realPred = []
    for col in range(len(preds[1])):
        for row in range(len(preds)):
            normTally += preds[row][col][1]
            adNormTally += preds[row][col][0]
        if normTally > adNormTally:
            realPred.append("normal")
        else:
            realPred.append("abnormal")
        normTally = adNormTally = 0

    print("Acc: {0}".format(metrics.accuracy_score(y_true=y_test, y_pred=realPred)))
    print("Prec: {0}".format(metrics.precision_score(y_true=y_test, y_pred=realPred, average='binary', pos_label='normal')))
    print("F1: {0}".format(metrics.f1_score(y_true=y_test, y_pred=realPred, pos_label='normal')))



# Function to test decision tree bagging
def dtBagging(X_train, y_train, X_test, y_test, num=10):
    preds = [[] for i in range(num)]
    dts = [DecisionTreeClassifier(min_samples_leaf=1000, min_samples_split=1000) for i in range(num)]

    avAcc = 0
    avPred = 0
    avF1 = 0

    for i in range(num):
        X_res, y_res = resample(X_train,y_train) # Get bootstrap samples
        dt = dts[i]
        dt.fit(X_res,y_res)
        preds[i] = dt.predict(X_test)
        avAcc += metrics.accuracy_score(y_true=y_test, y_pred=preds[i])
        avPred += metrics.precision_score(y_true=y_test, y_pred=preds[i], pos_label='normal')
        avF1 += metrics.f1_score(y_true=y_test, y_pred=preds[i], pos_label='normal')


    print(avAcc / num)
    print(avPred / num)
    print(avF1 / num)

    normTally = 0
    adNormTally = 0
    realPred = []

    # Aggregate results
    for col in range(len(X_test)):
        for row in range(len(preds)):
            if preds[row][col] == "normal":
                normTally += 1
            else:
                adNormTally += 1

        if normTally > adNormTally:
            realPred.append("normal")
        else:
            realPred.append("abnormal")
        normTally = adNormTally = 0

    print("Acc: {0}".format(metrics.accuracy_score(y_true=y_test, y_pred=realPred)))
    print("Prec: {0}".format(metrics.precision_score(y_true=y_test, y_pred=realPred, average='binary', pos_label='normal')))
    print("F1: {0}".format(metrics.f1_score(y_true=y_test, y_pred=realPred, pos_label='normal')))



# Function to test random forest classifier
def randForest(X_train, y_train, X_test, y_test, num=100):
    dts = []
    preds = []

    for i in range(num):
        curr = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1000, min_samples_split=1000)
        dts.append(curr)
        X_res, y_res = resample(X_train, y_train, n_samples=100000) # Get bootstrap sample
        drop_indices = np.random.choice(X_res.columns, 7, replace=False) # Drop subset of features
        X_res.drop(drop_indices, axis=1)
        curr.fit(X_res, y_res)
        preds.append(curr.predict(X_test))

    normTally = 0
    adNormTally = 0
    realPred = []
    for col in range(len(X_test)):
        for row in range(len(preds)):
            if preds[row][col] == "normal":
                normTally += 1
            else:
                adNormTally += 1

        if normTally > adNormTally:
            realPred.append("normal")
        else:
            realPred.append("abnormal")
        normTally = 0
        adNormTally = 0

    print("Acc: {0}".format(metrics.accuracy_score(y_true=y_test, y_pred=realPred)))
    print("Prec: {0}".format(metrics.precision_score(y_true=y_test, y_pred=realPred, average='binary', pos_label='normal')))
    print("F1: {0}".format(metrics.f1_score(y_true=y_test, y_pred=realPred, pos_label='normal')))



# Function to test out-of-box AdaBoost classifier
def adaBoost(X_train, y_train, X_test, y_test, num=100):
    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3,min_samples_leaf=1000, min_samples_split=1000), n_estimators=num, learning_rate=0.0001)
    ada.fit(X_train, y_train)

    realPred = ada.predict(X_test)

    print("Acc: {0}".format(metrics.accuracy_score(y_true=y_test, y_pred=realPred)))
    print("Prec: {0}".format(metrics.precision_score(y_true=y_test, y_pred=realPred, average='binary', pos_label='normal')))
    print("F1: {0}".format(metrics.f1_score(y_true=y_test, y_pred=realPred, pos_label='normal')))


# Get training dataset
df = pd.read_csv("KDDTrain+Bin.csv")

y = df['Column42']
X = df.drop('Column42', axis=1)
X = pd.get_dummies(X) # Apply one-hot encoding to training samples

X_fake, X_train, y_fake, y_train = train_test_split(X, y, test_size=0.5) # Get 50% subset of training data to make computation easier

# Keep only most important features found in feature selection process
X_train = X_train[['Column5','Column6','Column4_SF','Column29','Column3_ecr_i','Column30','Column2_icmp','Column3_http','Column33']]#,'Column34','Column23','Column36','Column35','Column10','private','Column37','Column24','Column12','ftp_data']]

df = pd.read_csv("KDDTest+Bin.csv")

y = df['Column42']
X = df.drop(['Column43','Column42'], axis=1)
X = pd.get_dummies(X) # Apply one-hot encoding to testing samples

X_fake, X_test, y_fake, y_test = train_test_split(X, y, test_size=0.5) # Get 50% subset of testing data to make computation easier

# Keep only most important features found in feature selection process
X_test = X_test[['Column5','Column6','Column4_SF','Column29','Column3_ecr_i','Column30','Column2_icmp','Column3_http','Column33']]#,'Column34','Column23','Column36','Column35','Column10','private','Column37','Column24','Column12','ftp_data']]

# Elminate features that appear in testing or training set that do not appear in other. (Occurs during one-hot encoding)
for i in (set(X_train.columns) ^ set(X_test.columns)):
    if i in X_train.columns:
        X_train = X_train.drop(i, axis=1)
    else:
        X_test = X_test.drop(i, axis=1)

#Run test on the training and testing sets
adaBoost(X_train, y_train, X_test, y_test)
