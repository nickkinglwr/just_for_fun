import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.svm import LinearSVC
from sklearn.metrics import plot_roc_curve, roc_curve, auc

import matplotlib.pyplot as plt
import keras
import time

'''
    Nicholas Lower
  
    Conducts experiment on Logistic Regression, Support Vector Machine, and Neural Network on Network Intrusion Data.
    
    preprocess function - takes in full dataset and separates into data and labels, one-hot encodes categorical features
                          and applies feature selection. Returns new data and labels.
    *_tune function - tunes a model's hyperparameter using train/validation split of training data passed. Returns 
                      hyperparameters that maximize accuracy of validation data.
    *_experiment function - runs the 10-fold cross validation experiment on the passed data and labels using the named
                            model. Outputs Average ROC Curve, average AUC, and average accuracy.
    full_experiment function - runs the 10-fold cross validation experiment on the passed data and labels using all of
                               models for direct comparison. Outputs Average ROC Curves, average AUCs, and average 
                               accuracies. Runs full_latency and full_throughput on trained models automatically.
    full_latency function - runs latency test on all passed models. Outputs average latency for each model predicting
                            ~1 million packets with a bar graph.
    full_throughput function - runs throughput test on all passed models. Outputs average throughput for each model
                               predicting for 1 second with a bar graph.
    
'''


def preprocess(data):
    # Preprocesses the data
    # One-hot encodes categorical features, implements feature selection

    labels = (data['class'] == 'normal').astype('int32')
    data = data.drop('class', axis=1)

    # One-hot encoding
    data = data.drop('protocol_type', axis=1).join(pd.get_dummies(data['protocol_type']))
    data = data.drop('service', axis=1).join(pd.get_dummies(data['service']))
    data = data.drop('flag', axis=1).join(pd.get_dummies(data['flag']))

    # Feature Selection
    #idx = SelectKBest(k=15).fit(data,labels)
    idx =  SelectFromModel(LinearSVC(max_iter=1000)).fit(data,labels)
    data = data[data.columns[idx.get_support()]]

    return (data,labels)



def nn_tune(full_train_data, labels):
# Tunes the Neural Network hyperparameters
# Uses train/validation split of whole passed data set

    learning_rates = [0.5, 0.1, 0.01, 0.001, 0.0001]
    batches = [128, 256, 512, 1024, 4096]
    best = ()
    best_acc = 0.0
    for lr in learning_rates:
        for b in batches:
            nn = keras.models.Sequential(
                [
                    keras.layers.InputLayer((full_train_data.shape[1],)),
                    keras.layers.Dense(50, activation='sigmoid'),
                    keras.layers.Dense(20, activation='sigmoid'),
                    keras.layers.Dense(1, activation='sigmoid')
                ]
            )
            nn.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy',
                       metrics=['accuracy'])

            train, valid, train_labels, valid_labels = train_test_split(full_train_data, labels, test_size=0.33)

            nn.fit(train, train_labels, batch_size=b, epochs=50, verbose=0)
            score, acc = nn.evaluate(valid, valid_labels, verbose=0)

            if acc > best_acc:
                best = (lr, best)
                best_acc = acc
    return best



def svm_tune(full_train_data, labels):
    # Tunes the C parameter for SVM using all of passed training data

    train, valid, train_labels, valid_labels = train_test_split(full_train_data, labels, test_size=0.33)
    best = 0.0
    best_acc = 0.0
    Cs = [1, 0.9, 0.5, 0.25, 0.1, 0.01, 0.001]
    for c in Cs:
        svm = LinearSVC(C=c)
        svm.fit(train, train_labels)

        acc = svm.score(valid, valid_labels)
        if acc > best_acc:
            best = c
            best_acc = acc

    return best



def lr_tune(full_train_data, labels):
    # Tunes the C parameter for LR using all of passed training data

    train, valid, train_labels, valid_labels = train_test_split(full_train_data, labels, test_size=0.33)
    best = 0.0
    best_acc = 0.0
    Cs = [1, 0.9, 0.5, 0.25, 0.1, 0.01, 0.001]
    for c in Cs:
        lr = LogisticRegression(solver = 'lbfgs', C=c)
        lr.fit(train, train_labels)

        acc = lr.score(valid, valid_labels)
        if acc > best_acc:
            best = c
            best_acc = acc

    return best



def nn_experiment(data, labels):
# Experiment for only Neural Network
# Includes learning curve calculation

    nn = keras.models.Sequential(
        [
            keras.layers.InputLayer((data.shape[1],)),
            keras.layers.Dense(50, activation='sigmoid'),
            keras.layers.Dense(20, activation='sigmoid'),
            keras.layers.Dense(1, activation='sigmoid')
        ]
    )

    fold = KFold(10, shuffle=True)
    avg_acc = 0.0

    hist = nn.fit(data, labels, batch_size=256, verbose=0, epochs=100, validation_split=0.33)
    plt.plot(hist.history['loss'], color='blue', label='Training loss')
    plt.plot(hist.history['val_loss'], color='red', label='Validation loss')
    plt.legend()
    plt.show()

    nn.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy',
               metrics=['accuracy'])

    print(nn.summary())

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    for train_idx, test_idx in fold.split(data,labels):
        train, train_labels = data.iloc[train_idx], labels.iloc[train_idx]
        test, test_labels = data.iloc[test_idx], labels.iloc[test_idx]

        nn.fit(train, train_labels, batch_size=128, verbose=0, epochs=50, validation_split=0.33)

        fpr, tpr, thres = roc_curve(test_labels, nn.predict_proba(test))
        score, acc = nn.evaluate(test, test_labels, verbose=0)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr,tpr))

        print(acc)
        avg_acc += acc

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, label='Mean ROC')

    ax.legend()
    plt.show()
    avg_acc /= 10
    print('Average NN Acc: {0}'.format(avg_acc))
    print(aucs)
    print(mean_auc)



def svm_experiment(data,labels):
    # Experiment for only SVM

    fold = KFold(10, shuffle=True)
    avg_acc = 0.0
    svm = LinearSVC(max_iter=10000)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for train_idx, test_idx in fold.split(data,labels):
        train, train_labels = data.iloc[train_idx], labels.iloc[train_idx]
        test, test_labels = data.iloc[test_idx], labels.iloc[test_idx]

        svm.fit(train,train_labels)

        acc = svm.score(test,test_labels)

        viz = plot_roc_curve(svm, test, test_labels)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        print(acc)
        avg_acc += acc

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, label='Mean ROC')

    ax.legend()
    plt.show()
    avg_acc /= 10
    print('Average SVM Acc: {0}'.format(avg_acc))
    print(aucs)
    print(mean_auc)



def lr_experiment(data, labels):
    full = data.copy(deep=True)
    fold = KFold(10, shuffle=True)
    avg_acc = 0.0
    lr = LogisticRegression(solver='lbfgs', max_iter = 1000)

    tprs1 = []
    aucs1 = []
    mean_fpr1 = np.linspace(0, 1, 100)

    print(data.shape)

    fig, ax = plt.subplots()
    for trainIdx, testIdx in fold.split(data, labels):
        train, trainLabels = data.iloc[trainIdx], labels.iloc[trainIdx]
        test, testLabels = data.iloc[testIdx], labels.iloc[testIdx]

        lr.fit(train, trainLabels)
        acc = lr.score(test, testLabels)
        viz = plot_roc_curve(lr,test,testLabels)
        interp_tpr1 = np.interp(mean_fpr1, viz.fpr, viz.tpr)
        interp_tpr1[0] = 0.0
        tprs1.append(interp_tpr1)
        aucs1.append(viz.roc_auc)

        print(acc)
        avg_acc += acc

    mean_tpr1 = np.mean(tprs1, axis=0)
    mean_tpr1[-1] = 1.0
    mean_auc1 = auc(mean_fpr1, mean_tpr1)
    ax.plot(mean_fpr1, mean_tpr1, label='No Selection Mean ROC', color = 'red')

    idx = SelectKBest(k=15).fit(data, labels)
    data = data[data.columns[idx.get_support()]]
    print(data.shape)
    tprs2 = []
    aucs2 = []
    mean_fpr2 = np.linspace(0, 1, 100)

    for trainIdx, testIdx in fold.split(data, labels):
        train, trainLabels = data.iloc[trainIdx], labels.iloc[trainIdx]
        test, testLabels = data.iloc[testIdx], labels.iloc[testIdx]

        lr.fit(train, trainLabels)
        acc = lr.score(test, testLabels)
        viz = plot_roc_curve(lr,test,testLabels)
        interp_tpr2 = np.interp(mean_fpr2, viz.fpr, viz.tpr)
        interp_tpr2[0] = 0.0
        tprs2.append(interp_tpr2)
        aucs2.append(viz.roc_auc)

        print(acc)
        avg_acc += acc

    mean_tpr2 = np.mean(tprs2, axis=0)
    mean_tpr2[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_tpr2)
    ax.plot(mean_fpr2, mean_tpr2, label='K-Best Mean ROC', color='green')

    data = full
    idx = SelectFromModel(LinearSVC(max_iter=1000)).fit(data,labels)
    data = data[data.columns[idx.get_support()]]
    print(data.shape)
    tprs3 = []
    aucs3 = []
    mean_fpr3 = np.linspace(0, 1, 100)

    for trainIdx, testIdx in fold.split(data, labels):
        train, trainLabels = data.iloc[trainIdx], labels.iloc[trainIdx]
        test, testLabels = data.iloc[testIdx], labels.iloc[testIdx]

        lr.fit(train, trainLabels)
        acc = lr.score(test, testLabels)
        viz = plot_roc_curve(lr, test, testLabels)
        interp_tpr3 = np.interp(mean_fpr3, viz.fpr, viz.tpr)
        interp_tpr3[0] = 0.0
        tprs3.append(interp_tpr3)
        aucs3.append(viz.roc_auc)

        print(acc)
        avg_acc += acc

    mean_tpr3 = np.mean(tprs3, axis=0)
    mean_tpr3[-1] = 1.0
    mean_auc3 = auc(mean_fpr3, mean_tpr3)
    ax.plot(mean_fpr3, mean_tpr3, label='LinearSVM Selected Mean ROC', color='blue')

    ax.legend()
    plt.show()
    avg_acc /= 10
    print('Average LR Acc: {0}'.format(avg_acc))
    print(aucs1)
    print(aucs2)
    print(aucs3)
    print(mean_auc1, mean_auc2, mean_auc3)
    
    

def full_experiment(data,labels):
    # Conducts full experiment on all models
    # Calls functions for latency and throughput

    fold = KFold(10, shuffle=True)

    # Variables for average ROC and AUC calculations
    lr_tprs = []
    lr_aucs = []
    lr_mean_fpr = np.linspace(0, 1, 100)
    svm_tprs = []
    svm_aucs = []
    svm_mean_fpr = np.linspace(0, 1, 100)
    nn_tprs = []
    nn_aucs = []
    nn_mean_fpr = np.linspace(0, 1, 100)

    avglr = avgsvm = avgnn = 0.0
    avgLRTNR = avgSVMTNR = avgNNTNR = 0.0
    lrC = svmC = 1

    fig, ax = plt.subplots()
    for trainIdx, testIdx in fold.split(data, labels):
        train, trainLabels = data.iloc[trainIdx], labels.iloc[trainIdx]
        test, testLabels = data.iloc[testIdx], labels.iloc[testIdx]

        # lrC = lr_tune(train, trainLabels)
        # svmC = svm_tune(train, trainLabels)
        # rate, batch = nn_tune(train, trainLabels)

        lr = LogisticRegression(solver='lbfgs', max_iter=1000, C=lrC)
        svm = LinearSVC(max_iter=10000, C=svmC)
        nn = keras.models.Sequential(
            [
                keras.layers.InputLayer((data.shape[1],)),
                keras.layers.Dense(120, activation='sigmoid'),
                keras.layers.Dense(20, activation='sigmoid'),
                keras.layers.Dense(1, activation='sigmoid')
            ]
        )
        nn.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy',
                   metrics=['accuracy'])

        lr.fit(train, trainLabels)
        svm.fit(train, trainLabels)
        nn.fit(train, trainLabels, batch_size=128, verbose=0, epochs=50)

        # Logistic Regression Results
        viz = plot_roc_curve(lr, test, testLabels)
        lrinterp_tpr = np.interp(lr_mean_fpr, viz.fpr, viz.tpr)
        lrinterp_tpr[0] = 0.0
        lr_tprs.append(lrinterp_tpr)
        lr_aucs.append(viz.roc_auc)
        lracc = lr.score(test,testLabels)
        pred = lr.predict(test)
        lrFPR = (pred[(testLabels == 0)] == 0).sum() / (testLabels == 0).sum()
        avglr += lracc
        avgLRTNR += lrFPR

        # SVM Results
        viz = plot_roc_curve(svm, test, testLabels)
        svminterp_tpr = np.interp(svm_mean_fpr, viz.fpr, viz.tpr)
        svminterp_tpr[0] = 0.0
        svm_tprs.append(svminterp_tpr)
        svm_aucs.append(viz.roc_auc)
        svmacc = svm.score(test,testLabels)
        pred = svm.predict(test)
        svmFPR = (pred[(testLabels == 0)] == 0).sum() / (testLabels == 0).sum()
        avgsvm += svmacc
        avgSVMTNR += svmFPR

        # Neural Network Results
        nnfpr, nntpr, nnthres = roc_curve(testLabels, nn.predict_proba(test))
        nninterp_tpr = np.interp(nn_mean_fpr, nnfpr, nntpr)
        nninterp_tpr[0] = 0.0
        nn_tprs.append(nninterp_tpr)
        nn_aucs.append(auc(nnfpr, nntpr))
        score, nnacc = nn.evaluate(test,testLabels, verbose=0)
        pred = (nn.predict(test) > 0.9).astype('int32')
        nnFPR = (pred[(testLabels == 0)] == 0).sum() / (testLabels == 0).sum()
        avgNNTNR += nnFPR
        avgnn += nnacc

    lr_mean_tpr = np.mean(lr_tprs, axis=0)
    lr_mean_tpr[-1] = 1.0
    lr_mean_auc = auc(lr_mean_fpr, lr_mean_tpr)
    ax.plot(lr_mean_fpr, lr_mean_tpr, label='LR Mean ROC', color='red')

    svm_mean_tpr = np.mean(svm_tprs, axis=0)
    svm_mean_tpr[-1] = 1.0
    svm_mean_auc = auc(svm_mean_fpr, svm_mean_tpr)
    ax.plot(svm_mean_fpr, svm_mean_tpr, label='SVM Mean ROC', color='green')

    nn_mean_tpr = np.mean(nn_tprs, axis=0)
    nn_mean_tpr[-1] = 1.0
    nn_mean_auc = auc(nn_mean_fpr, nn_mean_tpr)
    ax.plot(nn_mean_fpr, nn_mean_tpr, label='NN Mean ROC', color='blue')

    print('LR Average AUC: {0}'.format(lr_mean_auc))
    print('SVM Average AUC: {0}'.format(svm_mean_auc))
    print('NN Average AUC: {0}'.format(nn_mean_auc))
    print()

    avglr /= 10
    avgsvm /= 10
    avgnn /= 10

    print('LR Average Acc: {0}'.format(avglr))
    print('SVM Average Acc: {0}'.format(avgsvm))
    print('NN Average Ac: {0}'.format(avgnn))
    print()

    avgLRTNR /= 10
    avgSVMTNR /= 10
    avgNNTNR /= 10

    print('LR Average FPR: {0}'.format(avgLRTNR))
    print('SVM Average FPR: {0}'.format(avgSVMTNR))
    print('NN Average FPR: {0}'.format(avgNNTNR))
    print()

    ax.legend()
    plt.show()

    # Get performance results
    full_latency(lr,svm,nn,data)
    full_throughput(lr,svm,nn,data)



def full_latency(lr, svm, nn, data):
    # Finds the prediction latency of passed models on passed data
    # Duplicates and shuffles data to reach 1 million packets
    # Runs 1 million packet latency test 20 times for accurate average measure

    inx = np.array(data.index.repeat(42))
    np.random.shuffle(inx)
    data = data.loc[inx]
    avglr = avgsvm = avgnn = 0.0

    for _ in range(20):
        t0 = time.time()
        lr.predict(data)
        t = time.time() - t0
        avglr += t

        t0 = time.time()
        svm.predict(data)
        t = time.time() - t0
        avgsvm += t

        t0 = time.time()
        nn.predict(data)
        t = time.time() - t0
        avgnn += t

    avglr /= 20
    avgsvm /= 20
    avgnn /= 20

    print('Average LR time {0}'.format(avglr))
    print('Average SVM time {0}'.format(avgsvm))
    print('Average NN time {0}'.format(avgnn))
    print()

    plt.bar('LR', avglr)
    plt.bar('SVM', avgsvm)
    plt.bar('NN', avgnn)
    plt.title('Prediction Latency for ~1 million packets')
    plt.ylabel('Seconds')
    plt.show()



def full_throughput(lr,svm,nn,data):
    # Finds the prediction throughput of passed models on passed data
    # Duplicates and shuffles data to reach 1 million packets
    # Runs 1 second throughput test 20 times for accurate average measure

    inx = np.array(data.index.repeat(50))
    np.random.shuffle(inx)
    data = data.loc[inx]

    avglr = avgsvm = avgnn = 0.0
    for _ in range(20):
        lrpreds = 0
        t0 = time.time()
        while time.time() - t0 < 1:
            lr.predict(data.iloc[0:10000, :])
            lrpreds += 10000

        avglr += lrpreds

        svmpreds = 0
        t0 = time.time()
        while time.time() - t0 < 1:
            svm.predict(data.iloc[0:10000, :])
            svmpreds += 10000

        avgsvm += svmpreds

        nnpreds = 0
        t0 = time.time()
        while time.time() - t0 < 1:
            nn.predict(data.iloc[0:10000, :])
            nnpreds += 10000

        avgnn += nnpreds

    avglr /= 20
    avgsvm /= 20
    avgnn /= 20

    print('Avg LR throughput {0}'.format(avglr))
    print('Avg SVM throughput {0}'.format(avgsvm))
    print('Avg NN throughput {0}'.format(avgnn))

    plt.bar('LR', avglr)
    plt.bar('SVM', avgsvm)
    plt.bar('NN', avgnn)
    plt.title('Prediction Throughput for 1 second')
    plt.ylabel('Predictions')
    plt.show()



# Read in dataset, preprocess, then start experiment
data_path = input('Enter file path for full data (training and testing): ')
full = pd.read_csv(data_path)
data, labels = preprocess(full)

full_experiment(data,labels)