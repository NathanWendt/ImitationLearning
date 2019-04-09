from sklearn import svm
from sklearn.linear_model import Perceptron, SGDClassifier
import numpy as np
import random

def getFeat(x, y_back, labelFeatDict):
    _feat = x
    for label in y_back:
        _feat = np.concatenate([_feat, labelFeatDict[label]])
    return _feat

def Prediction(clf, D, reachBack, labelFeatDict):
    tot = 0
    correct = 0
    for seq in D:
        seqLen = len(seq)
        _y = []
        [_y.append('em') for item in range(reachBack)]

        for item in seq:
            y_temp = item[2]
            _feat = getFeat(item[1], _y, labelFeatDict)
            y_hat = clf.predict([_feat])
            y_hat = str(y_hat[0])
            #print(y_hat, y_temp)
            if y_hat == y_temp:
                correct += 1
            tot += 1
            #print('total: ', tot, 'correct: ', correct)
            acc = correct/tot
            #print('y_gold: ', y_temp, 'y_hat: ', y_hat, 'accuracy: ', acc)
    return acc

def RCLF_ExactImit(D, reachBack, labelFeatDict):
    _L = []
    for seq in D:
        seqLen = len(seq)
        _y = []
        [_y.append('em') for item in range(reachBack)]
        labels = [item[2] for item in seq]
        for item in seq:
            y_temp = item[2]
            #print('current y: ', y_temp, 'back y: ', _y, 'labels: ', labels)
            _feat = getFeat(item[1], _y, labelFeatDict)
            _L.append([_feat,y_temp])
            _y.pop(0)
            _y.append(y_temp)
    #clf = svm.LinearSVC(C=1, max_iter=3000)
    #clf = svm.SVC(gamma='scale')
    clf = Perceptron(tol=1e-3, random_state=123)
    #clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=500, tol=1e-3)
    x = np.array([item[0] for item in _L])
    y = [item[1] for item in _L]
    clf.fit(x,y)
    return [clf, _L]

def RCLF_DAgger(D, d_max, reachBack, labelFeatDict, L, clf, B):
    acc = []
    print('Beginning DAgger, L = ', len(L))
    for i in range(d_max):
        for seq in D:
            seqLen = len(seq)
            _y = []
            [_y.append('em') for item in range(reachBack)]
            labels = [item[2] for item in seq]
            #leave = 0
            for item in seq:
                # if leave == 1:
                #     break
                y_temp = item[2]
                _feat = getFeat(item[1], _y, labelFeatDict)
                roll = random.random()
                if roll < B:
                    y_hat = y_temp
                else:
                    y_hat = clf.predict([_feat])
                    y_hat = str(y_hat[0])
                #print('current y: ', y_temp, 'back y: ', _y, 'labels: ', labels)
                _y.pop(0)
                _y.append(y_hat)

                if y_hat != y_temp:
                    #leave = 1
                    L.append([_feat, y_temp])
        B = np.power(B,1.5)
        x = np.array([item[0] for item in L])
        y = [item[1] for item in L]
        clf.fit(x,y)
        #acc.append(Prediction(clf, D, reachBack, labelFeatDict))
        #print('itter:', i, 'accuracy: ', acc, 'L: ', len(L))
    return [clf, L]
