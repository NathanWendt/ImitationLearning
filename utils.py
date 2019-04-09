import csv
import re
import numpy as np
import matplotlib.pyplot as plt


def Read(file):
    handle = open(file)
    setList = []
    dataList = []
    for lines in handle:
        data = lines.split()
        if not re.match(r'\w', lines):
            setList.append(dataList)
            dataList = []
            continue
        binaryString = data[1][2:]
        binaryInts = []
        for i in binaryString:
            binaryInts.append(int(i))
        binaryInts = np.array(binaryInts)
        dataList.append([data[0], binaryInts, data[2]])
    return setList

def LabelFeatDictGen(labelSet):
    labelDict = dict()
    labelLen = len(labelSet)
    for i in range(labelLen):
        label = labelSet[i]
        feat = np.zeros(labelLen)
        feat[i] = 1
        labelDict[label] = feat
    labelDict['em'] = np.zeros(labelLen)
    return labelDict

def Plotter(csv_handle, title):
    with open(csv_handle) as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        _list = []
        for row in csv_reader:
            if row:
                _temp = []
                for item in row:
                    _temp.append(float(item))
                _list.append(_temp)
                line_count += 1
        _array = np.array(_list)

    Ndecimals = 3
    decade = 10**Ndecimals
    _array = np.trunc(_array*decade)/decade

    _len = len(_array[0])
    x = [x for x in range(_len)]
    #print('x ax: ', x, ' len: ', _len)

    for i in range(len(_array)):
        labelval = str(0.5+0.1*i)
        plt.plot(x, _array[i], label=labelval)

    plt.xlabel('DAgger Iterations')
    plt.ylabel('Accuracy')

    plt.title(title)

    plt.legend()

    plt.show()
