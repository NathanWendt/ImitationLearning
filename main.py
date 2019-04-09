from utils import Read, LabelFeatDictGen, Plotter
from algorithms import RCLF_ExactImit, Prediction, RCLF_DAgger
from sklearn import svm
import csv

NT_TEST = 'C:/Users/natha/OneDrive/Documents/WSU/Classes/CptS_577/HW1/datasets/nettalk_stress_test.txt'
NT_TRAIN = 'C:/Users/natha/OneDrive/Documents/WSU/Classes/CptS_577/HW1/datasets/nettalk_stress_train.txt'
OCR_TEST = 'C:/Users/natha/OneDrive/Documents/WSU/Classes/CptS_577/HW1/datasets/ocr_fold0_sm_test.txt'
OCR_TRAIN = 'C:/Users/natha/OneDrive/Documents/WSU/Classes/CptS_577/HW1/datasets/ocr_fold0_sm_train.txt'

ntCSVFileTrain = 'nt_trainPercept.csv'
ntCSVFileTest = 'nt_testPercept.csv'
ocrCSVFileTrain = 'ocr_trainPercept.csv'
ocrCSVFileTest = 'ocr_testPercept.csv'

ntLabelSet = ['00','01','02','03','04']
ocrLabelSet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# ntTest = Read(NT_TEST)
# ntTrain = Read(NT_TRAIN)
# ocrTest = Read(OCR_TEST)
# ocrTrain = Read(OCR_TRAIN)

ntLabelDict = LabelFeatDictGen(ntLabelSet)
ocrLabelDict = LabelFeatDictGen(ocrLabelSet)

# reachBack = 3
# d_max = 1
# B = 0.9
#
# ntTrain_acc = []
# ntTest_acc = []
# ocrTrain_acc = []
# ocrTest_acc = []
# for i in range(6):
#     B = float(i)*0.1+0.5
#     print('$$$$$$', B, '$$$$$$')
#     temp_ntTrain_acc = []
#     temp_ntTest_acc = []
#     temp_ocrTrain_acc = []
#     temp_ocrTest_acc = []
#     [ntclf, ntL] = RCLF_ExactImit(ntTrain, reachBack, ntLabelDict)
#     [ocrclf, ocrL] = RCLF_ExactImit(ocrTrain, reachBack, ocrLabelDict)
#     temp_ntTrain_acc.append(Prediction(ntclf, ntTrain, reachBack, ntLabelDict))
#     temp_ntTest_acc.append(Prediction(ntclf, ntTrain, reachBack, ntLabelDict))
#     temp_ocrTrain_acc.append(Prediction(ocrclf, ocrTrain, reachBack, ocrLabelDict))
#     temp_ocrTest_acc.append(Prediction(ocrclf, ocrTrain, reachBack, ocrLabelDict))
#     for j in range(10):
#         [dag_ntclf, dag_ntL] = RCLF_DAgger(ntTrain, d_max, reachBack, ntLabelDict, ntL, ntclf, B)
#         [dag_ocrclf, dag_ocrL] = RCLF_DAgger(ocrTrain, d_max, reachBack, ocrLabelDict, ocrL, ocrclf, B)
#         temp_ntTrain_acc.append(Prediction(dag_ntclf, ntTrain, reachBack, ntLabelDict))
#         temp_ntTest_acc.append(Prediction(dag_ntclf, ntTest, reachBack, ntLabelDict))
#         temp_ocrTrain_acc.append(Prediction(dag_ocrclf, ocrTrain, reachBack, ocrLabelDict))
#         temp_ocrTest_acc.append(Prediction(dag_ocrclf, ocrTest, reachBack, ocrLabelDict))
#         print('B: ', B, ' d_max: ', j)
#         print('ntTrain Acc: ', temp_ntTrain_acc[j], ' ntTest Acc: ', temp_ntTest_acc[j])
#         print('ocrTrain Acc: ', temp_ocrTrain_acc[j], ' ocrTest Acc: ', temp_ocrTest_acc[j])
#     ntTrain_acc.append(temp_ntTrain_acc)
#     ntTest_acc.append(temp_ntTest_acc)
#     ocrTrain_acc.append(temp_ocrTrain_acc)
#     ocrTest_acc.append(temp_ocrTest_acc)


# with open(ntCSVFileTrain, 'w') as file:
#     writer = csv.writer(file)
#     writer.writerows(ntTrain_acc)
#
# with open(ntCSVFileTest, 'w') as file:
#     writer = csv.writer(file)
#     writer.writerows(ntTest_acc)
#
# with open(ocrCSVFileTrain, 'w') as file:
#     writer = csv.writer(file)
#     writer.writerows(ocrTrain_acc)
#
# with open(ocrCSVFileTest, 'w') as file:
#     writer = csv.writer(file)
#     writer.writerows(ocrTest_acc)

#Plotter(ntCSVFileTrain, 'Net Talk Training Error')
#Plotter(ntCSVFileTest, 'Net Talk Testing Error')
Plotter(ocrCSVFileTrain, 'OCR Training Error')
#Plotter(ocrCSVFileTest, 'OCR Testing Error')
#acc = Prediction(dag_ntclf, ntTest, reachBack, ntLabelDict)
#print(acc)
#Prediction(ocrclf, ocrTrain, reachBack, ocrLabelDict)
