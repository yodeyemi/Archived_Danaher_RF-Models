import numpy as np
from scipy.signal import medfilt
import os
import matplotlib.pyplot as plt
import cv2
import joblib

import re 
from os import path, makedirs


# Works for Gram Neg training/inference
strainSet = ['E.COLI', 'K.OXYTOCA', 'S.MARCESCENS', 'P.VULGARIS', 'A.BAUMANNII', 'E.AEROGENES', 'C.FREUNDII', 'K.PNEUMONIAE', 'E.CLOACAE', 'P.RETTGERI', 'P.MIRABILIS', 'P.AERUGINOSA']
antiSet = ['GM', 'TOB', 'CPFX', 'AS', 'LVFX', 'IPM', 'MEPM', 'ST', 'CAZ', 'CTX', 'CFPM', 'TP']
# Works for Gram Pos training/inference
# strainSet = ['S.SAPROPHYTICUS', 'S.WARNERI', 'E.FAECIUM', 'S.AUREUS', 'S.INTERMEDIUS', 'S.CAPITIS', 'S.HOMINIS', 'S.EPIDERMIDIS', 'E.FAECALIS']
# antiSet = ['CFX-S', 'VCM', 'I-CLDM', 'MPIPC', 'EM', 'LVFX', 'CLDM', 'DAP', 'LZD', 'CPFX', 'ST', 'PCG', 'TC']
# Works for inference only
# strainSet = []
# antiSet = []

# dictionary of all pred data
predByTest = {}

# error analysis output parameters
isErrAnalysis = False
errDataFolder = './'
colorMap = ['r', 'g', 'b', 'm', 'y', 'c', 'r', 'g', 'b', 'm', 'y', 'c']
markerMap = ['^', '^', '^', 'o', 'o', 'o', 'v', 'v', 'v', 's', 's', 's']


def resetInferDataDict(errAnalysis=False, dataFolder='./'):
    global predByTest
    global isErrAnalysis
    global errDataFolder

    predByTest = {}
    isErrAnalysis = errAnalysis
    errDataFolder = dataFolder


def saveImgSeries(img, filePath):
    img = np.clip((img+1.5)*24.0, 0, 255).astype(np.uint8)
    H, W = img.shape[1:3]
    img = img[:,H//4:H//4*3:2,W//4:W//4*3:2,0]
    img[:,:,0:5] = 2.5
    img = img.transpose([1,0,2])
    imgShape = img.shape
    img = img.reshape([imgShape[0], imgShape[1]*imgShape[2], 1])

    #plt.imsave(filePath, img, cmap='gray')
    cv2.imwrite(filePath, img)


def addInferData(infoBatch, labelBatch, yProbBatch, numFeatureBatch=np.array([]), imgBatch=np.array([]), imgRefBatch=np.array([])):
    for n in range(labelBatch.shape[0]):
        index = infoBatch[n][1]
        strain = infoBatch[n][2]
        ID = infoBatch[n][3]
        anti = infoBatch[n][4]
        conc = infoBatch[n][5]
        imgFolder = infoBatch[n][-1]
        
        if strain not in strainSet:
            strainSet.append(strain)
        
        if anti not in antiSet:
            antiSet.append(anti)

        if strain not in predByTest:
            predByTest[strain] = {}

        if anti not in predByTest[strain]:
            predByTest[strain][anti] = {}
            
        if ID not in predByTest[strain][anti]:
            predByTest[strain][anti][ID] = {'index': [], 'conc': [], 'truth': [], 'pred': [],\
                 'imgFolder': '', 'numFeature': []}
            
        if float(conc) not in predByTest[strain][anti][ID]['conc']:
            predByTest[strain][anti][ID]['index'].append(int(index))
            predByTest[strain][anti][ID]['conc'].append(float(conc))
            predByTest[strain][anti][ID]['truth'].append(int(labelBatch[n]))
            predByTest[strain][anti][ID]['pred'].append(float(yProbBatch[n]))
            if len(predByTest[strain][anti][ID]['imgFolder']) == 0:
                predByTest[strain][anti][ID]['imgFolder'] = os.path.abspath(os.path.join(imgFolder, os.pardir))

            if isErrAnalysis:
                predByTest[strain][anti][ID]['numFeature'].append(numFeatureBatch[n])

                if imgBatch.size > 0:
                    dataFolder = errDataFolder + f'{strain}/{anti}/{ID}/{conc}'
                    if not os.path.isdir(dataFolder): # to avoid repeated attempts by different workers
                        createFolder(dataFolder)
                        filePath = dataFolder + '/time_series.png'
                        saveImgSeries(imgBatch[n], filePath)

                if imgRefBatch.size > 0:
                    dataFolder = errDataFolder + f'{strain}/{anti}/{ID}/0'
                    if not os.path.isdir(dataFolder):
                        createFolder(dataFolder)
                        filePath = dataFolder + '/time_series.png'
                        saveImgSeries(imgRefBatch[n], filePath)

        
def maxToRightFilt(data):
    dataFilt = np.array([np.max(data[n:n+2]) for n in range(data.size-1)] + [data[-1]])
    return dataFilt


def analyzeInferData(testData, predThresh=0.5):
    concList = np.array(testData['conc'])
    truthList = np.array(testData['truth'])
    predList = np.array(testData['pred'])
    indexList = np.array(testData['index'])
    numFeatureList = np.array(testData['numFeature'])

    sortInd = np.argsort(concList)
    concList = concList[sortInd]
    truthList = truthList[sortInd]
    predList = predList[sortInd]
    indexList = indexList[sortInd]
    testData['conc'] = list(concList)
    testData['truth'] = list(truthList)
    testData['pred'] = list(predList)
    testData['index'] = list(indexList)
    if numFeatureList.size > 0:
        numFeatureList = numFeatureList[sortInd]
        testData['numFeature'] = numFeatureList

    predList = (predList > predThresh).astype(np.int)
    # predList = np.pad(predList, (1,1), 'edge')
    # predList = (medfilt(predList, kernel_size=3)[1:-1]).astype(np.int)
    predList = maxToRightFilt(predList).astype(np.int)

    truthDiff = np.diff(truthList)
    if len(np.argwhere(truthDiff == 1)) != 0:
        micTruth = 0xffff
    else:
        micTruth = np.argwhere(truthDiff == -1)
        if len(micTruth) == 0:
            micTruth = 0 if truthList[0] == 0 else truthList.size
        elif len(micTruth) == 1:
            micTruth = micTruth[0][0] + 1
        else:
            micTruth = 0xffff       

    predDiff = np.diff(predList)
    if len(np.argwhere(predDiff == 1)) != 0:
        micPred = 0xffff
    else:
        micPred = np.argwhere(predDiff == -1)
        if len(micPred) == 0:
            micPred = 0 if predList[0] == 0 else predList.size
        elif len(micPred) == 1:
            micPred = micPred[0][0] + 1
        else: # not possible
            micPred = 0xffff

    testData['micTruth'] = micTruth
    testData['micPred'] = micPred
    return testData


def compileAllInferData(predThresh=0.5):
    groupCount = np.zeros((len(strainSet), len(antiSet)))
    accAACount = np.zeros((len(strainSet), len(antiSet)))
    accEACount = np.zeros((len(strainSet), len(antiSet)))
    undetermCount = np.zeros((len(strainSet), len(antiSet)))

    for strain in predByTest:
        rowInd = strainSet.index(strain)
        
        @joblib.delayed
        def processAntiData(anti):
            for ID in predByTest[strain][anti]:
                predByTest[strain][anti][ID] = analyzeInferData(predByTest[strain][anti][ID], predThresh)

                micTruth = predByTest[strain][anti][ID]['micTruth']
                micPred = predByTest[strain][anti][ID]['micPred']
                
                colInd = antiSet.index(anti)

                if  micTruth == 0xffff:
                    print(f'!!! Undetermined MIC in labelled data: {ID}, {anti} !!!')
                else:
                    groupCount[rowInd, colInd] += 1

                    if micPred != 0xffff:
                        micBias = micPred - micTruth
                        accAACount[rowInd, colInd] += micBias==0
                        accEACount[rowInd, colInd] += abs(micBias)<=1
                    else:
                        #print(f'!!! Undetermined MIC in test data: {ID}, {anti} !!!')
                        undetermCount[rowInd, colInd] += 1

                if isErrAnalysis:
                    origDataFolder = errDataFolder + f'{strain}/{anti}/{ID}'
                    dataFolder = origDataFolder + f'_{micPred-micTruth}' # append MIC bias to folder name

                    if os.path.isdir(origDataFolder) and (not os.path.isdir(dataFolder)):   # to avoid repeated attempts by different workers
                        os.system(f'mv {origDataFolder} {dataFolder}')
                        os.system('ln -s %s %s/imagedata' %(predByTest[strain][anti][ID]['imgFolder'], dataFolder))

                        #figure = plt.figure(figsize=(15,4))
                        figure = plt.figure(figsize=(4,15))
                        nConc = len(predByTest[strain][anti][ID]['conc'])
                        # ax1 = plt.subplot(1, 3, 1)
                        ax1 = plt.subplot(3, 1, 1)
                        plt.plot(predByTest[strain][anti][ID]['truth'], 'k+-.', lw=2)
                        plt.plot(predByTest[strain][anti][ID]['pred'], 'bo-', lw=1)
                        plt.plot(np.ones(nConc)*predThresh, 'r-', lw=1)
                        ax1.set_xticks(np.arange(0, nConc))
                        ax1.set_xticklabels(predByTest[strain][anti][ID]['conc'])
                        plt.xlabel('conc')
                        plt.ylabel('G/I')
                        plt.title(f'{strain}, {anti}, {ID}')

                        # ax2 = plt.subplot(1, 3, 2)
                        ax2 = plt.subplot(3, 1, 2)
                        for n, featureData in enumerate(predByTest[strain][anti][ID]['numFeature']):
                            conc = predByTest[strain][anti][ID]['conc'][n]
                            if n == 0:
                                plt.plot(featureData[1:,3]/1000, 'k+-.', label='0', lw=2)
                            plt.plot(featureData[1:,1]/1000, f'{colorMap[n]}{markerMap[n]}-', label=f'{conc}', lw=1)
                        # plt.legend(loc=2)
                        ax2.set_xticks(np.arange(0, featureData.shape[0]-1))
                        ax2.set_xticklabels(np.arange(2, featureData.shape[0]+1))
                        plt.xlabel('cycle')
                        plt.ylabel('bac count (K)')

                        # ax3 = plt.subplot(1, 3, 3)
                        ax3 = plt.subplot(3, 1, 3)
                        for n, featureData in enumerate(predByTest[strain][anti][ID]['numFeature']):
                            conc = predByTest[strain][anti][ID]['conc'][n]
                            if n == 0:
                                plt.plot(featureData[1:,2]/1000, 'k+-.', label='0', lw=2)
                            plt.plot(featureData[1:,0]/1000, f'{colorMap[n]}{markerMap[n]}-', label=f'{conc}', lw=1)
                        plt.legend(loc=2)
                        ax3.set_xticks(np.arange(0, featureData.shape[0]-1))
                        ax3.set_xticklabels(np.arange(2, featureData.shape[0]+1))
                        plt.xlabel('cycle')
                        plt.ylabel('area sum (K)')
                        
                        plt.tight_layout()
                        plt.savefig(dataFolder+'/report_curves.png', format='png')
                        plt.close(figure)
                    # else:
                    #     print(f'-------- skip a repeated attempt at {strain}/{anti}/{ID} with bias of {micPred-micTruth}')

        parallelProc = joblib.Parallel(np.max([1, np.min([os.cpu_count(), len(predByTest[strain].keys())])]), 'threading')
        parallelProc(processAntiData(anti) for anti in predByTest[strain])

    return groupCount, accAACount, accEACount, undetermCount, strainSet, antiSet,
        