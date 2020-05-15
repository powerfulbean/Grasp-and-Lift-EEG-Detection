# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:37:42 2020

@author: Jin Dou
"""

from StellarBrainwav.DataStruct.LabelData import CLabels, CLabelInfoGeneral
from StellarBrainwav.DataStruct.RawData import CRawData
from StellarBrainwav.DataIO import getFileName
from StellarBrainwav.DataStruct.StimuliData import CStimuli
from StellarBrainwav.DataProcessing.DeepLearning import CPytorch,CTorchNNYaml
import pandas as pd

class CGALEDRawData(CRawData):
    
    def __init__(self,srate):
        super(CGALEDRawData,self).__init__()
        self.sampleRate = srate
    
    def readFile(self,fileName):
        df = pd.read_csv(fileName)
        self.timestamps = list(df['id'])
        colNames = list(df.keys())
        channelsName = colNames[1:]
        self.description['chName'] = channelsName
        self.numChannels = len(channelsName)
        self.rawdata = df[channelsName].values.T

class CGALEDLabels(CLabels):
    
    def __init__(self):
        super(CGALEDLabels,self).__init__()
        self.oLabelInfo = None
        
    def readFile(self,filePath):
        self.description = getFileName(filePath)
        df = pd.read_csv(filePath)
        self.timestamps = list(df['id'])
        colNames = list(df.keys())
        eventsName = colNames[1:]
        self.oLabelInfo = CLabelInfoGeneral(self.description,0,eventsName)
        self.rawdata = df[eventsName].values.T
    
    def loadStimuli(self):
        pass

def getSeqIndex(filePath:str):
    fileName = getFileName(filePath)
    ans = fileName[fileName.rfind('_')+1:]
    return ans
      
def getSeriesName(filePath:str):
    fileName = getFileName(filePath)
    ans = fileName[fileName.find('_')+1:fileName.rfind('_')]
    return ans

def getSubjectName(filePath:str):
    fileName = getFileName(filePath)
    ans = fileName[0:fileName.find('_')]
    return ans

def getSeriesId(filePath:str):
    seriesName = getSeriesName(filePath)
    ans = int(seriesName[6:])
    return ans
        
def keysFunc(timestamp:str):
    subjName = getSubjectName(timestamp)
    seriesName = getSeriesName(timestamp)
    seqIndex = getSeqIndex(timestamp)
    subjIndex = int(subjName[4:])
    seriesIndex = int(seriesName[6:])
    return (subjIndex,seriesIndex,seqIndex)


pytorchRoot = CPytorch()._ImportTorch()
nn = pytorchRoot.nn

class CCRNN(pytorchRoot.nn.Module):
    def __init__(self,cnnDir,denseDir,lstmInSize,lstmHidSize):
        super().__init__()
        self.oLstm = nn.LSTM(input_size = lstmInSize,hidden_size = lstmHidSize)
        oCNNYaml = CTorchNNYaml()
        self.oCNN = oCNNYaml(cnnDir)
        self.oDense = oCNNYaml(denseDir)
        
    def forward(self,x):
        #input shape (num_seq, batch, input_size)     
        
        xCNNList = list()
        for i in range(x.shape[0]):
#            print(x[i])
            xCNNTemp = x[i] # shape (batch,input_size)
            xCNNTemp1 = xCNNTemp.view(x[i].shape[0],1,x[i].shape[1])
            xCNNOutTemp = self.oCNN(xCNNTemp1)# shape (batch,output_size)
            xCNNOutTemp = xCNNOutTemp.view(1,xCNNOutTemp.shape[0],xCNNOutTemp.shape[1])
            xCNNList.append(xCNNOutTemp) 
        
        xCNNOut = pytorchRoot.cat(xCNNList,0)
        xLSTMOut = self.oLstm(xCNNOut)
        xDenseOut = self.oDense(xLSTMOut[0][-1])
        return xDenseOut

class CCRNNChannels(pytorchRoot.nn.Module):
    def __init__(self,cnnDir,denseDir,lstmInSize,lstmHidSize):
        super().__init__()
        self.oLstm = nn.LSTM(input_size = lstmInSize,hidden_size = lstmHidSize)
        oCNNYaml = CTorchNNYaml()
        self.oCNN = oCNNYaml(cnnDir)
        self.oDense = oCNNYaml(denseDir)
        
    def forward(self,x):
        #input shape (num_seq, batch, input_size)     
        
        xCNNList = list()
        for i in range(x.shape[0]):
#            print(x[i])
            xCNNTemp = x[i] # shape (batch,input_size)
            xCNNTemp1 = xCNNTemp
            xCNNOutTemp = self.oCNN(xCNNTemp1)# shape (batch,output_size)
            xCNNOutTemp = xCNNOutTemp.view(1,xCNNOutTemp.shape[0],xCNNOutTemp.shape[1])
            xCNNList.append(xCNNOutTemp) 
        
        xCNNOut = pytorchRoot.cat(xCNNList,0)
        xLSTMOut = self.oLstm(xCNNOut)
        xDenseOut = self.oDense(xLSTMOut[0][-1])
        return xDenseOut

class CSlidingWinDataset(pytorchRoot.utils.data.Dataset):
    
    def __init__(self, *tensors,window):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        zeroTensor = pytorchRoot.FloatTensor([0])
#        print(tensors[0].size())
        shape = list()
        for i in range(1,len(tensors[0].shape)):
            shape.append(tensors[0].shape[i])
            
        paddingTensor = zeroTensor.expand(window-1,*shape)
        self.window = window
#        self.tensors = tensors
        Temp = (pytorchRoot.cat([paddingTensor,tensors[0]]),) + tensors[1:]
        self.tensors = tuple()
        for i in range(len(Temp)):
            self.tensors += (Temp[i].cuda(),)

    def __getitem__(self, index):
        return tuple(tensor[index : self.window + index] if idx==0 else tensor[index] for idx,tensor in enumerate(self.tensors))

    def __len__(self):
        return self.tensors[0].size(0) - self.window + 1