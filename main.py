import sys
sys.path.append('..')
from StellarBrainwav.DataIO import CDirectoryConfig, getFileList,CLog
from StellarBrainwav.DataStruct.RawData import CRawData #LabelData, 
from StellarBrainwav.DataStruct.DataSet import CDataOrganizorLite
from StellarBrainwav.outsideLibInterfaces import CIfMNE
from StellarBrainwav.Helper.StageControl import CStageControl
from StellarBrainwav.Helper.DataObjectTransform import CEpochToDataLoader, CDataRecordToDataLoader
from StellarBrainwav.DataProcessing.DeepLearning import CTorchNNYaml,CPytorch
from StellarBrainwav.DataProcessing.SignalProcessing import MNECutInFreqBands
from MiddleWare import CGALEDRawData, CGALEDLabels, keysFunc, getSeriesId, CCRNN,CSlidingWinDataset,CCRNNChannels #getSeriesName, getSubjectName, getSeqIndex,


'''
the *_data.csv files contain the raw 32 channels EEG data (sampling rate 500Hz)
the *_events.csv files contains the ground truth frame-wise labels for all events
'''
stageList = [1,7,8]


dirList = ['Root','Train','Test','MiddleStage','Output','Models']
dataExt = '_data.csv'
eventsExt = '_events.csv'
seriesForTest = [7,8]
oDir = CDirectoryConfig(dirList,'Dataset.conf')
oDir.checkFolders()

oLog = CLog(oDir['Output'],'RunnningLog')
oStageCtrl = CStageControl(stageList,oLog)
dataFiles = getFileList(oDir['Train'],dataExt)
oData = CGALEDRawData(500)
oData.readFile(dataFiles[0])
chanList = oData.description['chName']
#oMNE = CIfMNE(chanList,500,'eeg',oLog = oLog)
#oMNE.Montage = oMNE.LibMNE.channels.read_montage('chanlabels_32channel_test',path = oDir['Root'])

if oStageCtrl(1) is True:
    dataFiles = getFileList(oDir['Train'],dataExt)
    eventFiles = getFileList(oDir['Train'],eventsExt)
    oData = CGALEDRawData(500)
    oData.readFile(dataFiles[0])
    oEvent = CGALEDLabels()
    oEvent.readFile(eventFiles[0])
    
    EventList = oEvent.oLabelInfo.labelClassList
    
    oDataOrgTrain = CDataOrganizorLite(oData.numChannels,oData.sampleRate,
                                  oData.description['chName'],keysFunc)
    oDataOrgTest =  CDataOrganizorLite(oData.numChannels,oData.sampleRate,
                                  oData.description['chName'],keysFunc)
    
    oLog.safeRecordTime('preparation finished')

if oStageCtrl(2) is True:
    for idx,dataFile in enumerate(dataFiles):
        oDataTemp = CGALEDRawData(500)
        oDataTemp.readFile(dataFile)
        oEventTemp = CGALEDLabels()
        oEventTemp.readFile(eventFiles[idx])
        
        if(getSeriesId(dataFile) in seriesForTest):
            oDataOrgTest.insert(oDataTemp,oEventTemp)
        else:
            oDataOrgTrain.insert(oDataTemp,oEventTemp)
    
        del oDataTemp
        del oEventTemp
    
    oLog.safeRecordTime('Data Organization finished')
    
    oLog.safeRecordTime('Epochs Preparation Started')
    
    for idx,event in enumerate(EventList):
        oLog.safeRecordTime(str(idx))
        oDataSet,eventList = oDataOrgTrain.dataSetBasedOnStimuliDesc(event,1)
    #    oDataSet.name = eventList
    #    oDataSet.save(oDir['MiddleStage'],event + '_train')
        eventIdList = [idx] * len(eventList)
        oEpoch = oMNE.CDataSetToEpochs(oDataSet,eventIdList,{event:idx})
        oEpoch.save(oDir['MiddleStage']+event+ '_train' + '-epo.fif')
        del oDataSet
        del oEpoch
        
    oLog.safeRecordTime('Epochs For Train Preparation Finished')
    
    for idx,event in enumerate(EventList):
        oLog.safeRecordTime(str(idx))
        oDataSet,eventList = oDataOrgTest.dataSetBasedOnStimuliDesc(event,1)
    #    oDataSet.name = eventList
    #    oDataSet.save(oDir['MiddleStage'],event + '_test')
        eventIdList = [idx] * len(eventList)
        oEpoch = oMNE.CDataSetToEpochs(oDataSet,eventIdList,{event:idx})
        oEpoch.save(oDir['MiddleStage']+event+ '_test' + '-epo.fif')
        del oDataSet
        del oEpoch
        
    oLog.safeRecordTime('Epochs For Test Preparation Finished')

if oStageCtrl(3) is True:
    epochFiles_train = getFileList(oDir['MiddleStage'],'_train-epo.fif')
    epochFiles_test  = getFileList(oDir['MiddleStage'],'_test-epo.fif')
    epochs_train = [oMNE.LibMNE.read_epochs(file) for file in epochFiles_train]
    epochs_test = [oMNE.LibMNE.read_epochs(file) for file in epochFiles_test]
    epochTrain = oMNE.LibMNE.concatenate_epochs(epochs_train)
    epochTest = oMNE.LibMNE.concatenate_epochs(epochs_test)
    epochTrain.save(oDir['MiddleStage'] + 'train' + '_Total-epo.fif')
    epochTest.save(oDir['MiddleStage'] + 'test' + '_Total-epo.fif')
    del epochFiles_train
    del epochFiles_test
    del epochTrain
    del epochTest
    
if oStageCtrl(4) is True:
    file1 = getFileList(oDir['MiddleStage'],'train_Total-epo.fif')[0]
    file2 = getFileList(oDir['MiddleStage'],'test_Total-epo.fif')[0]
    epochTrain = oMNE.LibMNE.read_epochs(file1)
    epochTest = oMNE.LibMNE.read_epochs(file2)
    epochTrain = epochTrain.filter(0,30,n_jobs = 3,method='iir')
    epochTest = epochTest.filter(0,30,n_jobs = 3,method='iir')
    
if oStageCtrl(5) is True:
    ''' 
    
    
    '''
    epochTrainList = list()
    files = getFileList(oDir['MiddleStage']+'ClassifiedData/','train-epo.fif')
    for file in files:
        epochTrainList.append(oMNE.LibMNE.read_epochs(file))

if oStageCtrl(6) is True:
    '''
    Test pytorch yaml
    '''
    ymlFiles = getFileList(oDir['Models'],'yml')
    oCNNYaml = CTorchNNYaml()
    oCNN = oCNNYaml(oDir['Models']+'2DCNN_1.yml')
    oDataLoaderTrans = CEpochToDataLoader()
    trainDataLoader = oDataLoaderTrans(epochTrain)
    testDataLoader = oDataLoaderTrans(epochTest)
    ans = oCNNYaml.fitClassificationModel(oCNN,trainDataLoader,testDataLoader,100,0.001,0.005)
    
    
if oStageCtrl(7) is True:
    '''
    CRNN model data prep. and test 
    '''
    # prepare time series data
    for idx,dataFile in enumerate(dataFiles):
        oDataTemp = CGALEDRawData(500)
        oDataTemp.readFile(dataFile)
        oEventTemp = CGALEDLabels()
        oEventTemp.readFile(eventFiles[idx])
        
        if(getSeriesId(dataFile) in seriesForTest):
            oDataOrgTest.insert(oDataTemp,oEventTemp)
        else:
            oDataOrgTrain.insert(oDataTemp,oEventTemp)
    
        del oDataTemp
        del oEventTemp
        if(idx >=7):
            break

if oStageCtrl(8) is True:    
    cnnDir = oDir['Models']+'RCNN_CNN.yml'
    denseDir = oDir['Models']+'RCNN_Dense.yml'
    oCRNN = CCRNN(cnnDir,denseDir,256,100).cuda()
    oDataRecordTrain = oDataOrgTrain.dataRecordBasedOnTime()
    oDataRecordTest = oDataOrgTest.dataRecordBasedOnTime()
    oDataLoaderTrans = CDataRecordToDataLoader()
    
    argsTrain = {'DataRecordArgs':{'window':100},
            'DataLoaderArgs':{'shuffle':False,'batch_size':100},
            'SamplerArgs':{'replacement':True,'num_samples':500}
            }
    
    argsTest = {'DataRecordArgs':{'window':100},
            'DataLoaderArgs':{'shuffle':False,'batch_size':100},
            'SamplerArgs':{'replacement':True,'num_samples':200}
            }
    
    pytorchRoot = CPytorch().Lib
    samplerType = pytorchRoot.utils.data.RandomSampler
    trainDataLoader = oDataLoaderTrans(oDataRecordTrain,CSlidingWinDataset,samplerType,**argsTrain)
    testDataLoader = oDataLoaderTrans(oDataRecordTest,CSlidingWinDataset,samplerType,**argsTest)
    
    oLossFunc = pytorchRoot.nn.BCELoss()
    metrics = CPytorch().trainClassificationModel(oCRNN,trainDataLoader,testDataLoader,10,0.001,0.001,oLossFunc)
    
    oLog.safeRecord('#train_loss\ttest_loss\ttrain_accu\ttest_accu')
    for metric in metrics:
        logTemp = str(metric[0]) + '\t' + str(metric[1]) + '\t' + str(metric[2]) + '\t' + str(metric[3])
        oLog.safeRecord(logTemp)
    
    #load model 
#    oDataOrgTrain
    #train

if oStageCtrl(9) is True:
    cnnDir = oDir['Models']+'RCNN_CNN_4channels.yml'
    denseDir = oDir['Models']+'RCNN_Dense.yml'
    
    oDataRecordTrain = oDataOrgTrain.dataRecordBasedOnTime()
    oDataRecordTest = oDataOrgTest.dataRecordBasedOnTime()
    
    oRawTrain = oMNE.getMNERaw(oDataRecordTrain.data)
    oRawTest = oMNE.getMNERaw(oDataRecordTest.data)
    
    bands = [0.5,4,7,15,30]
    
    oBandedDataTrain = MNECutInFreqBands(oRawTrain,bands)
    oBandedDataTest = MNECutInFreqBands(oRawTest,bands)
    
    oDataRecordTrain.data = oBandedDataTrain
    oDataRecordTest.data  = oBandedDataTest
    
    oDataLoaderTrans = CDataRecordToDataLoader()
    
    argsTrain = {'DataRecordArgs':{'window':100},
            'DataLoaderArgs':{'shuffle':False,'batch_size':100},
            'SamplerArgs':{'replacement':True,'num_samples':100000}
            }
    
    argsTest = {'DataRecordArgs':{'window':100},
            'DataLoaderArgs':{'shuffle':False,'batch_size':100},
            'SamplerArgs':{'replacement':True,'num_samples':50000}
            }
    
    pytorchRoot = CPytorch().Lib
    samplerType = pytorchRoot.utils.data.RandomSampler
    trainDataLoader = oDataLoaderTrans(oDataRecordTrain,CSlidingWinDataset,samplerType,**argsTrain)
    testDataLoader = oDataLoaderTrans(oDataRecordTest,CSlidingWinDataset,samplerType,**argsTest)
#    import sys
#    sys.exit()
    oLossFunc = pytorchRoot.nn.BCELoss()
    
    oCRNN = CCRNNChannels(cnnDir,denseDir,256,100).cuda()
    metrics = CPytorch().trainClassificationModel(oCRNN,trainDataLoader,testDataLoader,10,0.001,0.001,oLossFunc)
    

#oData = CGALEDRawData(500)
#oData.readFile(dataFiles[0])l
#oEvent = CGALEDLabels()
#oEvent.readFile(eventFiles[0])
#
#oData1 = CGALEDRawData(500)
#oData1.readFile(dataFiles[24])
#oEvent1 = CGALEDLabels()
#oEvent1.readFile(eventFiles[24])

#oDataOrg = CDataOrganizorLite(oData.numChannels,oData.sampleRate,
#                              oData.description['chName'],keysFunc)
#oDataOrg.insert(oData1,oEvent1)
#oDataOrg.insert(oData,oEvent)
#oDataRecord = oDataOrg.dataRecordBasedOnTime()
#oDataOrg.getSortedKeys()
#timeId = oData.timestamps
#timeId1 = oData1.timestamps
#oDataSet,eventList = oDataOrg.dataSetBasedOnStimuliDesc('Replace',1)

#oMNE = CIfMNE(oDataOrg.channelList,oDataOrg.srate,'eeg')
#oMNE.Montage = oMNE.LibMNE.channels.read_montage('chanlabels_32channel_test',path = oDir['Root'])
#epochs1 = oMNE.CDataSetToEpochs(oDataSet,eventList,{'Replace':1})

## 200509-2168167 

