

import textattack
from textattack.models.helpers import WordCNNForClassification
import sys
import csv
from math import exp
import nltk
import nltk.data
from random import randint
import json

from sample import getRandomSamples, getShiftingSamples, getEntireShiftingSamples, getDistinctShiftingSamples, getRandomWordSamples


def testCNN(modelLocation, testSet):
    argsfile = open(modelLocation + 'train_args.json')
    args = json.load(argsfile)

    model = WordCNNForClassification(max_seq_length = args['max_length'])
    model.load_from_disk(modelLocation)
    model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)
    
    testcsv = csv.reader(open(testSet), delimiter = '\t')
    outcsv = csv.writer(open(testSet.split('.')[0] + '_CNNPredictionsOut', 'w'), delimiter = '\t')

    for cur in testcsv:
        output = model([cur[1]])
        output = output[0]
        
        smbot = exp(output[0]) + exp(output[1])
        prob0 = exp(output[0])/smbot
        prob1 = exp(output[1])/smbot
        
        if(prob0 > prob1):
            pred = 0.0
        else:
            pred = 1.0

        outcsv.writerow([cur[0], pred, prob0, prob1])





def sampleTestCNN(modelLocation, testSet, k = 100, percentage = 0.8, sampleType = 'randomSample'):
    argsfile = open(modelLocation + 'train_args.json')
    args = json.load(argsfile)

    model = WordCNNForClassification(max_seq_length = args['max_length'], num_labels = args['num_labels'])
    model.load_from_disk(modelLocation)
    model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)


    num_labels = args['num_labels']
    pos_labels = [float(x) for x in range(num_labels)]


    testcsv = csv.reader(open(testSet), delimiter = '\t')
    str_p = str(percentage).replace('.', '')
    if(sampleType == 'randomSample'):
        outcsv = csv.writer(open(testSet.split('.')[0] + '_CNNSamplePredictionsOut_k' + str(k) + '_p' + str_p, 'w'), delimiter = '\t')
        output_votes = csv.writer(open('votes_' + testSet.split('.')[0] + '_CNNSamplePredictionsOut_k' + str(k) + '_p' + str_p, 'w'), delimiter = '\t')
    elif(sampleType == 'randomWordSample'):
        outcsv = csv.writer(open(testSet.split('.')[0] + '_CNNWordSamplePredictionsOut_k' + str(k) + '_p' + str_p, 'w'), delimiter = '\t')
        output_votes = csv.writer(open('votes_' + testSet.split('.')[0] + '_CNNWordSamplePredictionsOut_k' + str(k) + '_p' + str_p, 'w'), delimiter = '\t')
    elif(sampleType == 'shiftingSample'):
        outcsv = csv.writer(open(testSet.split('.')[0] + '_CNNShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')
        output_votes = csv.writer(open('votes_' + testSet.split('.')[0] + '_CNNShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')
    elif(sampleType == 'entireShiftingSample'):
        outcsv = csv.writer(open(testSet.split('.')[0] + '_CNNEntireShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')
        output_votes = csv.writer(open('votes_' + testSet.split('.')[0] + '_CNNEntireShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')
    elif(sampleType == 'distinctShiftingSample'):
        outcsv = csv.writer(open(testSet.split('.')[0] + '_CNNDistinctShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')
        output_votes = csv.writer(open('votes_' + testSet.split('.')[0] + '_CNNDistinctShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')



    k = int(k)
    percentage = float(percentage)


    for cur in testcsv:
        
        text = cur[1]
        
        if(sampleType == 'randomSample'):
            text_samples = getRandomSamples(text, k, percentage)
        elif(sampleType == 'randomWordSample'):
            text_samples = getRandomWordSamples(text, k, percentage)
        elif(sampleType == 'shiftingSample'):
            text_samples = getShiftingSamples(text, percentage)
        elif(sampleType == 'entireShiftingSample'):
            text_samples = getEntireShiftingSamples(text, percentage)
        elif(sampleType == 'distinctShiftingSample'):
            text_samples = getDistinctShiftingSamples(text, percentage)
        

        votes = {}
        probs = []
        for cur_sample in text_samples:
            
            output = model([cur_sample])
            output = output[0]
        
            probs.append(output[0])
            

            pred = float(output.argmax())
            #if(output[0] > output[1]):
            #    pred = 0.0
            #else:
            #    pred = 1.0
                
            if(pred not in votes):
                votes[pred] = 0

            votes[pred] += 1


        for x in pos_labels:
            if(x not in votes):
                votes[x] = 0
        
        
        out_id = cur[0]
        outrow = [out_id]
        for x in range(len(votes)):
            outrow.append(votes[x])


        outrow.extend(probs)
        output_votes.writerow(outrow)

        # majority vote
        final_pred = sorted(votes, key = votes.get, reverse = True)[0]



        outcsv.writerow([cur[0], final_pred])





if(sys.argv[-1] == 'test'):
    testCNN(sys.argv[1], sys.argv[2])
elif(sys.argv[-1] == 'sampleTest'):
    if(len(sys.argv) == 4):
        sampleTestCNN(sys.argv[1], sys.argv[2])
    elif(len(sys.argv) == 6):
        sampleTestCNN(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif(len(sys.argv) == 7):
        sampleTestCNN(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    


#testCNN(sys.argv[1], sys.argv[2]) 
        
    

    
    
