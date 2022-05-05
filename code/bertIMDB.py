# wrapper for using trained bert with textattack
import sys
import csv
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AutoTokenizer, AutoModelForSequenceClassification
from random import randrange, randint
import nltk
import nltk.data
import textattack
from math import exp
from sample import getRandomSamples, getShiftingSamples, getEntireShiftingSamples, getDistinctShiftingSamples, getRandomWordSamples





def testModel(testFile, pred_file):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 512
    
    model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-imdb')
    tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-imdb')
 
    
    model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer, batch_size = 1)


    testcsv = csv.reader(open(testFile), delimiter = '\t')
    outcsv = csv.writer(open(pred_file, 'w'), delimiter = '\t')


    for cur in testcsv:
        attacked_text = textattack.shared.AttackedText(cur[1])
        pred = textattack.shared.utils.batch_model_predict(model, [attacked_text.tokenizer_input])

        #pred = model([cur[1]])
        output = pred[0]
        

        smbot = exp(output[0]) + exp(output[1])
        prob0 = exp(output[0])/smbot
        prob1 = exp(output[1])/smbot
        
        if(prob0 > prob1):
            pred = 0.0
        else:
            pred = 1.0

        outcsv.writerow([cur[0], pred, prob0, prob1])






def sampleTestModel(testFile, pred_file, k = 100, percentage = 0.8, sampleType = 'randomSample'):
    k = int(k)
    percentage = float(percentage)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 512
    
    model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-imdb')
    tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-imdb')
 
    
    model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer, batch_size = 1)

    
    test_csv = csv.reader(open(testFile, 'r'), delimiter = '\t')
    pred_csv = csv.writer(open(pred_file, 'w'), delimiter = '\t')
    output_votes = csv.writer(open('votes_' + pred_file, 'w'), delimiter = '\t')


    for line in test_csv:
        text = line[1]
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

        # get predictions for each sample, take majority vote of samples
        for cur_sample in text_samples:
            pred, prob = testSingle(model, tokenizer, cur_sample, MAX_LEN)

            if(pred not in votes):
                votes[pred] = 0

            votes[pred] += 1
            probs.append(prob)      
            

        if(0.0 not in votes):
            votes[0] = 0
        if(1.0 not in votes):
            votes[1] = 0

        
        out_id = line[0]
        outrow = [out_id, votes[0], votes[1]]
        outrow.extend(probs)
        output_votes.writerow(outrow)

        # majority vote
        final_pred = sorted(votes, key = votes.get, reverse = True)[0]

        outPred = [out_id, final_pred]
        pred_csv.writerow(outPred)



def testSingle(model, tokenizer, text, MAX_LEN = 512):

    attacked_text = textattack.shared.AttackedText(text)
    pred = textattack.shared.utils.batch_model_predict(model, [attacked_text.tokenizer_input])
    
    #pred = model([cur[1]])
    output = pred[0]
    
    
    smbot = exp(output[0]) + exp(output[1])
    prob0 = exp(output[0])/smbot
    prob1 = exp(output[1])/smbot
    
    if(prob0 > prob1):
        pred = 0.0
    else:
        pred = 1.0

    return pred, prob0






if(sys.argv[-1] == 'test'):
    testModel(sys.argv[1], sys.argv[2])
elif(sys.argv[-1] == 'sampleTest'):
    if(len(sys.argv) == 4):
        sampleTestModel(sys.argv[1], sys.argv[2])
    elif(len(sys.argv) == 6):
        sampleTestModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif(len(sys.argv) == 7):
        sampleTestModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
elif(sys.argv[-1] == 'attack'):
    if(len(sys.argv) == 2):
        attackSampleShieldModel(sys.argv[1])
    elif(len(sys.argv) == 7):
        attackSampleShieldModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
