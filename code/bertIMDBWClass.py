# wrapper for using trained bert with textattack
import sys
import csv
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AutoTokenizer, AutoModelForSequenceClassification
from textattack.models.wrappers import HuggingFaceModelWrapper
from random import randrange, randint
import nltk
import nltk.data
import textattack
from math import exp
from sample import getRandomSamples, getShiftingSamples, getEntireShiftingSamples, getDistinctShiftingSamples, getRandomWordSamples
import numpy as np



class SampleShieldModelWrapper(HuggingFaceModelWrapper):
    '''

    '''

    def __init__(self, model, tokenizer, k = 100, p = 0.3, method = 'randomWordSample'):
        self.model = model
        self.tokenizer = tokenizer
        self.num_labels = 2
        self.model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
        self.k = k
        self.p = p
        self.method = method



        
    def __call__(self, text_input_list, batch_size=32):
        outputs = []

        
        pos_preds = [float(x) for x in range(self.num_labels)]

        for text in text_input_list:
            if(self.method == 'randomSample'):
                text_samples = getRandomSamples(text, self.k, self.p)
            elif(self.method == 'randomWordSample'):
                text_samples = getRandomWordSamples(text, self.k, self.p)
            elif(self.method == 'shiftingSample'):
                text_samples = getShiftingSamples(text, self.p)
            elif(self.method == 'entireShiftingSample'):
                text_samples = getEntireShiftingSamples(text, self.p)
            elif(self.method == 'distinctShiftingSample'):
                text_samples = getDistinctShiftingSamples(text, self.p)
        
            
            #print(text)
            #print(text_samples)
            votes = {x:0 for x in pos_preds}

            for cur_sample in text_samples:            
                attacked_text = textattack.shared.AttackedText(cur_sample)
                pred = textattack.shared.utils.batch_model_predict(self.model_wrapper, [attacked_text.tokenizer_input])

                #print(pred)

                output = pred[0]
                

                pred = float(output.argmax())

                votes[pred] += 1

            

            #final_pred = sorted(votes, key = votes.get, reverse = True)[0]
            batch_preds = []
            for x in sorted(votes):
                batch_preds.append(votes[x]/self.k)
                
            batch_preds = np.array(batch_preds)
            outputs.append(batch_preds)
            #print(batch_preds)
            #print('concat:', np.concatenate(outputs, axis=0))
            
        #print('outputs:', outputs)
        #return np.concatenate(outputs, axis=0)
        return outputs



def attackSampleShieldModel(k = 100, percentage = 0.8, outputfile = None, num_examples = 100, offset = 0, method = 'randomWordSample', attack_method = 'textfooler'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 512
    
    model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-imdb')
    tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-imdb')
     
    model_wrapper = SampleShieldModelWrapper(model, tokenizer, int(k), float(percentage), method)

    
    if(attack_method == 'textfooler'):
        attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    elif(attack_method == 'bertattack'):
        attack = textattack.attack_recipes.BERTAttackLi2020.build(model_wrapper)
    elif(attack_method == 'textbugger'):
        attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)
    elif(attack_method == 'pwws'):
        attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
    

    dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")
    

    if(not outputfile):
        outputfile = "adversarialAttackBERT.csv"

    attack_args = textattack.AttackArgs(
        num_examples= int(num_examples),
        num_examples_offset = int(offset),
        log_to_csv= outputfile,
        csv_coloring_style = 'plain',
        checkpoint_interval=1,
        checkpoint_dir="checkpoints",
        disable_stdout=True)

    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()



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
    elif(len(sys.argv) == 8):
        attackSampleShieldModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    elif(len(sys.argv) == 9):
        attackSampleShieldModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
