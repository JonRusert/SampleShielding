
import textattack
from textattack.models.helpers import LSTMForClassification
from textattack.models.wrappers import PyTorchModelWrapper
import sys
import csv
from math import exp
import nltk
import nltk.data
from random import randint
import json
import numpy as np

from sample import getRandomSamples, getShiftingSamples, getEntireShiftingSamples, getDistinctShiftingSamples, getRandomWordSamples



class SampleShieldModelWrapper(PyTorchModelWrapper):
    '''

    '''

    def __init__(self, model, tokenizer, k = 100, p = 0.3, method = 'randomWordSample'):
        self.model = model
        self.tokenizer = tokenizer
        self.num_labels = model._config["num_labels"]
        self.model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
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



def attackSampleShieldModel(modelLocation, k = 100, percentage = 0.8, outputfile = None, num_examples = 100, offset = 0, method = 'randomWordSample', attack_method = 'textfooler', data = 'imdb'):
    argsfile = open(modelLocation + 'train_args.json')
    args = json.load(argsfile)

    model = LSTMForClassification(max_seq_length = args['max_length'], num_labels = args['num_labels'])
    model.load_from_disk(modelLocation)
    model_wrapper = SampleShieldModelWrapper(model, model.tokenizer, int(k), float(percentage), method)

    if(attack_method == 'textfooler'):
        attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    elif(attack_method == 'bertattack'):
        attack = textattack.attack_recipes.BERTAttackLi2020.build(model_wrapper)
    elif(attack_method == 'textbugger'):
        attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)
    elif(attack_method == 'pwws'):
        attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)

    if(data == 'imdb'):
        dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")
    elif(data == 'yelp'):
        dataset = textattack.datasets.HuggingFaceDataset("yelp_polarity", split="test")
    elif(data == 'ag'):
        dataset = textattack.datasets.HuggingFaceDataset("ag_news", split="test")
    

    if(not outputfile):
        outputfile = "adversarialAttackLSTM.csv"

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
    
    


def testLSTM(modelLocation, testSet):
    argsfile = open(modelLocation + 'train_args.json')
    args = json.load(argsfile)

    model = LSTMForClassification(max_seq_length = args['max_length'])
    model.load_from_disk(modelLocation)
    model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer, batch_size = 1)
    
    testcsv = csv.reader(open(testSet), delimiter = '\t')
    outcsv = csv.writer(open(testSet.split('.')[0] + '_LSTMPredictionsOut', 'w'), delimiter = '\t')

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




def sampleTestLSTM(modelLocation, testSet, k = 100, percentage = 0.8, sampleType = 'randomSample'):
    argsfile = open(modelLocation + 'train_args.json')
    args = json.load(argsfile)

    model = LSTMForClassification(max_seq_length = args['max_length'], num_labels = args['num_labels'])
    model.load_from_disk(modelLocation)
    model = PyTorchModelWrapper(model, model.tokenizer)
    
    num_labels = args['num_labels']
    pos_labels = [float(x) for x in range(num_labels)]


    testcsv = csv.reader(open(testSet), delimiter = '\t')
    str_p = str(percentage).replace('.', '')
    
    if(sampleType == 'randomSample'):
        outcsv = csv.writer(open(testSet.split('.')[0] + '_LSTMSamplePredictionsOut_k' + str(k) + '_p' + str_p, 'w'), delimiter = '\t')
        output_votes = csv.writer(open('votes_' + testSet.split('.')[0] + '_LSTMSamplePredictionsOut_k' + str(k) + '_p' + str_p, 'w'), delimiter = '\t')
    elif(sampleType == 'randomWordSample'):
        outcsv = csv.writer(open(testSet.split('.')[0] + '_LSTMWordSamplePredictionsOut_k' + str(k) + '_p' + str_p, 'w'), delimiter = '\t')
        output_votes = csv.writer(open('votes_' + testSet.split('.')[0] + '_LSTMWordSamplePredictionsOut_k' + str(k) + '_p' + str_p, 'w'), delimiter = '\t')
    elif(sampleType == 'shiftingSample'):
        outcsv = csv.writer(open(testSet.split('.')[0] + '_LSTMShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')
        output_votes = csv.writer(open('votes_' + testSet.split('.')[0] + '_LSTMShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')
    elif(sampleType == 'entireShiftingSample'):
        outcsv = csv.writer(open(testSet.split('.')[0] + '_LSTMEntireShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')
        output_votes = csv.writer(open('votes_' + testSet.split('.')[0] + '_LSTMEntireShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')
    elif(sampleType == 'distinctShiftingSample'):
        outcsv = csv.writer(open(testSet.split('.')[0] + '_LSTMDistinctShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')
        output_votes = csv.writer(open('votes_' + testSet.split('.')[0] + '_LSTMDistinctShiftingSamplePredictionsOut_p' + str_p, 'w'), delimiter = '\t')



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
            
            attacked_text = textattack.shared.AttackedText(cur_sample)
            pred = textattack.shared.utils.batch_model_predict(model, [attacked_text.tokenizer_input])

            #pred = model([cur_sample])
            output = pred[0]
        
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
    testLSTM(sys.argv[1], sys.argv[2])
elif(sys.argv[-1] == 'sampleTest'):
    if(len(sys.argv) == 4):
        sampleTestLSTM(sys.argv[1], sys.argv[2])
    elif(len(sys.argv) == 6):
        sampleTestLSTM(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif(len(sys.argv) == 7):
        sampleTestLSTM(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
elif(sys.argv[-1] == 'attack'):
    if(len(sys.argv) == 3):
        attackSampleShieldModel(sys.argv[1])
    elif(len(sys.argv) == 9):
        attackSampleShieldModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    elif(len(sys.argv) == 10):
        attackSampleShieldModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
    elif(len(sys.argv) == 11):
        attackSampleShieldModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9])

    
    
