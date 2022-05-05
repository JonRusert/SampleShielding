
import nltk
import nltk.data
from random import randint
from random import choice


#takes in text and returns 100 sample containing percentage amount of the original, USES SENTENCES AS SAMPLES
def getRandomSamples(text, k=100, percentage = 0.8): #default to 80%
    samples = []
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = tokenizer.tokenize(text)
    
    sample_size = round(percentage*len(sentences))
    
    if(sample_size == 0):
        print("ERROR: NOT ENOUGH SENTENCES IN REGARD TO PERCENTAGE CHOSEN, DEFAULTING TO ALL SENTENCES")
        sample_size = len(sentences)

    for i in range(k):
        sent_copy = [x for x in range(len(sentences))]

        sample_indices = []

        for j in range(sample_size):
            rand = choice(sent_copy)
            sample_indices.append(rand)
            sent_copy.pop(sent_copy.index(rand))
            
        final_sample = []
        # want to preserve order for sample
        for j in range(len(sentences)):
            if(j in sample_indices):
                final_sample.append(sentences[j])
        print(' '.join(final_sample))
        samples.append(' '.join(final_sample))
        

    return samples


#takes in text and returns 100 sample containing percentage amount of the original, USES WORDS AS SAMPLES
def getRandomWordSamples(text, k=100, percentage = 0.8): #default to 80%
    samples = []
    

    words = text.split(' ')
    
    sample_size = round(percentage*len(words))
    
    if(sample_size == 0):
        print("ERROR: NOT ENOUGH WORDS IN REGARD TO PERCENTAGE CHOSEN, DEFAULTING TO ALL WORDS")
        sample_size = len(words)

    for i in range(k):
        sent_copy = [x for x in range(len(words))]
        sample_indices = []

        for j in range(sample_size):
            rand =choice(sent_copy)
            sample_indices.append(rand)
            sent_copy.pop(sent_copy.index(rand))
            
        final_sample = []
        # want to preserve order for sample
        for j in range(len(words)):
            if(j in sample_indices):
                final_sample.append(words[j])
        print(' '.join(final_sample))
        samples.append(' '.join(final_sample))
        

    return samples



#takes in text and returns 100 sample containing percentage amount of the original
def getShiftingSamples(text, percentage = 0.8): #default to 80%
    samples = []
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = tokenizer.tokenize(text)
    
    sample_size = round(percentage*len(sentences))
    
    if(sample_size == 0 or sample_size == 1):
        print("ERROR: NOT ENOUGH SENTENCES IN REGARD TO PERCENTAGE CHOSEN, DEFAULTING TO ALL SENTENCES")
        sample_size = len(sentences)
        samples = [text]
        return samples

    for i in range(len(sentences) - sample_size):
        
        final_sample = []
        
        for j in range(i, i + sample_size):
            final_sample.append(sentences[j])
            
        print(' '.join(final_sample))
        samples.append(' '.join(final_sample))
        

    return samples


#takes in text
def getEntireShiftingSamples(text, percentage = 0.2): #default to 20%
    samples = []
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = tokenizer.tokenize(text)
    
    sample_size = round(percentage*len(sentences))
    
    if(sample_size == 0 or sample_size == 1):
        print("ERROR: NOT ENOUGH SENTENCES IN REGARD TO PERCENTAGE CHOSEN, DEFAULTING TO ALL SENTENCES")
        sample_size = len(sentences)
        samples = [text]
        return samples

    for i in range(len(sentences)):
        
        sample_indices = []

        for j in range(i, i + sample_size):
            loc = j%len(sentences)

            sample_indices.append(loc)

        final_sample = []
        
        # want to preserve order for sample
        for j in range(len(sentences)):
            if(j in sample_indices):
                final_sample.append(sentences[j])


        print(' '.join(final_sample))
        samples.append(' '.join(final_sample))
        
        

    return samples







#takes in text
def getDistinctShiftingSamples(text, percentage = 0.2): #default to 20%
    samples = []
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = tokenizer.tokenize(text)
    
    sample_size = round(percentage*len(sentences))
    
    if(sample_size == 0 or sample_size == 1):
        print("ERROR: NOT ENOUGH SENTENCES IN REGARD TO PERCENTAGE CHOSEN, DEFAULTING TO ALL SENTENCES")
        sample_size = len(sentences)
        samples = [text]
        return samples

    for i in range(0, len(sentences), sample_size):
        
        final_sample = sentences[i:i+sample_size]
        
        print(' '.join(final_sample))

        samples.append(' '.join(final_sample))
        
        

    return samples
