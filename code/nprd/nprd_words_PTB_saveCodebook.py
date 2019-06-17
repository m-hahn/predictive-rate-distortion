

# Was called yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE_WordsLR.py.

from paths import LOG_PATH_WORDS
import torchkit.optim
import torchkit.nn, torchkit.flows, torchkit.utils
import numpy as np
import random
import sys

objectiveName = "LM"

language = sys.argv[1]
languageCode = sys.argv[2]
dropout_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.33
emb_dim = int(sys.argv[4]) if len(sys.argv) > 4 else 100
rnn_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 512
rnn_layers = int(sys.argv[6]) if len(sys.argv) > 6 else 2
lr = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
model = sys.argv[8]
input_dropoutRate = float(sys.argv[9]) # 0.33
batchSize = int(sys.argv[10])
replaceWordsProbability = float(sys.argv[11])
horizon = int(sys.argv[12]) if len(sys.argv) > 12 else 11
beta = float(sys.argv[13]) if len(sys.argv) > 13 else 0.01
flow_length = int(sys.argv[14]) if len(sys.argv) > 14 else 5
flowtype = sys.argv[15]
prescripedID = sys.argv[16] if len(sys.argv)> 16 else None

weight_decay=1e-5

if len(sys.argv) == 17:
  del sys.argv[16]
assert len(sys.argv) in [15,16,17]


assert dropout_rate <= 0.5
assert input_dropoutRate <= 0.5

devSurprisalTable = [None] * horizon
if prescripedID is not None:
  myID = int(prescripedID)
else:
  myID = random.randint(0,10000000)






#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]

import math
from math import log, exp
from random import random, shuffle


from corpusIterator_PTB import CorpusIterator_PTB


corporaCached = {}
corporaCached["train"] = CorpusIterator_PTB("PTB","train")
corporaCached["test"] = CorpusIterator_PTB("PTB","test")
corporaCached["dev"] = CorpusIterator_PTB("PTB","dev")


originalDistanceWeights = {}

morphKeyValuePairs = set()

def removeNonWords(leaves):
 return [("(" if x == "-LRB-" else (")" if x == "-RRB-" else x.replace("\/", "/").replace("\*","*"))).lower() for x in leaves if "*-" not in x and (not x.startswith("*")) and x not in ["0", "*U*", "*?*"]]


def initializeOrderTable():
   vocab = {}
   for partition in ["train", "dev", "test"]:
     for sentence in corporaCached[partition].iterator():
      for line in removeNonWords(sentence.leaves()):
          vocab[line] = vocab.get(line, 0) + 1
   return None, vocab, None, None 

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()




def orderSentence(sentence):
   return removeNonWords(sentence.leaves())


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()

words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
word = [x[0] for x in words]
itos_word = word
stoi_word = dict(zip(word, range(len(word))))

print(itos_word)
print(stoi_word)
assert u'the' in stoi_word




import os

originalCounter = "NA"


vocab_size = 10000
vocab_size = min(len(itos_word),vocab_size)


word_pos_morph_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim=emb_dim).cuda()
outVocabSize = 3+vocab_size

itos_total = ["EOS", "EOW", "SOS"] + itos_word[:vocab_size] #+ itos_lemmas[:vocab_size] + itos_morph
assert len(itos_total) == outVocabSize



dropout = nn.Dropout(dropout_rate).cuda()

rnn_past = nn.LSTM(emb_dim, rnn_dim, rnn_layers).cuda()
for name, param in rnn_past.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)
rnn_future = nn.LSTM(emb_dim, rnn_dim, rnn_layers).cuda()
for name, param in rnn_future.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)

decoder = nn.Linear(rnn_dim,outVocabSize).cuda()


components = [rnn_past, rnn_future, decoder, word_pos_morph_embeddings]





hiddenToLogSDHidden = nn.Linear(rnn_dim, rnn_dim).cuda()
cellToMean = nn.Linear(rnn_dim, rnn_dim).cuda()
sampleToHidden = nn.Linear(rnn_dim, rnn_dim).cuda()
sampleToCell = nn.Linear(rnn_dim, rnn_dim).cuda()

hiddenToLogSDHidden.bias.data.fill_(0)
cellToMean.bias.data.fill_(0)
sampleToHidden.bias.data.fill_(0)
sampleToCell.bias.data.fill_(0)

hiddenToLogSDHidden.weight.data.fill_(0)
cellToMean.weight.data.fill_(0)
sampleToHidden.weight.data.fill_(0)
sampleToCell.weight.data.fill_(0)






import torchkit.nn as nn_


################################################
################################################
# The following block is due to Chin-Wei Huang, https://github.com/CW-Huang/NAF/
class BaseFlow(torch.nn.Module):
    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()





class IAF(BaseFlow):
    
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), realify=nn_.sigmoid, fixed_order=False):
        super(IAF, self).__init__()
        self.realify = realify
        
        self.dim = dim
        self.context_dim = context_dim
        
        if type(dim) is int:
            self.mdl = torchkit.iaf_modules.cMADE(
                    dim, hid_dim, context_dim, num_layers, 2, 
                    activation, fixed_order)
            self.reset_parameters()
        else:
           assert False        
        
    def reset_parameters(self):
        self.mdl.hidden_to_output.cscale.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cscale.bias.data.uniform_(0.0, 0.0)
        self.mdl.hidden_to_output.cbias.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cbias.bias.data.uniform_(0.0, 0.0)
        if self.realify == nn_.softplus:
            inv = np.log(np.exp(1-nn_.delta)-1) 
            self.mdl.hidden_to_output.cbias.bias.data[1::2].uniform_(inv,inv)
        elif self.realify == nn_.sigmoid:
            self.mdl.hidden_to_output.cbias.bias.data[1::2].uniform_(2.0,2.0)
        
        
    def forward(self, inputs):
        x, logdet, context = inputs
        if torch.isnan(x).any():
           assert False, x
        if torch.isnan(context).any():
           assert False, context

        out, _ = self.mdl((x, context))
        if torch.isnan(out).any():
           assert False, out
        if isinstance(self.mdl, torchkit.iaf_modules.cMADE):
            mean = out[:,:,0]
            lstd = out[:,:,1]
        else:
            assert False
    
        std = self.realify(lstd)
        
        if self.realify == nn_.softplus:
            x_ = mean + std * x
        elif self.realify == nn_.sigmoid:
            x_ = (-std+1.0) * mean + std * x
        elif self.realify == nn_.sigmoid2:
            x_ = (-std+2.0) * mean + std * x
        logdet_ = nn_.sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context

 



num_ds_dim = 16 #64
num_ds_layers = 1

if flowtype == 'affine':
    flow = IAF
elif flowtype == 'dsf':
    flow = lambda **kwargs:torchkit.flows.IAF_DSF(num_ds_dim=num_ds_dim,
                                         num_ds_layers=num_ds_layers,
                                         **kwargs)
elif flowtype == 'ddsf':
    flow = lambda **kwargs:torchkit.flows.IAF_DDSF(num_ds_dim=num_ds_dim,
                                          num_ds_layers=num_ds_layers,
                                          **kwargs)


########################################################
########################################################




components = components + [hiddenToLogSDHidden, cellToMean, sampleToHidden, sampleToCell]
context_dim = 1
flows = [flow(dim=rnn_dim, hid_dim=512, context_dim=context_dim, num_layers=2, activation=torch.nn.ELU()).cuda() for _ in range(flow_length)]
components = components + flows

def parameters():
 for c in components:
   for param in c.parameters():
      yield param

initrange = 0.1
word_pos_morph_embeddings.weight.data.uniform_(-initrange, initrange)

decoder.bias.data.fill_(0)
decoder.weight.data.uniform_(-initrange, initrange)




crossEntropy = 10.0
optimizer = torch.optim.Adam(parameters(), lr=lr, betas=(0.9, 0.999) , weight_decay=weight_decay)

import torch.cuda
import torch.nn.functional

inputDropout = torch.nn.Dropout2d(p=input_dropoutRate)
counter = 0
lastDevLoss = None
failedDevRuns = 0
devLosses = [] 
devMemories = []

lossModule = nn.NLLLoss()
lossModuleTest = nn.NLLLoss(size_average=False, reduce=False, ignore_index=2)


standardNormal = torch.distributions.Normal(loc=torch.FloatTensor([[0.0 for _ in range(rnn_dim)] for _ in range(batchSize)]).cuda(), scale=torch.FloatTensor([[1.0 for _ in range(rnn_dim)] for _ in range(batchSize)]).cuda())



def doForwardPass(input_indices, wordStartIndices, surprisalTable=None, doDropout=True, batchSizeHere=1):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       if printHere:
           print "wordStartIndices"
           print wordStartIndices

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSizeHere, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSizeHere, 128).zero_()))
       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0

       optimizer.zero_grad()

       for c in components:
          c.zero_grad()
       totalQuality = 0.0

       if True:
           
           sequenceLength = max(map(len, input_indices))
           for i in range(batchSizeHere):
              input_indices[i] = input_indices[i][:]
              while len(input_indices[i]) < sequenceLength:
                 input_indices[i].append(2)

           inputTensor = Variable(torch.LongTensor(input_indices).transpose(0,1).contiguous()).cuda() # so it will be sequence_length x batchSizeHere

           inputTensorIn = inputTensor[:-1]
           inputTensorOut = inputTensor[1:]

           inputEmbeddings = word_pos_morph_embeddings(inputTensorIn.view(sequenceLength-1, batchSizeHere))
           if doDropout:
              inputEmbeddings = inputDropout(inputEmbeddings)
              if dropout_rate > 0:
                 inputEmbeddings = dropout(inputEmbeddings)


           halfSeqLen = int(sequenceLength/2)

           output1, hidden = rnn_past(inputEmbeddings[:halfSeqLen+1], None)

           assert rnn_layers == 1
           meanHidden = cellToMean(hidden[1][0])

           klLoss = [None for _ in inputEmbeddings]
           logStandardDeviationHidden = hiddenToLogSDHidden(hidden[1][0])
           memoryDistribution = torch.distributions.Normal(loc=meanHidden, scale=torch.exp(logStandardDeviationHidden))

           encodedEpsilon = standardNormal.sample()
           sampled = meanHidden + torch.exp(logStandardDeviationHidden) * encodedEpsilon

           logProbConditional = memoryDistribution.log_prob(sampled).sum(dim=1)  # TODO not clear whether back-prob through sampled?

           if True:
              adjustment = []
              epsilon = sampled
              logdet = torch.autograd.Variable(torch.from_numpy(np.zeros(batchSize).astype('float32')).cuda())
              context = torch.autograd.Variable(torch.from_numpy(np.zeros((batchSize,context_dim)).astype('float32')).cuda())
              for flowStep in range( flow_length):
                epsilon, logdet, context = flows[flowStep]((epsilon, logdet, context))
                if flowStep +1 < flow_length:
                   epsilon, logdet, context = torchkit.flows.FlipFlow(1)((epsilon, logdet, context))

              plainPriorLogProb = standardNormal.log_prob(epsilon).sum(dim=1) #- (0.5 * torch.sum(sampled * sampled, dim=1))
              logProbMarginal = plainPriorLogProb + logdet


           klLoss = logProbConditional - logProbMarginal
           hiddenNew = sampleToHidden(sampled).unsqueeze(0)
           cellNew = sampleToCell(sampled).unsqueeze(0)
           output, _ = rnn_future(torch.cat([word_pos_morph_embeddings(torch.cuda.LongTensor([[2 for _ in range(batchSizeHere)]])), inputEmbeddings[halfSeqLen+1:]], dim=0), (hiddenNew, cellNew))
           output = torch.cat([output1[:halfSeqLen], output], dim=0)
           if doDropout:
              output = dropout(output)
           word_logits = decoder(output)
           word_logits = word_logits.view((sequenceLength-1)*batchSizeHere, outVocabSize)
           word_softmax = logsoftmax(word_logits)
           lossesWord = lossModuleTest(word_softmax, inputTensorOut.view((sequenceLength-1)*batchSizeHere))
           lossWords = lossesWord.sum(dim=0).sum(dim=0)
           loss = lossesWord.sum()

           klLossSum = klLoss.sum()
           if counter % 10 == 0:
              print(beta, flow_length, klLoss.mean(), lossesWord.mean(), beta * klLoss.mean() + lossesWord.mean() )
           loss = loss + beta * klLossSum

           if surprisalTable is not None or printHere:           
             lossesCPU = lossesWord.data.cpu().view((sequenceLength-1), batchSizeHere).numpy()
             if printHere:
                for i in range(0,len(input_indices[0])-1): #range(1,maxLength+1): # don't include i==0
                         j = 0
                         print (i, itos_total[input_indices[j][i+1]], lossesCPU[i][j])

             if surprisalTable is not None: 
                if printHere:
                   print surprisalTable
                for j in range(batchSizeHere):
                  for r in range(horizon):
                    assert wordStartIndices[j][r]< wordStartIndices[j][r+1]
                    assert wordStartIndices[j][r] < len(lossesWord)+1, (wordStartIndices[j][r],wordStartIndices[j][r+1], len(lossesWord))
                    assert input_indices[j][wordStartIndices[j][r+1]-1] != 2
                    if r == horizon-1:
                      assert wordStartIndices[j][r+1] == len(input_indices[j]) or input_indices[j][wordStartIndices[j][r+1]] == 2
                    surprisalTable[r] += sum(lossesCPU[wordStartIndices[j][r]-1:wordStartIndices[j][r+1]-1,j]) #.data.cpu().numpy()[0]

           wordNum = (len(wordStartIndices[0]) - 1)*batchSizeHere
           assert len(wordStartIndices[0]) == horizon+1, map(len, wordStartIndices)
                    
       if wordNum == 0:
         print input_words
         print batchOrdered
         return 0,0,0,0,0
       if printHere:
         print loss/wordNum
         print lossWords/wordNum
         print ["CROSS ENTROPY", crossEntropy, exp(crossEntropy)]
         print ("beta", beta)
       crossEntropy = 0.99 * crossEntropy + 0.01 * (lossWords/wordNum).data.cpu().numpy()
       totalQuality = loss.data.cpu().numpy() # consists of lossesWord + lossesPOS
       numberOfWords = wordNum
       return loss, None, None, totalQuality, numberOfWords, klLoss.mean()


parameterList = list(parameters())

def  doBackwardPass(loss, baselineLoss, policy_related_loss):
       global lastDevLoss
       global failedDevRuns
       loss.backward()
       if printHere:
         print "BACKWARD 3 "+__file__+" "+language+" "+str(myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)+"  "+(" ".join(map(str,["Dropout (real)", dropout_rate, "Emb_dim", emb_dim, "rnn_dim", rnn_dim, "rnn_layers", rnn_layers, "MODEL", model])))
         print devLosses
         print lastDevLoss
       torch.nn.utils.clip_grad_norm(parameterList, 5.0, norm_type='inf')
       optimizer.step()
       for param in parameters():
         if param.grad is None:
           print "WARNING: None gradient"



def createStream(corpus):
    global crossEntropy
    global printHere
    global devLosses

    input_indices = [2] # Start of Segment
    wordStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       ordered = orderSentence(sentence)
       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
            target_word = stoi_word[line]
            if target_word >= vocab_size:
               target_word = vocab_size-1
            input_indices.append(target_word+3)
          if len(wordStartIndices) == horizon:
             yield input_indices, wordStartIndices+[len(input_indices)]
             input_indices = [2] # Start of Segment (makes sure that first word can be predicted from this token)
             wordStartIndices = []



def createStreamContinuous(corpus):
    global crossEntropy
    global printHere
    global devLosses
    input_indices = [2] # Start of Segment
    wordStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       if sentCount % 10 == 0:
         print ["DEV SENTENCES", sentCount]
       ordered = orderSentence(sentence)
       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
            target_word = stoi_word[line]
            if target_word >= vocab_size:
               target_word = vocab_size-1
            input_indices.append(target_word+3)
          if len(wordStartIndices) == horizon:
             yield input_indices, wordStartIndices+[len(input_indices)]
             input_indices = [2]+input_indices[wordStartIndices[1]:] # Start of Segment (makes sure that first word can be predicted from this token)
             wordStartIndices = [x-wordStartIndices[1]+1 for x in wordStartIndices[1:]]
             assert wordStartIndices[0] == 1




def computeDevLoss(test=False):
   global printHere
   global horizon
   devLoss = 0.0
   devWords = 0
   corpusDev = corporaCached["test" if test else "dev"].iterator()
   stream = createStreamContinuous(corpusDev) if test else createStream(corpusDev)

   surprisalTable = [0 for _ in range(horizon)]
   devCounter = 0
   devMemory = 0
   while True:
     try:
        input_indices_list = []
        wordStartIndices_list = []
        for _ in range(batchSize):
           input_indices, wordStartIndices = next(stream)
           input_indices_list.append(input_indices)
           wordStartIndices_list.append(wordStartIndices)
     except StopIteration:
        break
     devCounter += 1
     printHere = (devCounter % 100 == 0)
     _, _, _, newLoss, newWords, devMemoryHere = doForwardPass(input_indices_list, wordStartIndices_list, surprisalTable = surprisalTable, doDropout=False, batchSizeHere=batchSize)
     devMemory += devMemoryHere.data.cpu().numpy()
     devLoss += newLoss
     devWords += newWords
     if printHere:
         print "Dev examples "+str(devCounter)
   devSurprisalTableHere = [surp/(devCounter*batchSize) for surp in surprisalTable]
   return devLoss/devWords, devSurprisalTableHere, devMemory/devCounter

import time
startTime = time.time()

epochCount = 0
corpusBase = corporaCached["train"]
while failedDevRuns == 0:
  epochCount += 1
  print "Starting new epoch, permuting corpus"
  corpusBase.permute()
  corpus = corpusBase.iterator()
  stream = createStream(corpus)


  if counter > 5:
       if True: #counter % DEV_PERIOD == 0:
          newDevLoss, devSurprisalTableHere, newDevMemory = computeDevLoss()
          devLosses.append(newDevLoss)
          devMemories.append(newDevMemory)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)

          if lastDevLoss is None or newDevLoss < lastDevLoss:
              devSurprisalTable = devSurprisalTableHere
          print(devSurprisalTable[horizon/2])
          print(devMemories)
          with open(LOG_PATH_WORDS+"/estimates-"+language+"_"+__file__+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
              print >> outFile, " ".join(sys.argv)
              print >> outFile, " ".join(map(str,devLosses))
              print >> outFile, " ".join(map(str,devSurprisalTable))
              print >> outFile, " ".join(map(str, devMemories))
              print >> outFile, str(sum([x-y for x, y in zip(devSurprisalTable[:horizon/2], devSurprisalTable[horizon/2:])]))
              print >> outFile, str(time.time()-startTime)



          if lastDevLoss is None or newDevLoss < lastDevLoss:
             lastDevLoss = newDevLoss
             failedDevRuns = 0
          else:
             failedDevRuns += 1
             print "Skip saving, hoping for better model"
             print devLosses
             print "Epoch "+str(epochCount)+" "+str(counter)
             print zip(range(1,horizon+1), devSurprisalTable)
             print "MI(Bottleneck, Future) "+str(sum([x-y for x, y in zip(devSurprisalTable[0:horizon/2], devSurprisalTable[horizon/2:])]))
             print "Memories "+str(devMemories)
             #break



  while True:
       counter += 1
       printHere = (counter % 50 == 0)

       try:
          input_indices_list = []
          wordStartIndices_list = []
          for _ in range(batchSize):
             input_indices, wordStartIndices = next(stream)
             input_indices_list.append(input_indices)
             wordStartIndices_list.append(wordStartIndices)
       except StopIteration:
          break
       loss, baselineLoss, policy_related_loss, _, wordNumInPass, _ = doForwardPass(input_indices_list, wordStartIndices_list, batchSizeHere=batchSize)
       if wordNumInPass > 0:
         doBackwardPass(loss, baselineLoss, policy_related_loss)
       else:
         print "No words, skipped backward"
       if printHere:
          print "Epoch "+str(epochCount)+" "+str(counter)
          print zip(range(1,horizon+1), devSurprisalTable)
          if devSurprisalTable[0] is not None:
             print "MI(Bottleneck, Future) "+str(sum([x-y for x, y in zip(devSurprisalTable[0:horizon/2], devSurprisalTable[horizon/2:])]))
             print "Memories "+str(devMemories)







newDevLoss, devSurprisalTableHere, newDevMemory = computeDevLoss(test=True)
devLosses.append(newDevLoss)
devMemories.append(newDevMemory)
print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)

devSurprisalTable = devSurprisalTableHere


print(devSurprisalTable[horizon/2])
print(devMemories)
with open(LOG_PATH_WORDS+"/test-estimates-"+language+"_"+__file__+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
    print >> outFile, " ".join(sys.argv)
    print >> outFile, " ".join(map(str,devLosses))
    print >> outFile, " ".join(map(str,devSurprisalTable))
    print >> outFile, " ".join(map(str, devMemories))
    print >> outFile, str(sum([x-y for x, y in zip(devSurprisalTable[:horizon/2], devSurprisalTable[horizon/2:])]))
    print >> outFile, str(time.time()-startTime)


print(devSurprisalTable[int(horizon/2)])
print(devMemories)

state = {"arguments" : sys.argv, "words" : itos_word, "components" : [c.state_dict() for c in components]}
torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+language+"_"+__file__+"_code_"+str(myID)+".txt")


