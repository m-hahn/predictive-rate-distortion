# Runs NPRD on analytically known processes.

# Was called yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE_TOY.py.

from paths import LOG_PATH
import torchkit.optim
import torchkit.nn, torchkit.flows, torchkit.utils
import numpy as np
import random
import sys

language = sys.argv[1]
languageCode = sys.argv[2]
dropout_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.33
emb_dim = int(sys.argv[4]) if len(sys.argv) > 4 else 100
rnn_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 512
rnn_layers = int(sys.argv[6]) if len(sys.argv) > 6 else 2
lr = 0.0001
model = sys.argv[8]
input_dropoutRate = float(sys.argv[9]) # 0.33
batchSize = int(sys.argv[10])
horizon = int(sys.argv[11]) if len(sys.argv) > 11 else 10
beta = float(sys.argv[12]) if len(sys.argv) > 12 else 0.01
flow_length = int(sys.argv[13]) if len(sys.argv) > 13 else 5
flowtype = sys.argv[14] if len(sys.argv) > 14 else "ddsf"
flow_hid_dim = int(sys.argv[15]) if len(sys.argv) > 15 else 512
flow_num_layers = int(sys.argv[16]) if len(sys.argv) > 16 else 2
prescripedID = sys.argv[17] if len(sys.argv)> 17 else None

weight_decay=1e-5

if len(sys.argv) == 16:
  del sys.argv[15]
assert len(sys.argv) in [17,18]


assert dropout_rate <= 0.5
assert input_dropoutRate <= 0.5

devSurprisalTable = [None] * horizon
if prescripedID is not None:
  myID = int(prescripedID)
else:
  myID = random.randint(0,10000000)





posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 



import math
from math import log, exp
from random import random, shuffle
from corpusIteratorToy import CorpusIteratorToy




def initializeOrderTable():
   for partition in ["train", "dev"]:
     for sentence in CorpusIteratorToy(language,partition, storeMorph=True).iterator():
      for line in sentence:
          posUni.add(line["posUni"])

import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()

initializeOrderTable()



posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))



import os

originalCounter = "NA"

vocab_size = 0
input_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+0+3, embedding_dim=emb_dim).cuda()
print posUni
print "VOCABULARY "+str(len(posUni)+0+3)
outVocabSize = 3+len(posUni) #+0+len(morphKeyValuePairs)+3
assert len(posUni)+0+3 < 200
itos_total = ["EOS", "EOW", "SOS"] + itos_pos_uni #+ itos_lemmas[:0] + itos_morph
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


components = [rnn_past, rnn_future, decoder, input_embeddings]





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
# flow_hid_dim used to be 512
# flow_num_layers used to be 2
flows = [flow(dim=rnn_dim, hid_dim=flow_hid_dim, context_dim=context_dim, num_layers=flow_num_layers, activation=torch.nn.ELU()).cuda() for _ in range(flow_length)]


components = components + flows




def parameters():
 for c in components:
   for param in c.parameters():
      yield param



initrange = 0.1
input_embeddings.weight.data.uniform_(-initrange, initrange)

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

           inputEmbeddings = input_embeddings(inputTensorIn.view(sequenceLength-1, batchSizeHere))
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
           output, _ = rnn_future(torch.cat([input_embeddings(torch.cuda.LongTensor([[2 for _ in range(batchSizeHere)]])), inputEmbeddings[halfSeqLen+1:]], dim=0), (hiddenNew, cellNew))
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
       ordered = sentence

       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
            input_indices.append(stoi_pos_uni[line["posUni"]]+3)
          if len(wordStartIndices) == horizon:
             yield input_indices, wordStartIndices+[len(input_indices)]
             input_indices = [2] # Start of Segment (makes sure that first word can be predicted from this token)
             wordStartIndices = []



def createStreamContinuous(corpus):
#    global counter
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
       ordered = sentence
       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
            input_indices.append(stoi_pos_uni[line["posUni"]]+3)
          if len(wordStartIndices) == horizon:
             yield input_indices, wordStartIndices+[len(input_indices)]
             input_indices = [2]+input_indices[wordStartIndices[1]:] # Start of Segment (makes sure that first word can be predicted from this token)
             wordStartIndices = [x-wordStartIndices[1]+1 for x in wordStartIndices[1:]]
             assert wordStartIndices[0] == 1




def computeDevLoss():
   global printHere
#   global counter
#   global devSurprisalTable
   global horizon
   devLoss = 0.0
   devWords = 0
#   corpusDev = getNextSentence("dev")
   corpusDev = CorpusIteratorToy(language,"dev", storeMorph=True).iterator(rejectShortSentences = False)
   stream = createStreamContinuous(corpusDev)

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
#     counter += 1
     printHere = (devCounter % 50 == 0)
     _, _, _, newLoss, newWords, devMemoryHere = doForwardPass(input_indices_list, wordStartIndices_list, surprisalTable = surprisalTable, doDropout=False, batchSizeHere=batchSize)
     devMemory += devMemoryHere.data.cpu().numpy()
     devLoss += newLoss
     devWords += newWords
     if printHere:
         print "Dev examples "+str(devCounter)
   devSurprisalTableHere = [surp/(devCounter*batchSize) for surp in surprisalTable]
   return devLoss/devWords, devSurprisalTableHere, devMemory/devCounter

DEV_PERIOD = 5000
epochCount = 0
while failedDevRuns < 10 and len(devLosses) < 20:
  epochCount += 1
  print "Starting new epoch, permuting corpus"
  corpusBase = CorpusIteratorToy(language, storeMorph=True)
#  corpusBase.permute()
#  corpus = getNextSentence("train")
  corpus = corpusBase.iterator(rejectShortSentences = False)
  stream = createStream(corpus)

  while True:
       counter += 1
       printHere = (counter % 50 == 0)

       if counter % DEV_PERIOD == 0:
          newDevLoss, devSurprisalTableHere, newDevMemory = computeDevLoss()
          devLosses.append(newDevLoss)
          devMemories.append(newDevMemory)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)

          if lastDevLoss is None or newDevLoss < lastDevLoss:
              devSurprisalTable = devSurprisalTableHere
          


          print(devSurprisalTable[horizon/2])
          print(devMemories)
          with open(LOG_PATH+"/estimates-"+language+"_"+__file__+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
              print >> outFile, " ".join(sys.argv)
              print >> outFile, " ".join(map(str,devLosses))
              print >> outFile, " ".join(map(str,devSurprisalTable))
              print >> outFile, " ".join(map(str, devMemories))
              print >> outFile, str(sum([x-y for x, y in zip(devSurprisalTable[:horizon/2], devSurprisalTable[horizon/2:])]))




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




print(devSurprisalTable[int(horizon/2)])
print(devMemories)





