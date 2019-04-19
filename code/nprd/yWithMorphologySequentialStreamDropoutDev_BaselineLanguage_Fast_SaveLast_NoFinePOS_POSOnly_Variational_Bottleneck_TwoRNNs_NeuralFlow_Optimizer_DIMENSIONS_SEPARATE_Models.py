#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"


# Good results without dropout:
# ./python27 yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE.py English ru 0.0 300 512 1 0.001 REAL 0.0 64 30 0.02 1 ddsf


#./python27 yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE.py English ru 0.0 300 32 1 0.001 REAL_REAL 0.0 100 10 0.02 5 ddsf


import torchkit.optim
import torchkit.nn, torchkit.flows, torchkit.utils


import numpy as np

#import torchkit.transforms.from_numpy
#import torchkit.transforms.binarize


# TODO also try other optimizers

import random
import sys

objectiveName = "LM"

language = sys.argv[1]
languageCode = sys.argv[2]
dropout_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.33
emb_dim = int(sys.argv[4]) if len(sys.argv) > 4 else 100
rnn_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 512
rnn_layers = int(sys.argv[6]) if len(sys.argv) > 6 else 2

# NOTE lr ends up being ignored
lr_lm = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
lr = 0.0001

model = sys.argv[8]

assert model in ["REAL_REAL", "RANDOM_BY_TYPE", "GROUND"]

input_dropoutRate = float(sys.argv[9]) # 0.33
batchSize = int(sys.argv[10])
horizon = int(sys.argv[11]) if len(sys.argv) > 11 else 10
beta = float(sys.argv[12]) if len(sys.argv) > 12 else 0.01
flow_length = int(sys.argv[13]) if len(sys.argv) > 13 else 5
flowtype = sys.argv[14]
prescripedID = sys.argv[15] if len(sys.argv)> 15 else None

weight_decay=1e-5

if len(sys.argv) == 16:
  del sys.argv[15]
assert len(sys.argv) in [14,15,16]


assert dropout_rate <= 0.5
assert input_dropoutRate <= 0.5

devSurprisalTable = [None] * horizon
if prescripedID is not None:
  myID = int(prescripedID)
else:
  myID = random.randint(0,10000000)


print("TESTING, NO LOGGING")
#with open("/u/scr/mhahn/deps/LOG"+language+"_"+__file__+"_model_"+str(myID)+".txt", "w") as outFile:
#    print >> outFile, " ".join(sys.argv)



posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]



deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]

import math
from math import log, exp
from random import random, shuffle

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator import CorpusIterator

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}

def initializeOrderTable():
   orderTable = {}
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIterator(language,partition, storeMorph=True).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          vocab_lemmas[line["lemma"]] = vocab_lemmas.get(line["lemma"], 0) + 1

          depsVocab.add(line["dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          for morph in line["morph"]:
              morphKeyValuePairs.add(morph)
          if line["dep"] == "root":
             continue

          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["dep"]
          direction = "HD" if line["head"] < line["index"] else "DH"
          key = (posHead, dep, posHere)
          keyWithDir = (posHead, dep, posHere, direction)
          orderTable[keyWithDir] = orderTable.get(keyWithDir, 0) + 1
          keys.add(key)
          distanceCounts[key] = distanceCounts.get(key,0.0) + 1.0
          distanceSum[key] = distanceSum.get(key,0.0) + abs(line["index"] - line["head"])
   #print orderTable
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key[0], key[1], key[2], "HD"), 0) + 1.0
      dh = orderTable.get((key[0], key[1], key[2], "DH"), 0) + 1.0
      dhLogit = log(dh) - log(hd)
      dhLogits[key] = dhLogit
      originalDistanceWeights[key] = (distanceSum[key] / distanceCounts[key])
   return dhLogits, vocab, keys, depsVocab

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum #+ sum(line.get("children_decisions_logprobs",[]))

#   if "linearization_logprobability" in line:
#      allGradients += line["linearization_logprobability"] # the linearization of this element relative to its siblings affects everything starting at the start of the constituent, but nothing to the left of it
#   else:
#      assert line["dep"] == "root"


   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   result.append(line)
#   print ["DECISIONS_PREPARED", line["index"], line["word"], line["dep"], line["head"], allGradients.data.numpy()[0]]
   line["relevant_logprob_sum"] = allGradients
   if "children_HD" in line:
      for child in line["children_HD"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   return allGradients

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       global model
#       childrenLinearized = []
#       while len(remainingChildren) > 0:
#       if model == "REAL":
 #         return remainingChildren
       logits = [(x, distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]]) for x in remainingChildren]
       logits = sorted(logits, key=lambda x:x[1], reverse=(not reverseSoftmax))
       childrenLinearized = map(lambda x:x[0], logits)
           
#           #print logits
#           if reverseSoftmax:
#              
#              logits = -logits
#           #print (reverseSoftmax, logits)
#           softmax = softmax_layer(logits.view(1,-1)).view(-1)
#           selected = numpy.random.choice(range(0, len(remainingChildren)), p=softmax.data.numpy())
#    #       log_probability = torch.log(softmax[selected])
#   #        assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
#  #         sentence[remainingChildren[selected]-1]["linearization_logprobability"] = log_probability
#           childrenLinearized.append(remainingChildren[selected])
#           del remainingChildren[selected]
       return childrenLinearized           
#           softmax = torch.distributions.Categorical(logits=logits)
#           selected = softmax.sample()
#           print selected
#           quit()
#           softmax = torch.cat(logits)



def orderSentence(sentence, dhLogits, printThings):
   global model

   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   if model == "REAL_REAL":
      eliminated = []
   for line in sentence:
      if line["dep"] == "root":
          root = line["index"]
          continue
      if line["dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         if model == "REAL_REAL":
            eliminated.append(line)
         continue
      key = (sentence[line["head"]-1]["posUni"], line["dep"], line["posUni"])
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
#      probability = 1/(1 + torch.exp(-dhLogit))
      if model == "REAL":
         dhSampled = (line["head"] > line["index"]) #(random() < probability.data.numpy()[0])
      else:
         dhSampled = (dhLogit > 0) #(random() < probability.data.numpy())
#      logProbabilityGradient = (1 if dhSampled else -1) * (1-probability)
#      line["ordering_decision_gradient"] = logProbabilityGradient
      #line["ordering_decision_log_probability"] = torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))

      
     
      direction = "DH" if dhSampled else "HD"
#torch.exp(line["ordering_decision_log_probability"]).data.numpy()[0],
#      if printThings: 
#         print "\t".join(map(str,["ORD", line["index"], ("|".join(line["morph"])+"           ")[:10], ("->".join(list(key)) + "         ")[:22], line["head"], dhLogit, dhSampled, direction]))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
      #sentence[headIndex]["children_decisions_logprobs"] = (sentence[headIndex].get("children_decisions_logprobs", []) + [line["ordering_decision_log_probability"]])


   if model != "REAL_REAL":
      for line in sentence:
         if "children_DH" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
            line["children_DH"] = childrenLinearized
         if "children_HD" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
            line["children_HD"] = childrenLinearized
   if model == "REAL_REAL":
       while len(eliminated) > 0:
          line = eliminated[0]
          del eliminated[0]
          if "removed" in line:
             continue
          line["removed"] = True
          if "children_DH" in line:
            assert 0 not in line["children_DH"]
            eliminated = eliminated + [sentence[x-1] for x in line["children_DH"]]
          if "children_HD" in line:
            assert 0 not in line["children_HD"]
            eliminated = eliminated + [sentence[x-1] for x in line["children_HD"]]

#         shuffle(line["children_HD"])
   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, 0)
   if model == "REAL_REAL":
      linearized = filter(lambda x:"removed" not in x, sentence)
   if printThings or len(linearized) == 0:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))
   return linearized, logits


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()
#print morphKeyValuePairs
#quit()

morphKeyValuePairs = list(morphKeyValuePairs)
itos_morph = morphKeyValuePairs
stoi_morph = dict(zip(itos_morph, range(len(itos_morph))))


posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

posFine = list(posFine)
itos_pos_ptb = posFine
stoi_pos_ptb = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

#print itos_deps

dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)

#dhWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#distanceWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#for i, key in enumerate(itos_deps):
#
#   # take from treebank, or randomize
#   dhLogits[key] = 2*(random()-0.5)
#   dhWeights.data[i] = dhLogits[key]
#
#   originalDistanceWeights[key] = random()  
#   distanceWeights.data[i] = originalDistanceWeights[key]

import os

#if model != "RANDOM_MODEL" and model != "REAL" and model != "RANDOM_BY_TYPE":
#   inpModels_path = "/u/scr/mhahn/deps/"+"/manual_output/"
#   models = os.listdir(inpModels_path)
#   models = filter(lambda x:"_"+model+".tsv" in x, models)
#   if len(models) == 0:
#     assert False, "No model exists"
#   if len(models) > 1:
#     assert False, [models, "Multiple models exist"]
#   
#   with open(inpModels_path+models[0], "r") as inFile:
#      data = map(lambda x:x.split("\t"), inFile.read().strip().split("\n"))
#      header = data[0]
#      data = data[1:]
#    
#   for line in data:
#      head = line[header.index("Head")]
#      dependent = line[header.index("Dependent")]
#      dependency = line[header.index("Dependency")]
#      key = (head, dependency, dependent)
#      dhWeights[stoi_deps[key]] = float(line[header.index("DH_Weight")].replace("[", "").replace("]",""))
#      distanceWeights[stoi_deps[key]] = float(line[header.index("DistanceWeight")].replace("[", "").replace("]",""))
#      originalCounter = int(line[header.index("Counter")])
if model == "RANDOM_MODEL":
  for key in range(len(itos_deps)):
     dhWeights[key] = random() - 0.5
     distanceWeights[key] = random()
  originalCounter = "NA"
elif model == "REAL" or model == "REAL_REAL":
  originalCounter = "NA"
elif model == "RANDOM_BY_TYPE":
  dhByType = {}
  distByType = {}
  for dep in itos_pure_deps:
    dhByType[dep.split(":")[0]] = random() - 0.5
    distByType[dep.split(":")[0]] = random()
  for key in range(len(itos_deps)):
     dhWeights[key] = dhByType[itos_deps[key][1].split(":")[0]]
     distanceWeights[key] = distByType[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"
elif model == "RANDOM_BY_TYPE_CONS":
  distByType = {}
  for dep in itos_pure_deps:
    distByType[dep.split(":")[0]] = random()
  for key in range(len(itos_deps)):
     dhWeights[key] = 1.0
     distanceWeights[key] = distByType[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"
elif model == "RANDOM_MODEL_CONS":
  for key in range(len(itos_deps)):
     dhWeights[key] = 1.0
     distanceWeights[key] = random()
  originalCounter = "NA"
elif model == "GROUND":
  groundPath = "/u/scr/mhahn/deps/manual_output_ground_coarse_final/"
  import os
  files = [x for x in os.listdir(groundPath) if x.startswith(language+"_infer")]
  assert len(files) > 0
  with open(groundPath+files[0], "r") as inFile:
     headerGrammar = next(inFile).strip().split("\t")
     print(headerGrammar)
     dhByDependency = {}
     distByDependency = {}
     for line in inFile:
         line = line.strip().split("\t")
         assert int(line[headerGrammar.index("Counter")]) >= 1000000
#         if line[headerGrammar.index("Language")] == language:
#           print(line)
         dependency = line[headerGrammar.index("Dependency")]
         dhHere = float(line[headerGrammar.index("DH_Mean_NoPunct")])
         distHere = float(line[headerGrammar.index("Distance_Mean_NoPunct")])
         print(dependency, dhHere, distHere)
         dhByDependency[dependency] = dhHere
         distByDependency[dependency] = distHere
  for key in range(len(itos_deps)):
     dhWeights[key] = dhByDependency[itos_deps[key][1].split(":")[0]]
     distanceWeights[key] = distByDependency[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"


lemmas = list(vocab_lemmas.iteritems())
lemmas = sorted(lemmas, key = lambda x:x[1], reverse=True)
itos_lemmas = map(lambda x:x[0], lemmas)
stoi_lemmas = dict(zip(itos_lemmas, range(len(itos_lemmas))))

words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))
#print stoi
#print itos[5]
#print stoi[itos[5]]

if len(itos) > 6:
   assert stoi[itos[5]] == 5

#print dhLogits

#for sentence in getNextSentence():
#   print orderSentence(sentence, dhLogits)

vocab_size = 50
vocab_size = min(len(itos_lemmas),vocab_size)
#print itos[:vocab_size]
#quit()

# 0 EOS, 1 UNK, 2 BOS
#word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = emb_dim).cuda()
#pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = 10).cuda()
#pos_p_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=10).cuda()
#morph_embeddings = torch.nn.Embedding(num_embeddings = len(morphKeyValuePairs)+3, embedding_dim=100).cuda()

word_pos_morph_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+vocab_size+len(morphKeyValuePairs)+3, embedding_dim=emb_dim).cuda()
print posUni
#print posFine
print morphKeyValuePairs
print itos_lemmas[:vocab_size]
print "VOCABULARY "+str(len(posUni)+vocab_size+len(morphKeyValuePairs)+3)
outVocabSize = 3+len(posUni) #+vocab_size+len(morphKeyValuePairs)+3
assert len(posUni)+vocab_size+len(morphKeyValuePairs)+3 < 200
#quit()


itos_total = ["EOS", "EOW", "SOS"] + itos_pos_uni #+ itos_lemmas[:vocab_size] + itos_morph
assert len(itos_total) == outVocabSize
# could also provide per-word subcategorization frames from the treebank as input???


#baseline = nn.Linear(emb_dim, 1).cuda()

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
#pos_ptb_decoder = nn.Linear(128,len(posFine)+3).cuda()


components = [rnn_past, rnn_future, decoder, word_pos_morph_embeddings]


#           klLoss = [None for _ in inputEmbeddings]
#           logStandardDeviationHidden = hiddenToLogSDHidden(hidden[1][0])
#           sampled = torch.normal(hiddenMean, torch.exp(logStandardDeviationHidden))
#           klLoss = 0.5 * (-1 - 2 * (logStandardDeviationHidden) + torch.pow(meanHidden, 2) + torch.exp(2*logStandardDeviationHidden))
#           hiddenNew = sampleToHidden(sampled)
#           cellNew = sampleToCell(sampled)
 


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



#
#weight_made = [torch.cuda.FloatTensor(rnn_dim, rnn_dim).fill_(0) for _ in range(flow_length)]
#for p in weight_made:
#  p.requires_grad=True
#  nn.init.xavier_normal(p)
#
#bias_made = [torch.cuda.FloatTensor(rnn_dim).fill_(0) for _ in range(flow_length)]
#for p in bias_made:
#   p.requires_grad=True
#
#weight_made_mu = [torch.cuda.FloatTensor(rnn_dim, rnn_dim).fill_(0) for _ in range(flow_length)]
#for p in weight_made_mu:
#   p.requires_grad=True
#
#bias_made_mu = [torch.cuda.FloatTensor(rnn_dim).fill_(0) for _ in range(flow_length)]
#for p in bias_made_mu:
#   p.requires_grad=True
#
#weight_made_sigma = [torch.cuda.FloatTensor(rnn_dim, rnn_dim).fill_(0) for _ in range(flow_length)]
#for p in weight_made_sigma:
#   p.requires_grad=True
#
#bias_made_sigma = [torch.cuda.FloatTensor(rnn_dim).fill_(0) for _ in range(flow_length)]
#for p in bias_made_sigma:
#   p.requires_grad=True
#
#
#
#parameters_made = [weight_made, bias_made, weight_made_mu, bias_made_mu, weight_made_sigma, bias_made_sigma]




import torchkit.nn as nn_



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


#           hiddenMade = torch.nn.ReLU(torch.nn.functional.linear(sampled, weight_made * mask, bias_made))
#
#           muMade = torch.ReLU(torch.nn.functional.linear(hiddenMade, weight_made_mu * mask, bias_made_mu))
#           logSigmaMade = (torch.nn.functional.linear(hiddenMade, weight_made_sigma * mask, bias_made_sigma))
#           sigmaMade = torch.exp(logSigmaMade)





components = components + [hiddenToLogSDHidden, cellToMean, sampleToHidden, sampleToCell]

context_dim = 1
flows = [flow(dim=rnn_dim, hid_dim=512, context_dim=context_dim, num_layers=2, activation=torch.nn.ELU()).cuda() for _ in range(flow_length)]


components = components + flows




def parameters():
 for c in components:
   for param in c.parameters():
      yield param
# for q in parameters_made:
#   for p in q:
#    yield p


# yield dhWeights
# yield distanceWeights

#for pa in parameters():
#  print pa

initrange = 0.1
#word_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_u_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_p_embeddings.weight.data.uniform_(-initrange, initrange)
#morph_embeddings.weight.data.uniform_(-initrange, initrange)
word_pos_morph_embeddings.weight.data.uniform_(-initrange, initrange)

decoder.bias.data.fill_(0)
decoder.weight.data.uniform_(-initrange, initrange)
#pos_ptb_decoder.bias.data.fill_(0)
#pos_ptb_decoder.weight.data.uniform_(-initrange, initrange)
#baseline.bias.data.fill_(0)
#baseline.weight.data.uniform_(-initrange, initrange)




crossEntropy = 10.0

#def encodeWord(w):
#   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)

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


mask1 = torch.FloatTensor([[1 if k > d else 0 for d in range(rnn_dim)] for k in range(rnn_dim)]).cuda()
mask2 = torch.FloatTensor([[1 if k < d else 0 for d in range(rnn_dim)] for k in range(rnn_dim)]).cuda()




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
#       for q in parameters_made:
#        for p in q:
#         if p.grad is not None:
#          p.grad.fill_(0)
       totalQuality = 0.0

       if True:
           
           sequenceLength = max(map(len, input_indices))
           for i in range(batchSizeHere):
              input_indices[i] = input_indices[i][:]
              while len(input_indices[i]) < sequenceLength:
                 input_indices[i].append(2)

           inputTensor = Variable(torch.LongTensor(input_indices).transpose(0,1).contiguous()).cuda() # so it will be sequence_length x batchSizeHere
#           print inputTensor
#           quit()

           inputTensorIn = inputTensor[:-1]
           inputTensorOut = inputTensor[1:]

           inputEmbeddings = word_pos_morph_embeddings(inputTensorIn.view(sequenceLength-1, batchSizeHere))
           if doDropout:
              inputEmbeddings = inputDropout(inputEmbeddings)
              if dropout_rate > 0:
                 inputEmbeddings = dropout(inputEmbeddings)

#           output, hidden = rnn(inputEmbeddings, hidden)

           halfSeqLen = int(sequenceLength/2)

           output1, hidden = rnn_past(inputEmbeddings[:halfSeqLen+1], None)

           assert rnn_layers == 1
           meanHidden = cellToMean(hidden[1][0])

           klLoss = [None for _ in inputEmbeddings]
           logStandardDeviationHidden = hiddenToLogSDHidden(hidden[1][0])
#           print(torch.exp(logStandardDeviationHidden))
           memoryDistribution = torch.distributions.Normal(loc=meanHidden, scale=torch.exp(logStandardDeviationHidden))
#           sampled = memoryDistribution.rsample()

           encodedEpsilon = standardNormal.sample()
           sampled = meanHidden + torch.exp(logStandardDeviationHidden) * encodedEpsilon

           #sampledDetach = sampled.detach()
           # now evaluate the density under the transformed prior
           logProbConditional = memoryDistribution.log_prob(sampled).sum(dim=1)  # TODO not clear whether back-prob through sampled?

 #          print(logProbConditional)



           # remove the normalization rnn_dim * 0.5 * math.log(2*math.pi) + 
#           logProbConditional = - rnn_dim * 0.5 * math.log(2*math.pi) - torch.sum(logStandardDeviationHidden, dim=1) - (0.5 * torch.sum(torch.div(encodedEpsilon,torch.exp(2.0 * logStandardDeviationHidden)) * encodedEpsilon, dim=1))
#           print(logProbConditional)


           if True:
   

           
              adjustment = []
              epsilon = sampled
              logdet = torch.autograd.Variable(torch.from_numpy(np.zeros(batchSize).astype('float32')).cuda())
#              n=1
              context = torch.autograd.Variable(torch.from_numpy(np.zeros((batchSize,context_dim)).astype('float32')).cuda())
              for flowStep in range( flow_length):
                epsilon, logdet, context = flows[flowStep]((epsilon, logdet, context))
                if flowStep +1 < flow_length:
                   epsilon, logdet, context = torchkit.flows.FlipFlow(1)((epsilon, logdet, context))

              plainPriorLogProb = standardNormal.log_prob(epsilon).sum(dim=1) #- (0.5 * torch.sum(sampled * sampled, dim=1))
              logProbMarginal = plainPriorLogProb + logdet
           elif False: 
                logProbMarginal = standardNormal.log_prob(sampled).sum(dim=1) #- (0.5 * torch.sum(sampled * sampled, dim=1))


           klLoss = logProbConditional - logProbMarginal
#           print(logProbConditional, logProbMarginal)
#           print(logStandardDeviationHidden)
#           klLoss = 0.5 * (-1 - 2 * (logStandardDeviationHidden) + torch.pow(meanHidden, 2) + torch.exp(2*logStandardDeviationHidden))
 #          klLoss = klLoss.sum(1)
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
#           print lossesWord

           if surprisalTable is not None or printHere:           
             lossesCPU = lossesWord.data.cpu().view((sequenceLength-1), batchSizeHere).numpy()
             if printHere:
                for i in range(0,len(input_indices[0])-1): #range(1,maxLength+1): # don't include i==0
#                   for j in range(batchSizeHere):
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
#                    print lossesCPU[wordStartIndices[j][r]:wordStartIndices[j][r+1],j]
 #                   surprisalTable[r] += sum([x.mean() for x in lossesCPU[wordStartIndices[j][r]:wordStartIndices[j][r+1],j]]) #.data.cpu().numpy()[0]
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
#       probabilities = torch.sigmoid(dhWeights)
#       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))

#       policy_related_loss = lr_policy * (entropy_weight * neg_entropy + policyGradientLoss) # lives on CPU
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
#           continue
#         param.data.sub_(lr_lm * param.grad.data)



def createStream(corpus):
#    global counter
    global crossEntropy
    global printHere
    global devLosses

    input_indices = [2] # Start of Segment
    wordStartIndices = []
#    sentenceStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       #printHere = (sentCount % 10 == 0)
       ordered, _ = orderSentence(sentence, dhLogits, printHere)

#       sentenceStartIndices.append(len(input_indices))
       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
            input_indices.append(stoi_pos_uni[line["posUni"]]+3)
#            if len(itos_pos_ptb) > 1:
#               input_indices.append(stoi_pos_ptb[line["posFine"]]+3+len(stoi_pos_uni))
#            if len(stoi_lemmas) > 1:
#              if stoi_lemmas[line["lemma"]] < vocab_size:
#                input_indices.append(stoi_lemmas[line["lemma"]]+3+len(stoi_pos_uni))
#            for morph in line["morph"]:
#                if morph != "_":
#                   input_indices.append(stoi_morph[morph]+3+len(stoi_pos_uni)+vocab_size)
#            input_indices.append(1)
  #     wordStartIndices.append(len(input_indices))
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
#    sentenceStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       if sentCount % 10 == 0:
         print ["DEV SENTENCES", sentCount]

#       if sentCount == 100:
       #printHere = (sentCount % 10 == 0)
       ordered, _ = orderSentence(sentence, dhLogits, printHere)

#       sentenceStartIndices.append(len(input_indices))
       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
            input_indices.append(stoi_pos_uni[line["posUni"]]+3)
#            if len(itos_pos_ptb) > 1:
#               input_indices.append(stoi_pos_ptb[line["posFine"]]+3+len(stoi_pos_uni))
#            if len(stoi_lemmas) > 1:
#              if stoi_lemmas[line["lemma"]] < vocab_size:
#                input_indices.append(stoi_lemmas[line["lemma"]]+3+len(stoi_pos_uni))
#            for morph in line["morph"]:
#                if morph != "_":
#                   input_indices.append(stoi_morph[morph]+3+len(stoi_pos_uni)+vocab_size)
#            input_indices.append(1)
  #     wordStartIndices.append(len(input_indices))
          if len(wordStartIndices) == horizon:
#             print input_indices
#             print wordStartIndices+[len(input_indices)]
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
   corpusDev = CorpusIterator(language,"dev", storeMorph=True).iterator(rejectShortSentences = False)
   stream = createStreamContinuous(corpusDev)

   surprisalTable = [0 for _ in range(horizon)]
   devCounter = 0
   devMemory = 0
   while True:
#     try:
#        input_indices, wordStartIndices = next(stream)
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
corpusBase = CorpusIterator(language, storeMorph=True)
while failedDevRuns == 0:
  epochCount += 1
  print "Starting new epoch, permuting corpus"
  corpusBase.permute()
#  corpus = getNextSentence("train")
  corpus = corpusBase.iterator(rejectShortSentences = False)
  stream = createStream(corpus)

  while True:
       counter += 1
       printHere = (counter % 50 == 0)


       if counter % DEV_PERIOD == 0:
          newDevLoss, devSurprisalTableHere, newDevMemory = computeDevLoss()
#             devLosses.append(
          devLosses.append(newDevLoss)
          devMemories.append(newDevMemory)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
#          if newDevLoss > 10:
#              print "Abort, training too slow?"
#              devLosses.append(100)

          if lastDevLoss is None or newDevLoss < lastDevLoss:
              devSurprisalTable = devSurprisalTableHere
#          if counter == DEV_PERIOD and model != "REAL_REAL":
#             with open("/u/scr/mhahn/deps/memory-need-neural-pos-only/model-"+language+"_"+__file__+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
#                 print >> outFile, "\t".join(["Key", "DH_Weight", "Distance_Weight"])
#                 for i, key in enumerate(itos_deps):
#                   dhWeight = dhWeights[i]
#                   distanceWeight = distanceWeights[i]
#                   print >> outFile, "\t".join(map(str,[key, dhWeight, distanceWeight]))
          


#          print("Developing, not saving yet")
          print(devSurprisalTable[horizon/2])
          print(devMemories)
 #         continue
#          assert False, "Not saving yet"
          with open("/u/scr/mhahn/deps/memory-upper-neural-pos-only/estimates-"+language+"_"+__file__+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
              print >> outFile, " ".join(sys.argv)
              print >> outFile, " ".join(map(str,devLosses))
              print >> outFile, " ".join(map(str,devSurprisalTable))
              print >> outFile, " ".join(map(str, devMemories))
              print >> outFile, str(sum([x-y for x, y in zip(devSurprisalTable[:horizon/2], devSurprisalTable[horizon/2:])]))


#          if newDevLoss > 10:
#              print "Abort, training too slow?"
#              failedDevRuns = 1
#              break


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



#             lr_lm *= 0.5
 #            continue
#          print "Saving"
#          save_path = "/u/scr/mhahn/deps/"
#          #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
#          with open(save_path+"/manual_output/"+language+"_"+__file__+"_model_"+str(myID)+".tsv", "w") as outFile:
#             print >> outFile, "\t".join(map(str,["FileName","ModelName","Counter", "AverageLoss","Head","DH_Weight","Dependency","Dependent","DistanceWeight", "EntropyWeight", "ObjectiveName"]))
#             for i in range(len(itos_deps)):
#                key = itos_deps[i]
#                dhWeight = dhWeights[i].data.numpy()[0]
#                distanceWeight = distanceWeights[i].data.numpy()[0]
#                head, dependency, dependent = key
#                print >> outFile, "\t".join(map(str,[myID, __file__, counter, crossEntropy, head, dhWeight, dependency, dependent, distanceWeight, entropy_weight, objectiveName]))
#

#dhWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#distanceWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#for i, key in enumerate(itos_deps):
#
#   # take from treebank, or randomize
#   dhLogits[key] = 2*(random()-0.5)
#   dhWeights.data[i] = dhLogits[key]
#
#   originalDistanceWeights[key] = random()  
#   distanceWeights.data[i] = originalDistanceWeights[key]
#
#
#

print(devSurprisalTable[int(horizon/2)])
print(devMemories)

