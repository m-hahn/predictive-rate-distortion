# GD version of 5, appears to come up with better solutions than 5
# With arguments

# Was called zNgramIB_9_TOY.py.

import matplotlib
matplotlib.use('Agg')


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--language", type=str, dest="language")
parser.add_argument("--horizon", type=int, dest="horizon")
parser.add_argument("--code_number", type=int, dest="code_number")
parser.add_argument("--beta", type=float, dest="beta")
parser.add_argument("--dirichlet", type=float, dest="dirichlet")

args_names = ["language", "horizon", "code_number", "beta", "dirichlet"]
args = parser.parse_args()



# /u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7

import random
import sys


language = args.language
horizon = args.horizon
code_number = args.code_number
beta = 1/args.beta
dirichlet = args.dirichlet

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIteratorToy import CorpusIteratorToy

ngrams = {}

lastPosUni = ("EOS",)*(2*horizon-1)
for sentence in CorpusIteratorToy(language,"train", storeMorph=True).iterator():
 for line in sentence:
   nextPosUni = line["posUni"]
   ngram = lastPosUni+(nextPosUni,)
   ngrams[ngram] = ngrams.get(ngram, 0) + 1
   lastPosUni = lastPosUni[1:]+(nextPosUni,)
 nextPosUni = "EOS"
 ngram = lastPosUni+(nextPosUni,)
 ngrams[ngram] = ngrams.get(ngram, 0) + 1
 lastPosUni = lastPosUni[1:]+(nextPosUni,)

# /u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7 zNgramIB.py

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


ngrams = list(ngrams.iteritems())

#ngrams = [x for x in ngrams if x[1] > 100]
#print(ngrams)
print(["Number of ngrams", len(ngrams)])


keys = [x[0] for x in ngrams]

total = sum([x[1] for x in ngrams])



frequencies = [x[1] for x in ngrams]

pasts = [x[:horizon] for x in keys]  #range(horizon:range(horizon, 2*horizon)]
futures = [x[horizon:] for x in keys]


itos_pasts = list(set(pasts)) + ["_OOV_"]
itos_futures = list(set(futures)) + ["_OOV_"]
stoi_pasts = dict(zip(itos_pasts, range(len(itos_pasts))))
stoi_futures = dict(zip(itos_futures, range(len(itos_futures))))

import torch

pasts_int = torch.LongTensor([stoi_pasts[x] for x in pasts])
futures_int = torch.LongTensor([stoi_futures[x] for x in futures])


marginal_past = torch.zeros(len(itos_pasts))
for i in range(len(pasts)):
   marginal_past[pasts_int[i]] += frequencies[i]
marginal_past[-1] = dirichlet * len(itos_futures)
marginal_past = marginal_past.div(marginal_past.sum())
print(marginal_past)
print(len(marginal_past))

future_given_past = torch.zeros(len(itos_pasts), len(itos_futures))
for i in range(len(pasts)):
  future_given_past[pasts_int[i]][futures_int[i]] = frequencies[i]
future_given_past[-1].fill_(dirichlet)
future_given_past[:,-1].fill_(dirichlet)

future_given_past += 0.00001

print(future_given_past.sum(1))
#quit()
 
future_given_past = future_given_past.div(future_given_past.sum(1).unsqueeze(1))

print(future_given_past[0].sum())


def logWithoutNA(x):
   y = torch.log(x)
   y[x == 0] = 0
   return y


marginal_future = torch.zeros(len(itos_futures))
for i in range(len(futures)):
   marginal_future[futures_int[i]] += frequencies[i]
marginal_future[-1] = dirichlet * len(itos_pasts)
marginal_future = marginal_future.div(marginal_future.sum())
logFutureMarginal = logWithoutNA(marginal_future)

print(marginal_future)
print(len(marginal_future))

import torch.optim



encoding_logits = torch.empty(len(itos_pasts), code_number).uniform_(0.000001, 1)
#encoding = encoding.div(encoding.sum(1).unsqueeze(1))
encoding_logits.requires_grad = True

#decoding_logits = torch.empty(code_number, len(itos_futures)).uniform_(0.000001, 1)
#decoding = decoding.div(decoding.sum(1).unsqueeze(1))
#decoding_logits.requires_grad = True


optimizer = torch.optim.Adam([encoding_logits], lr= 0.01) # also try 0.001, 0.0001


softmax = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()

#print(decoding[0].sum())
#quit()



log_future_given_past = torch.log(future_given_past)
log_future_given_past[log_future_given_past == 0] = 0



objective = 10000000

for t in range(100000):
   print("Iteration", t)
   optimizer.zero_grad()

   encoding = softmax(encoding_logits)
#   decoding = softmax(decoding_logits)


   marginal_hidden = torch.matmul(marginal_past.unsqueeze(0), encoding).squeeze(0)
#   print(marginal_hidden)
#   print(marginal_hidden.sum(0))
   log_marginal_hidden = logWithoutNA(marginal_hidden)


   normalizedEncoding = encoding.div(marginal_hidden.unsqueeze(0))
   decoding = torch.matmul((marginal_past.unsqueeze(1) * future_given_past).t(), normalizedEncoding).t()
   print(decoding[0].sum())

   logEncoding = logsoftmax(encoding_logits)
   logDecoding = logWithoutNA(decoding)



#   print(encoding.sum(1))
   miWithFuture = torch.sum((decoding * (logDecoding - logFutureMarginal.unsqueeze(0))).sum(1) * marginal_hidden) # WRONG



   miWithPast = torch.sum((encoding * (logEncoding - log_marginal_hidden.unsqueeze(0))).sum(1) * marginal_past)
   assert miWithFuture <= miWithPast + 1e-5, (miWithFuture , miWithPast)
   newObjective = 1/beta * miWithPast - miWithFuture
   newObjective.backward()
   optimizer.step()
 
   print(["Mi with future", miWithFuture, "Mi with past", miWithPast])
   print(["objectives","last",objective, "new", newObjective])
   #assert newObjective - 0.1 <= objective, (newObjective, objective)
   if abs(newObjective - objective) < 1e-13:
     print("Ending")
     break
   objective = newObjective
#   quit()


futureSurprisal_train = -((future_given_past * marginal_past.unsqueeze(1)).unsqueeze(1) * encoding.unsqueeze(2) * logDecoding.unsqueeze(0)).sum()
miWithPast_train = miWithPast


#assert False, "how is the vocabulary for held-out data generated????"
# try on held-out data

ngrams = {}

lastPosUni = ("EOS",)*(2*horizon-1)
for sentence in CorpusIteratorToy(language,"dev", storeMorph=True).iterator():
 for line in sentence:
   nextPosUni = line["posUni"]
   ngram = lastPosUni+(nextPosUni,)
   ngrams[ngram] = ngrams.get(ngram, 0) + 1
   lastPosUni = lastPosUni[1:]+(nextPosUni,)
 nextPosUni = "EOS"
 ngram = lastPosUni+(nextPosUni,)
 ngrams[ngram] = ngrams.get(ngram, 0) + 1
 lastPosUni = lastPosUni[1:]+(nextPosUni,)

# /u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7 zNgramIB.py

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


ngrams = list(ngrams.iteritems())

#ngrams = [x for x in ngrams if x[1] > 100]
#print(ngrams)
#print(["Number of ngrams", len(ngrams)])


keys = [x[0] for x in ngrams]

total = sum([x[1] for x in ngrams])



frequencies = [x[1] for x in ngrams]

pasts = [x[:horizon] for x in keys]  #range(horizon:range(horizon, 2*horizon)]
futures = [x[horizon:] for x in keys]



import torch

pasts_int = torch.LongTensor([stoi_pasts[x] if x in stoi_pasts else stoi_pasts["_OOV_"] for x in pasts])
futures_int = torch.LongTensor([stoi_futures[x]  if x in stoi_futures else stoi_futures["_OOV_"] for x in futures])

#code_number = 20

marginal_past = torch.zeros(len(itos_pasts))
for i in range(len(pasts)):
   marginal_past[pasts_int[i]] += frequencies[i]
#marginal_past[-1] = len(itos_futures)
marginal_past = marginal_past.div(marginal_past.sum())

future_given_past = torch.zeros(len(itos_pasts), len(itos_futures))
for i in range(len(pasts)):
  future_given_past[pasts_int[i]][futures_int[i]] = frequencies[i]
#future_given_past[-1].fill_(1)
#future_given_past[:,-1].fill_(1)

future_given_past += 0.00001


future_given_past = future_given_past.div(future_given_past.sum(1).unsqueeze(1))

marginal_future = torch.zeros(len(itos_futures))
for i in range(len(futures)):
   marginal_future[futures_int[i]] += frequencies[i]
marginal_future = marginal_future.div(marginal_future.sum())


marginal_hidden = torch.matmul(marginal_past.unsqueeze(0), encoding).squeeze(0)


logDecoding = logWithoutNA(decoding) 
logFutureMarginal = logWithoutNA(marginal_future)
#miWithFuture = torch.sum((decoding * (logDecoding - logFutureMarginal.unsqueeze(0))).sum(1) * marginal_hidden)

# past,intermediate, future
futureSurprisal = -((future_given_past * marginal_past.unsqueeze(1)).unsqueeze(1) * encoding.unsqueeze(2) * logDecoding.unsqueeze(0)).sum()
#futureMarginalCrossEntropy = 
#futureSurprisal = torch.sum((decoding * (logDecoding)).sum(1) * marginal_hidden)



logEncoding = logWithoutNA(encoding)
#log_marginal_hidden = logWithoutNA(marginal_hidden) # this should NOT be recomputed

miWithPast = torch.sum((encoding * (logEncoding - log_marginal_hidden.unsqueeze(0))).sum(1) * marginal_past)
assert miWithFuture <= miWithPast + 1e-5, (miWithFuture , miWithPast)
newObjective = 1/beta * miWithPast - miWithFuture
print(["Mi with past", miWithPast, "Future Surprisal", futureSurprisal/horizon, "Horizon", horizon, "but the comparison with longer blocks isn't really fair"]) # "Mi with future", miWithFuture


myID = random.randint(0,10000000)


with open("/u/scr/mhahn/deps/memory-upper-neural-pos-only-discrete/estimates-"+language+"_"+__file__+"_model_"+str(myID)+".txt", "w") as outFile:
    print >> outFile, "\t".join(x+" "+str(getattr(args,x)) for x in args_names)
    print >> outFile, float(miWithPast)
    print >> outFile, float(futureSurprisal/horizon)
    print >> outFile, float(miWithPast_train)
    print >> outFile, float(futureSurprisal_train/horizon)





