# Computes estimates also from held-out data.

# Was called zNgramIB_5.py.


import random
import sys


language = "English"
horizon = 3

dirichlet = 0.00001
beta = 1/0.1

code_number = 100 # was 20 in most experiments

# ['Mi with future', tensor(0.8659), 'Mi with past', tensor(4.1936), 'objective', tensor(-0.7820)]


header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator import CorpusIterator

ngrams = {}

lastPosUni = ("EOS",)*(2*horizon-1)
for sentence in CorpusIterator(language,"train", storeMorph=True).iterator():
 for line in sentence:
   nextPosUni = line["posUni"]
   ngram = lastPosUni+(nextPosUni,)
   ngrams[ngram] = ngrams.get(ngram, 0) + 1
   lastPosUni = lastPosUni[1:]+(nextPosUni,)
 nextPosUni = "EOS"
 ngram = lastPosUni+(nextPosUni,)
 ngrams[ngram] = ngrams.get(ngram, 0) + 1
 lastPosUni = lastPosUni[1:]+(nextPosUni,)


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



marginal_future = torch.zeros(len(itos_futures))
for i in range(len(futures)):
   marginal_future[futures_int[i]] += frequencies[i]
marginal_future[-1] = dirichlet * len(itos_pasts)
marginal_future = marginal_future.div(marginal_future.sum())

print(marginal_future)
print(len(marginal_future))




encoding = torch.empty(len(itos_pasts), code_number).uniform_(0.000001, 1)
encoding = encoding.div(encoding.sum(1).unsqueeze(1))

decoding = torch.empty(code_number, len(itos_futures)).uniform_(0.000001, 1)
decoding = decoding.div(decoding.sum(1).unsqueeze(1))
print(decoding[0].sum())
#quit()

marginal_hidden = torch.matmul(marginal_past.unsqueeze(0), encoding).squeeze(0)

def logWithoutNA(x):
   y = torch.log(x)
   y[x == 0] = 0
   return y

objective = 10000000

for t in range(500):
   print("Iteration", t)
 #  print(future_given_past)


   divergence_by_past = (future_given_past * torch.log(future_given_past))
   divergence_by_past[future_given_past == 0] = 0
   divergence_by_past = divergence_by_past.sum(1)
#   print(divergence_by_past)
#   print(future_given_past.size())
#   print(decoding.size())
#   print(divergence_by_past)

   #decodingLog = torch.log(decoding.t())
   #decodingLog[decoding.t() == 0] = 0
   #divergence_by_future =  torch.matmul(future_given_past, decodingLog)   #(future_given_past.unsqueeze(0) * torch.log(decoding).unsqueeze(1)).sum(1)
#   print(divergence_by_future.size())
   #divergence = divergence_by_past.unsqueeze(1) - divergence_by_future

   log_future_given_past = torch.log(future_given_past)
   log_future_given_past[log_future_given_past == 0] = 0

   log_decoding = torch.log(decoding)
   log_decoding[log_decoding == 0] = 0


   ratios = log_future_given_past.unsqueeze(1) - log_decoding.unsqueeze(0)
   divergence2 = (future_given_past.unsqueeze(1) * ratios).sum(2)

   #print(ratios.size())
   print(divergence2.size())
#   print torch.min(divergence2)
   assert torch.min(divergence2) >= -1e-4, torch.min(divergence2)

#   assert torch.min(divergence) >= -1e-4, torch.min(divergence)



#   print(divergence-divergence2)
#   assert all(divergence >= 0), divergence
   total_distortion = torch.matmul(marginal_past.unsqueeze(0), divergence2 * encoding).sum()
   print("Distortion", total_distortion)
   print(decoding[0].sum())

   assert total_distortion >= 0, total_distortion
#   print(divergence.sum())
#   print(divergence.size())
#   print(divergence)
 
#   print(marginal_hidden.sum())
 
   newEncoding = marginal_hidden.unsqueeze(0) * torch.exp(-beta * divergence2)
   norm = newEncoding.sum(1).unsqueeze(1)
   newEncoding = newEncoding.div(norm)
   newEncoding[norm.expand(-1, code_number) == 0] = 0
 #  print(newEncoding.size())
#   quit()
   new_marginal_hidden = torch.matmul(marginal_past.unsqueeze(0), newEncoding).squeeze(0)
   newEncodingInverted = (newEncoding * marginal_past.unsqueeze(1)).div(new_marginal_hidden.unsqueeze(0))
   newEncodingInverted[new_marginal_hidden.unsqueeze(0).expand(len(itos_pasts), -1) == 0] = 0

   newDecoding = torch.matmul(future_given_past.t(), newEncodingInverted).t()
#   newDecoding = newDecoding.div(newDecoding.sum(1).unsqueeze(1))
   assert abs(newDecoding[0].sum()) < 0.01 or abs(newDecoding[0].sum() - 1.0) < 0.01 , newDecoding[0].sum()
#   print(future_given_past.t().size())
   
#   print(newDecoding.size())

   entropy = new_marginal_hidden * torch.log(new_marginal_hidden)
   entropy[new_marginal_hidden == 0] = 0
   entropy = -torch.sum(entropy)
    
   print("Entropy", entropy)
   encoding = newEncoding
   decoding = newDecoding
   marginal_hidden = new_marginal_hidden



   logDecoding = logWithoutNA(decoding) 
   logFutureMarginal = logWithoutNA(marginal_future)
   miWithFuture = torch.sum((decoding * (logDecoding - logFutureMarginal.unsqueeze(0))).sum(1) * marginal_hidden)


   logEncoding = logWithoutNA(encoding)
   log_marginal_hidden = logWithoutNA(marginal_hidden)

   miWithPast = torch.sum((encoding * (logEncoding - log_marginal_hidden.unsqueeze(0))).sum(1) * marginal_past)
   assert miWithFuture <= miWithPast, (miWithFuture , miWithPast)
   newObjective = 1/beta * miWithPast - miWithFuture
   print(["Mi with future", miWithFuture, "Mi with past", miWithPast])
   print(["objectives","last",objective, "new", newObjective])
   assert newObjective - 0.1 <= objective, (newObjective, objective)
   if newObjective == objective:
     print("Ending")
     break
   objective = newObjective
#   quit()

# try on held-out data

ngrams = {}

lastPosUni = ("EOS",)*(2*horizon-1)
for sentence in CorpusIterator(language,"dev", storeMorph=True).iterator():
 for line in sentence:
   nextPosUni = line["posUni"]
   ngram = lastPosUni+(nextPosUni,)
   ngrams[ngram] = ngrams.get(ngram, 0) + 1
   lastPosUni = lastPosUni[1:]+(nextPosUni,)
 nextPosUni = "EOS"
 ngram = lastPosUni+(nextPosUni,)
 ngrams[ngram] = ngrams.get(ngram, 0) + 1
 lastPosUni = lastPosUni[1:]+(nextPosUni,)


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
miWithFuture = torch.sum((decoding * (logDecoding - logFutureMarginal.unsqueeze(0))).sum(1) * marginal_hidden)


logEncoding = logWithoutNA(encoding)
log_marginal_hidden = logWithoutNA(marginal_hidden)

miWithPast = torch.sum((encoding * (logEncoding - log_marginal_hidden.unsqueeze(0))).sum(1) * marginal_past)
assert miWithFuture <= miWithPast, (miWithFuture , miWithPast)
newObjective = 1/beta * miWithPast - miWithFuture
print(["Mi with future", miWithFuture, "Mi with past", miWithPast, "objective", newObjective])
assert miWithFuture >= 0




