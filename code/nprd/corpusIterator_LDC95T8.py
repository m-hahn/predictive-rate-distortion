import os
import random
import sys

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]


import accessLDC95T8

class CorpusIterator_LDC95T8():
   def __init__(self, language="LDC95T8", partition="train", shuffleData=True, shuffleDataSeed=None):


      if partition == "train":
         self.data = accessLDC95T8.LDC95T8[3000:]
      elif partition == "test":
         self.data = accessLDC95T8.LDC95T8[:1500]
      elif partition == "dev":
         self.data = accessLDC95T8.LDC95T8[1500:3000]

      if shuffleData:
       if shuffleDataSeed is None:
         random.shuffle(self.data)
       else:
         random.Random(shuffleDataSeed).shuffle(data)

      self.partition = partition
      self.language = language
      assert len(self.data) > 0, (language, partition)
   def permute(self):
      random.shuffle(self.data)
   def length(self):
      return len(self.data)
   def getSentence(self, index):
      result = self.data[index]
      return result
   def iterator(self, rejectShortSentences = False):
     for sentence in self.data:
        if len(sentence) < 3 and rejectShortSentences:
           continue
        yield sentence


