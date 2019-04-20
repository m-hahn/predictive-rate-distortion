# Generates samples from analytically known processes.

import os
import random
import sys

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]


def encode(word):
   return {"posUni" : word}

def readUDCorpus(language, partition):
      """
        @language: the name of the process (even, rip, repeat). Note that repeat is called Copy3 in the paper.
        @partition: train or dev (for the held-out set). Note that since this function creates a fresh sample whenever it is called, no distinction between development and test set is made.
        
        In order to add functionality for another (stationary) process, create analogous functionality creating a sample from that process.
      """
      data = [[]]
      if language == "even":
         state = 0
         for _ in range(100000 if partition == "dev" else 100000):
           if state == 0:
             if random.random() < 0.5:
                 data[-1].append(encode("0"))
             else:
                state = 1
                data[-1].append(encode("1"))
           else:
               state = 0
               data[-1].append(encode("1"))
      elif language == "repeat":
        for _ in range(3000 if partition == "dev" else 3000):
            word = [random.choice(["0", "1", "2"]) for _ in range(15)]
            for i in range(len(word)):
               data[-1].append(encode(word[i]))
            word = word[::-1]
            for i in range(len(word)):
               data[-1].append(encode(word[i]))
      elif language == "rip":
         state = 0
         for _ in range(100000 if partition == "dev" else 100000):
             if state == 0:
                if random.random() < 0.5:
                    data[-1].append(encode("0"))
                    state = 1
                else:
                    data[-1].append(encode("1"))
                    state = 2
             elif state == 1:
                 if random.random() < 0.5:
                     data[-1].append(encode("0"))
                 else:
                     data[-1].append(encode("1"))
                 state = 2
             elif state == 2:
                 data[-1].append(encode("1"))
                 state = 0
             else:
                  assert False
      else:
          assert False
      return data

class CorpusIteratorToy():
   def __init__(self, language, partition="train", storeMorph=False, splitLemmas=False):
      data = readUDCorpus(language, partition)
      random.shuffle(data)
      self.data = data
      self.partition = partition
      self.language = language
      assert len(data) > 0, (language, partition)
   def permute(self):
      random.shuffle(self.data)
   def length(self):
      return len(self.data)
   def processSentence(self, sentence):
        return sentence 
   def getSentence(self, index):
      result = self.processSentence(self.data[index])
      return result
   def iterator(self, rejectShortSentences = False):
     for sentence in self.data:
        if len(sentence) < 3 and rejectShortSentences:
           continue
        yield self.processSentence(sentence)


