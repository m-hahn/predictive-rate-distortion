# for runMemoryManyConfigs_NeuralFlow.py

import os

path = "/u/scr/mhahn/deps/memory-upper-neural-pos-only-discrete/"

files = os.listdir(path)

with open("/u/scr/mhahn/results-oce.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Beta", "Horizon", "CodeNumber", "Dirichlet", "Memory", "Surprisal"])
  for name in files:
     if not (name.startswith("estimates") and "zNgramIB" in name):
       continue
     with open(path+name, "r") as inFile:
       data = inFile.read().strip().split("\n")
       params = dict([tuple(x.split(" ")) for x in data[0].split("\t")])
       memory = float(data[1])
       surprisal = float(data[2])
       print >> outFile, "\t".join(map(str,[params["language"], params["beta"], params["horizon"], params["code_number"], params["dirichlet"], memory, surprisal]))



