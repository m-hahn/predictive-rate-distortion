# Was called runMemoryManyConfigs_NeuralFlow_Words_English.py

import math
import subprocess
import random
import os
from paths import LOG_PATH_WORDS
import numpy as np

ids = [x.split("_")[-1][:-4] for x in os.listdir("/home/user/CS_SCR/CODEBOOKS/") if "nprd" in x]
print(ids)

language = "PTB"
model = "REAL"
ress = []
for idn in ids:
   with open("/home/user/CS_SCR/CODE/predictive-rate-distortion/results/outputs-nprd-words/test-estimates-"+language+"_"+"nprd_words_PTB_saveCodebook.py"+"_model_"+idn+"_"+model+".txt", "r") as inFile:
      args = next(inFile).strip().split(" ")
      print(args)
      beta = args[-3]
      beta = -math.log(float(beta))
      if abs(beta - round(beta)) > 0.001:
          continue
      if round(beta) not in [1.0, 3.0, 5.0]:
           continue
   dat = []
   with open("/home/user/CS_SCR/CODE/predictive-rate-distortion/results/nprd-samples/samples_"+idn+".txt", "r") as inFile:
       data = [x.split("\t") for x in inFile.read().strip().split("\n")]
       for i in range(0, len(data), 30):
           dat.append(data[i:i+30])
           assert data[i][0] == '0', data[i]
 #  print(len(dat))
   ress.append((idn, round(beta), dat))

ress = sorted(ress, key=lambda x:x[1])
#print(ress)



import matplotlib
import matplotlib.pyplot as plt



numsOfTexts = [len(x[2]) for x in ress]
print(numsOfTexts)

variations = []

for j in range(min(numsOfTexts)): #len(ress[0][2])):
   data = ress[0][2][j]
   print(data)

   pos = np.asarray([int(x[0]) for x in data])
   
  
   char = [x[1] for x in data]



   ys = []
   for i in range(len(ress)):
       ys.append([float(x[2]) for x in ress[i][2][j]])
       ys[-1] = np.asarray(ys[-1])
       print(ys[-1])
   
   
   
   print(ys[0])
   
   fig, ax = plt.subplots()
   for y, color, style in zip(ys, ["red", "green", "blue"], ["dotted", "dashdot", "solid"]):
       ax.plot(pos[16:], y[16:], color=color, linestyle=style)
   
   variation = [y[16] for y in ys]
   variation = max(variation) - min(variation)
   variations.append((j, variation))
   
   plt.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.17)
   fig.set_size_inches(10, 1.7)
   #ax.grid(False)
   plt.xticks(pos[16:], [x.decode("utf-8") for x in char][16:])
  
#   plt.axvline(x=15.5, color="green") 
   
   ax.grid(False)
#   figure(figsize=(25,10))
   fileName = "sample_"+str(j)
   fig.savefig("figures/"+fileName+".png")
#   plt.show()
   plt.gcf().clear()

   print("figures/"+fileName+".png")
   with open("figures/"+fileName+".txt", "w") as outFile:
       print >> outFile, (" ".join(char[:16]))

print(sorted(variations, key=lambda x:x[1]))


