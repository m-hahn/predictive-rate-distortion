# Was called runMemoryManyConfigs_NeuralFlow_Languages.py


import subprocess
import sys
import random
import math

while True:
   language = "English"
   languageCode = language
   dropout_rate = random.choice([0.0,0.0,0.0,0.1,0.2])
   emb_dim = random.choice([50,100,200,300])
   rnn_dim = random.choice([64,128,256,512])
   rnn_layers = random.choice([1])
   lr_lm = random.choice([0.0001, 0.0005, 0.001, 0.001,  0.001, 0.001, 0.002, 0.004])
   model = "REAL"
   input_dropoutRate = random.choice([0.0,0.0,0.0,0.1,0.2])

   batchSize = random.choice([16, 32,64])
   horizon = 30
   beta = math.exp(random.uniform(-6, -0.7))
   print(beta)
   flow_length = random.choice([1,2,3,4,5])
   flowtype = random.choice(["dsf", "ddsf"])
   command = map(str,["./python27", "nprd_pos.py", language, languageCode, dropout_rate, emb_dim, rnn_dim, rnn_layers, lr_lm, model, input_dropoutRate, batchSize, horizon, beta, flow_length, flowtype])
   print(" ".join(command))
   subprocess.call(command)
  
   
    
   
