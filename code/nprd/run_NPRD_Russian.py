# Was called runMemoryManyConfigs_NeuralFlow_Words_UD_Russian.py

import subprocess
import random, math

while True:
   language = "Russian"
   languageCode = "Russian"
   dropout_rate = random.choice([0.4])
   emb_dim = random.choice([150])
   rnn_dim = 256 #random.choice([256,512])
   rnn_layers = random.choice([1])
   lr_lm = random.choice([0.00005, 0.0001, 0.0005, 0.001]) # 0.0001, 0.0005, 
   model = "REAL_REAL"
   input_dropoutRate = random.choice([0.2])

   batchSize = 32 #random.choice([16, 32,64])
   input_noising = 0.0
   horizon = 30
   beta = math.exp(random.uniform(-6, -0.7)) #random.random() * 0.05
   print(beta)
   flow_length = 5 #random.choice([0,1,2,3]) #4,5,6])
   flowtype = random.choice(["dsf", "ddsf"])
   
   subprocess.call(map(str,["./python27", "nprd_words.py", language, languageCode, dropout_rate, emb_dim, rnn_dim, rnn_layers, lr_lm, model, input_dropoutRate, batchSize, input_noising, horizon, beta, flow_length, flowtype]))
  
   
    
   
