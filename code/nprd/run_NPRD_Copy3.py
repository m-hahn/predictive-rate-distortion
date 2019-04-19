# Was called runMemoryManyConfigs_NeuralFlow_Toy_Repeat.py


import subprocess


# This is one of the most successful configurations:
# Beta = 0.00268427
# dropout1 0
# emb_dim 50
# rnn_dim 256
# rnn_layers 1
# lr 0.003
# dropout2 0
# batchSize 128
# flow_layers 1
# flow_type ddsf
# flow_hid_dim 512
# in_flow_layers 2

import random

while True:
   language = random.choice(["repeat"]) 
   languageCode = "toy"
   dropout_rate = random.choice([0.0,0.0,0.1,0.2])
   emb_dim = random.choice([50]) # ,100
   rnn_dim = random.choice([32,64,128])
   rnn_layers = random.choice([1])
   lr_lm = random.choice([0.0001, 0.0005, 0.001, 0.001,  0.001, 0.001, 0.002, 0.003, 0.004])
   model = "REAL"
   input_dropoutRate = random.choice([0.0,0.0,0.0,0.1,0.2])

   batchSize = random.choice([16, 32,64, 128])
   horizon = 30
   beta = random.random() * 0.005
   print(beta)
   flow_length = 1 #random.choice([1,2,3,4]) #,4,5]) #,6,7,8,9,10,15,20])
   flowtype = random.choice(["dsf", "ddsf"])
   flow_hid_dim = random.choice([32,64,128,512])
   flow_layers = random.choice([2])
   weight_decay = random.choice([1e-8, 1e-7, 1e-6, 1e-6, 1e-6, 1e-6, 1e-5])
   klAnnealing = True #True #random.choice([True, False])
   subprocess.call(map(str,["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE_TOY_Repeat.py", language, languageCode, dropout_rate, emb_dim, rnn_dim, rnn_layers, lr_lm, model, input_dropoutRate, batchSize, horizon, beta, flow_length, flowtype, flow_hid_dim, flow_layers, weight_decay, klAnnealing]))
  
   
    
   
