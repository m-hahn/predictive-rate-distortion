# Was called runMemoryManyConfigs_NeuralFlow_Words_English.py


import subprocess
import random

while True:
   language = "PTB"
   languageCode = "en"
   dropout_rate = random.choice([0.0,0.1,0.2])
   emb_dim = random.choice([50,100,200,300])
   rnn_dim = random.choice([256,512]) #random.choice([256,512])
   rnn_layers = random.choice([1])
   lr_lm = random.choice([0.00001, 0.00002, 0.00005, 0.00005,0.0001,0.0001,0.0001,0.0002, 0.001, 0.001,  0.001, 0.001, 0.002, 0.004]) # 0.0001, 0.0005, 
   model = "REAL"
   input_dropoutRate = random.choice([0.0,0.0,0.0,0.1,0.2])

   batchSize = random.choice([32, 128])
   horizon = 30
   beta = math.exp(random.uniform(-6, -1.0))
   print(beta)
   flow_length = random.choice([1,2,3]) #4,5,6])
   flowtype = random.choice(["dsf", "ddsf"])
   
   subprocess.call(map(str,["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE_Words.py", language, languageCode, dropout_rate, emb_dim, rnn_dim, rnn_layers, lr_lm, model, input_dropoutRate, batchSize, horizon, beta, flow_length, flowtype]))
  
   
    
   
