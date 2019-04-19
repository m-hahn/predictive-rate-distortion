# Was called runMemoryManyConfigs_NeuralFlow_Words_UD_Arabic.py.

import subprocess



import random, math

while True:
   language = "Arabic"
   languageCode = language
   dropout_rate = random.choice([0.1, 0.4]) # dropout1
   emb_dim = random.choice([150])
   rnn_dim = random.choice([256,512])
   rnn_layers = random.choice([1])
   lr_lm = random.choice([0.00005, 0.0001, 0.0005, 0.001]) # 0.0001, 0.0005, 
   model = "REAL_REAL"
   input_dropoutRate = random.choice([0.05, 0.2]) #dropout2

   batchSize = random.choice([16, 32, 64])
   input_noising = 0.0
   horizon = 30
   beta = math.exp(random.uniform(-6, -0.5)) #random.random() * 0.05
   print(beta)
   flow_length = random.choice([1,2,3,4])
   flowtype = random.choice(["dsf", "ddsf"])
   
   subprocess.call(map(str,["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE_WordsUD.py", language, languageCode, dropout_rate, emb_dim, rnn_dim, rnn_layers, lr_lm, model, input_dropoutRate, batchSize, input_noising, horizon, beta, flow_length, flowtype]))
  
   
    
   
