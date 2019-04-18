import subprocess



import random

while True:
   language = random.choice(["repeat"]) # , "forget2_0_5b"
   languageCode = "toy"
   dropout_rate = random.choice([0.0,0.1,0.1,0.1,0.1,0.2])
   emb_dim = random.choice([50]) # ,100
   rnn_dim = random.choice([32,64,128,128,128,128,128,256])
   rnn_layers = random.choice([1])
   lr_lm = random.choice([0.0001, 0.0005, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,  0.001, 0.001, 0.002, 0.003, 0.004])
   model = "REAL"
   input_dropoutRate = random.choice([0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.2])

   batchSize = random.choice([16, 32,32,32,32,64, 128])
   horizon = 30
   if True:
      #beta = random.random() * 0.5
      beta = 0.00268427 #random.random() * 0.005
   else:
      broadRange = random.choice([0.001, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1])
      if broadRange < 0.09:
        beta = random.choice(range(1,10)) * broadRange
      elif broadRange == 0.1:
        beta = random.choice(range(1,7)) * broadRange + random.choice(range(1,10)) * 0.01
      else:
        assert False
      if beta > 0.7 or beta < 0.005:
         continue
   print(beta)
   flow_length = 1 #random.choice([1,2,3,4]) #,4,5]) #,6,7,8,9,10,15,20])
   flowtype = random.choice(["dsf", "ddsf", "ddsf", "ddsf", "ddsf"])
   flow_hid_dim = random.choice([32,64,128,512,512,512,512,512])
   flow_layers = random.choice([2])
   weight_decay = random.choice([1e-8, 1e-7, 1e-6, 1e-6, 1e-6, 1e-6,1e-6,1e-6,1e-6, 1e-5])
   klAnnealing = True #True #random.choice([True, False])
   subprocess.call(map(str,["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE_TOY_Repeat.py", language, languageCode, dropout_rate, emb_dim, rnn_dim, rnn_layers, lr_lm, model, input_dropoutRate, batchSize, horizon, beta, flow_length, flowtype, flow_hid_dim, flow_layers, weight_decay, klAnnealing]))
  
   
    
   
