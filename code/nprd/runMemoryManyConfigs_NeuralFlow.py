import subprocess



import random

while True:
   language = "English"
   languageCode = "en"
   dropout_rate = random.choice([0.0,0.0,0.0,0.1,0.2])
   emb_dim = random.choice([50,100,200,300])
   rnn_dim = random.choice([64,128,256,512])
   rnn_layers = random.choice([1])
   lr_lm = random.choice([0.0001, 0.0005, 0.001, 0.001,  0.001, 0.001, 0.002, 0.004])
   model = "REAL"
   input_dropoutRate = random.choice([0.0,0.0,0.0,0.1,0.2])

   batchSize = random.choice([16, 32,64])
   horizon = 30
   if True:
      beta = random.random() * 0.4
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
   flow_length = random.choice([0,1,2,3,4,5,6,7,8,9,10,15,20])
   flowtype = random.choice(["dsf", "ddsf"])
   
   subprocess.call(map(str,["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE.py", language, languageCode, dropout_rate, emb_dim, rnn_dim, rnn_layers, lr_lm, model, input_dropoutRate, batchSize, horizon, beta, flow_length, flowtype]))
  
   
    
   
