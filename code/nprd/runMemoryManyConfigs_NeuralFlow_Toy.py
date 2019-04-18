import subprocess



import random

while True:
   language = random.choice(["even", "rrxor","rip"]) # , "forget2_0_5b"
   languageCode = "toy"
   dropout_rate = random.choice([0.0,0.0,0.0,0.1,0.2])
   emb_dim = random.choice([50,100])
   rnn_dim = random.choice([32]) #,64,128])
   rnn_layers = random.choice([1])
   lr_lm = random.choice([0.0001, 0.0005, 0.001, 0.001,  0.001, 0.001, 0.002, 0.004])
   model = "REAL"
   input_dropoutRate = 0.0 #random.choice([0.0,0.0,0.0,0.1,0.2]) # for this kind of data, input droput seems like a bad idea

   batchSize = random.choice([16, 32,64])
   horizon = 30
   if True:
      #beta = random.random() * 0.5
      beta = random.random() * 0.4 + 0.2
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
   flow_length = random.choice([1,2]) # 0 #,3,4,5]) #,6,7,8,9,10,15,20])
   flowtype = random.choice(["dsf", "ddsf"])
   flow_hid_dim = random.choice([32,64,128,512])
   flow_layers = random.choice([2])
   subprocess.call(map(str,["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE_TOY.py", language, languageCode, dropout_rate, emb_dim, rnn_dim, rnn_layers, lr_lm, model, input_dropoutRate, batchSize, horizon, beta, flow_length, flowtype, flow_hid_dim, flow_layers]))
  
   
    
   
