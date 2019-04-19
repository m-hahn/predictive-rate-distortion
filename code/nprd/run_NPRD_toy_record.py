# Was called runMemoryManyConfigs_NeuralFlow_Toy_Record.py


language = "rip" #random.choice(["even", "rrxor","rip"]) # , "forget2_0_5b"
languageCode = "toy"
dropout_rate = 0.2 #random.choice([0.0,0.0,0.0,0.1,0.2])
emb_dim = 100 #random.choice([50,100])
rnn_dim = 32 #random.choice([32]) #,64,128])
rnn_layers = 1 #random.choice([1])
lr_lm = 1e-3 #random.choice([0.0001, 0.0005, 0.001, 0.001,  0.001, 0.001, 0.002, 0.004])
model = "REAL"
input_dropoutRate = 0.0 #random.choice([0.0,0.0,0.0,0.1,0.2])
batchSize = 16 #random.choice([16, 32,64])
horizon = 30
beta = 0.6
flow_length = 2 #random.choice([1,2]) # 0 #,3,4,5]) #,6,7,8,9,10,15,20])
flowtype = "ddsf" #random.choice(["dsf", "ddsf"])
flow_hid_dim = 512 #random.choice([32,64,128,512])
flow_layers = 2 #random.choice([2])
command = map(str,["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE_TOY_Record.py", language, languageCode, dropout_rate, emb_dim, rnn_dim, rnn_layers, lr_lm, model, input_dropoutRate, batchSize, horizon, beta, flow_length, flowtype, flow_hid_dim, flow_layers])
print(" ".join(command))

language = "rip" #random.choice(["even", "rrxor","rip"]) # , "forget2_0_5b"
languageCode = "toy"
dropout_rate = 0.2 #random.choice([0.0,0.0,0.0,0.1,0.2])
emb_dim = 100 #random.choice([50,100])
rnn_dim = 32 #random.choice([32]) #,64,128])
rnn_layers = 1 #random.choice([1])
lr_lm = 5e-4 #random.choice([0.0001, 0.0005, 0.001, 0.001,  0.001, 0.001, 0.002, 0.004])
model = "REAL"
input_dropoutRate = 0.0 #random.choice([0.0,0.0,0.0,0.1,0.2])
batchSize = 16 #random.choice([16, 32,64])
horizon = 30
beta = 0.25
flow_length = 2 #random.choice([1,2]) # 0 #,3,4,5]) #,6,7,8,9,10,15,20])
flowtype = "dsf" #random.choice(["dsf", "ddsf"])
flow_hid_dim = 512 #random.choice([32,64,128,512])
flow_layers = 2 #random.choice([2])
command = map(str,["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE_TOY_Record.py", language, languageCode, dropout_rate, emb_dim, rnn_dim, rnn_layers, lr_lm, model, input_dropoutRate, batchSize, horizon, beta, flow_length, flowtype, flow_hid_dim, flow_layers])
print(" ".join(command))
 
   
    
     
    
   
