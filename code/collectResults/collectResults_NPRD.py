# for runMemoryManyConfigs_NeuralFlow.py

import os

path = "../../results/outputs/nprd/"

files = os.listdir(path)

with open("../../results/results-en-upos-neuralflow.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Beta", "EE", "Memories", "UpperBound", "FutureSurp", "avg16", "avg17", "avg18", "avg19", " dropout1", "emb_dim", "rnn_dim", "rnn_layers", "lr", "model", "dropout2", "batchSize", "horizon", "beta", "flow_layers", "flow_type", "Iterations", "flow_hid_dim", "in_flow_layers"])
  for name in files:
     if not (name.startswith("estimates") and "Neural" in name and "SEPA" in name):
       continue
     with open(path+name, "r") as inFile:
       data = inFile.read().strip().split("\n")
       if len(data) < 2:
         continue
       devLosses = [float(x) for x in data[1].split(" ")]
       if len(devLosses) == 1 or devLosses[-1] < devLosses[-2]:
          if not ("TOY" in name and len(devLosses) >= 10) and ("epeat" not in name) and ("PTB" not in name):
             print("Reject ",name, len(devLosses))
             continue
       iterations_number = len(devLosses)
       parameters = data[0].split(" ")[1:]
       if len(parameters) == 14:
          parameters.append(512)
          parameters.append(2)
       if len(parameters) == 16:
          parameters.append(1e-5)
       if len(parameters) == 17:
          parameters.append(False)
       if len(parameters) == 18:
          parameters.append(0.0)
       language, _, dropout1, emb_dim, rnn_dim, rnn_layers, lr, model, dropout2, batchSize, horizon, beta, flow_layers, flow_type, flow_hid_dim, in_flow_layers, weight_decay, klAnnealing, klIncrease  = tuple(parameters)
       #assert str(flow_layers2) == flow_layers, (flow_layers, flow_layers2)
       surprisalTable = [float(x) for x in data[2].split(" ")]
       memories = [float(x) for x in data[3].split(" ")]
       ee = float(data[4])
#       beta = data[0].split(" ")[-3]
       assert len(surprisalTable) == 30
       avgSurpSecond = sum(surprisalTable[15:])/15
       first = min([x for x in range(len(surprisalTable)) if surprisalTable[x] <= avgSurpSecond])
       upToFirst = sum(surprisalTable[:first])
       avg16 = surprisalTable[15]
       avg17 = sum(surprisalTable[15:17])/2
       avg18 = sum(surprisalTable[15:18])/3
       avg19 = sum(surprisalTable[15:19])/4

       print >> outFile, "\t".join(map(str,[language, beta, ee, memories[-2] if len(memories) > 1 else memories[-1], upToFirst, avgSurpSecond, avg16, avg17, avg18, avg19,  dropout1, emb_dim, rnn_dim, rnn_layers, lr, model, dropout2, batchSize, horizon, beta, flow_layers, flow_type, iterations_number, flow_hid_dim, in_flow_layers]))



