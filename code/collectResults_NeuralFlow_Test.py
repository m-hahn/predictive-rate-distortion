# for runMemoryManyConfigs_NeuralFlow.py

import os

path = "/u/scr/mhahn/deps/memory-upper-neural-pos-only/"

files = os.listdir(path)

with open("/u/scr/mhahn/results-en-upos-neuralflow-test.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Beta", "EE", "Memories", "UpperBound", "FutureSurp", "avg16", "avg17", "avg18", "avg19", " dropout1", "emb_dim", "rnn_dim", "rnn_layers", "lr", "model", "dropout2", "batchSize", "horizon", "beta", "flow_layers", "flow_type", "Iterations", "flow_hid_dim", "in_flow_layers", "UpperBound2"])
  for name in files:
     if not (name.startswith("test-estimates") and "Neural" in name and "SEPA" in name):
       continue
     with open(path+name, "r") as inFile:
       data = inFile.read().strip().split("\n")
       devLosses = [float(x) for x in data[1].split(" ")]
       if len(devLosses) == 1 or (devLosses[-2] < devLosses[-3] if "WordsUD" not in data[0] else devLosses[-3] < devLosses[-4]):
          if not ("TOY" in name and len(devLosses) >= 10):
             print("Reject ",name, len(devLosses))
             continue
       iterations_number = len(devLosses)
       parameters = data[0].split(" ")[1:]
       if "WordsUD" not in data[0]:
          parameters.insert(10, "0.0")
       assert len(parameters) in [15,17], parameters
       if len(parameters) == 15:
          parameters.append(512)
          parameters.append(2)
       assert len(parameters) == 17, (len(parameters), parameters)
       language, _, dropout1, emb_dim, rnn_dim, rnn_layers, lr, model, dropout2, batchSize, replaceWordsProbability, horizon, beta, flow_layers, flow_type, flow_hid_dim, in_flow_layers = tuple(parameters)
       assert replaceWordsProbability == "0.0", parameters
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

       suffixSurprisals = [sum(surprisalTable[i:15])/(15.0-i) for i in range(15)]
       first = min([x for x in range(len(surprisalTable[:15])) if suffixSurprisals[x] <= avgSurpSecond]+[30])
       upToFirst2 = sum(surprisalTable[:first])
      
      

       print >> outFile, "\t".join(map(str,[language, beta, ee, memories[-2], upToFirst, avgSurpSecond, avg16, avg17, avg18, avg19,  dropout1, emb_dim, rnn_dim, rnn_layers, lr, model, dropout2, batchSize, horizon, beta, flow_layers, flow_type, iterations_number, flow_hid_dim, in_flow_layers, upToFirst2]))



