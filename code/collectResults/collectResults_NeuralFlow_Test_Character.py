# for runMemoryManyConfigs_NeuralFlow.py

import os
horizon = 80

path = "/u/scr/mhahn/deps/memory-upper-neural-characters/"

files = os.listdir(path)

with open("/u/scr/mhahn/results-en-upos-neuralflow-test-characters.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Script", "Beta", "EE", "Memories", "UpperBound", "FutureSurp", "avg16", "avg17", "avg18", "avg19", " dropout1", "emb_dim", "rnn_dim", "rnn_layers", "lr", "model", "dropout2", "batchSize", "horizon", "beta", "flow_layers", "flow_type", "Iterations", "flow_hid_dim", "in_flow_layers"])
  for name in files:
     if not (name.startswith("test-estimates") and "Neural" in name and "SEPA" in name):
       continue
     with open(path+name, "r") as inFile:
       data = inFile.read().strip().split("\n")
       devLosses = [float(x) for x in data[1].split(" ")]
       if len(devLosses) == 1: # or (devLosses[-2] < devLosses[-3] if "WordsUD" not in data[0] else devLosses[-3] < devLosses[-4]):
          if not ("TOY" in name and len(devLosses) >= 10):
             print("Reject ",name, len(devLosses))
             continue
       iterations_number = len(devLosses)
       parameters = data[0].split(" ")
       script = parameters[0]
       parameters = parameters[1:]
       if "WordsUD" not in data[0]:
          parameters.insert(10, "0.0")
       assert len(parameters) in [15,17], parameters
       if len(parameters) == 15:
          parameters.append(512)
          parameters.append(2)
       assert len(parameters) == 17, (len(parameters), parameters)
       language, _, dropout1, emb_dim, rnn_dim, rnn_layers, lr, model, dropout2, batchSize, replaceWordsProbability, horizonHere, beta, flow_layers, flow_type, flow_hid_dim, in_flow_layers = tuple(parameters)
       assert replaceWordsProbability == "0.0", parameters
       #assert str(flow_layers2) == flow_layers, (flow_layers, flow_layers2)
       surprisalTable = [float(x) for x in data[2].split(" ")]
       memories = [float(x) for x in data[3].split(" ")]
       ee = float(data[4])
#       beta = data[0].split(" ")[-3]
       assert len(surprisalTable) == horizon, len(surprisalTable)
       avgSurpSecond = sum(surprisalTable[(horizon/2):])/(horizon/2)
       first = min([x for x in range(len(surprisalTable)) if surprisalTable[x] <= avgSurpSecond] + [horizon/2])
       upToFirst = sum(surprisalTable[:first])
       avg16 = surprisalTable[(horizon/2)]
       avg17 = sum(surprisalTable[(horizon/2):(horizon/2)+2])/2
       avg18 = sum(surprisalTable[(horizon/2):(horizon/2)+3])/3
       avg19 = sum(surprisalTable[(horizon/2):(horizon/2)+4])/4

       print >> outFile, "\t".join(map(str,[language, script, beta, ee, memories[-2], upToFirst, avgSurpSecond, avg16, avg17, avg18, avg19,  dropout1, emb_dim, rnn_dim, rnn_layers, lr, model, dropout2, batchSize, horizonHere, beta, flow_layers, flow_type, iterations_number, flow_hid_dim, in_flow_layers]))



