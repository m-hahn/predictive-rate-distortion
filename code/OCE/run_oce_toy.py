# Was called runMemoryManyConfigs_Disc_SGD_Toy.py

import subprocess
import sys
import random

language = sys.argv[1] # Can be even (Even Process) or rip (Random Insertaion Process)
algorithm = sys.argv[2] if len(sys.argv) > 2 else "ba" # Can be ba (Blahut-Arimoto) or sgd (Gradient Descent)

while True:
   args = {}
   args["language"] = language
   args["horizon"] = random.choice([1,2,3,4,5,6,7,8]) # or 15
   args["code_number"] = 100 #random.choice([100, 100, 1000, 1000, 10000])
   args["dirichlet"] = random.choice([0.000001, 0.00001, 0.0001, 0.001, 0.01, 1.0])
   args["beta"] = random.random()

   args = (" ".join(["--"+x+" "+str(y) for x, y in args.iteritems()])).split(" ") 
   subprocess.call(map(str,["./python27", "oce_"+algorithm+"_pos.py"] + args))
  
   
    
   
