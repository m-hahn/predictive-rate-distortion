# Was called runMemoryManyConfigs_Disc_SGD.py

import subprocess
import sys
import random

language = sys.argv[1]
algorithm = sys.argv[2] if len(sys.argv) > 2 else "ba" # Can be ba (Blahut-Arimoto) or sgd (Gradient Descent)

while True:
   args = {}
   args["language"] = language 
   args["horizon"] = random.choice([1,2,3])
   args["code_number"] = 100 
   args["dirichlet"] = random.choice([0.000001, 0.00001, 0.0001, 0.001, 0.01])
   args["beta"] = (0.4 * random.random())

   args = (" ".join(["--"+x+" "+str(y) for x, y in args.iteritems()])).split(" ") 
   subprocess.call(map(str,["./python27", "oce_"+algorithm+"_pos.py"] + args))
  
   
    
   
