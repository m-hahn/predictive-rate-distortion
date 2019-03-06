import subprocess



import random

while True:
   args = {}
#parser.add_argument("--language", type=str, dest="language")
#parser.add_argument("--horizon", type=int, dest="horizon")
#parser.add_argument("--code_number", type=int, dest="code_number"
#parser.add_argument("--beta", type=float, dest="beta")
#parser.add_argument("--dirichlet", type=float, dest="dirichlet")


   args["language"] = random.choice(["even", "rrxor","rip"]) # , "forget2_0_5b"

   args["horizon"] = random.choice([1,2,3,4,5,6,7,8]) # or 15
   args["code_number"] = 100 #random.choice([100, 100, 1000, 1000, 10000])
   args["dirichlet"] = random.choice([0.000001, 0.00001, 0.0001, 0.001, 0.01, 1.0])
   args["beta"] = random.random()

   args = (" ".join(["--"+x+" "+str(y) for x, y in args.iteritems()])).split(" ") 
   subprocess.call(map(str,["./python27", "zNgramIB_9_TOY.py"] + args))
  
   
    
   
