# Visualize PCA of codes for RIP.


import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy as np
import numpy.linalg
import random
import sys

files = {2 : "../../results/RESULTS3920504", 3 : "/sailhome/mhahn/scr/RESULTS4383042"}
for modes, filename in files.iteritems():
    with open(filename, "r") as inFile:
       data = inFile.read().strip().split("\n")
       strings = [data[3*x].replace(" ","") for x in range(len(data)/3)]
       vectors = [np.asarray(map(float,data[3*x+1].split(" "))) for x in range(len(data)/3)]
       marginal = [float(data[3*x+2]) for x in range(len(data)/3)]
    
    states = []
    for i in range(len(strings)):
       string = strings[i][1:16]
       assert len(strings[i]) == 31
       assert len(string) == 15
       assert len(strings[i][16:]) == 15
       if "44" in string:
         pos = string.rfind("44")
         state = "C"
       elif "34" in string:
         pos = string.rfind("34")
         state = "B"
       else:
         print(string)
         state = "U"
         states.append(state)
         continue
       for j in range(pos+2, len(string)):
            letter = string[j]
            if state == "A":
                state = {"3" : "C", "4" : "B"}[letter]
            elif state == "B":
                state = {"3" : "C", "4" : "C"}[letter]
            elif state == "C":
                state = {"3" : "A"}[letter]
     #  print(string)
    #   print(state)
       states.append(state)
    

    
    vector_state = random.sample(zip(vectors, states, marginal), 5000)
    vectors = [x[0] for x in vector_state]
    states = [x[1] for x in vector_state]
    marginal = [x[2] for x in vector_state]
    length = len(vectors)
    colors = [{"A" : "red", "B" : "green", "C" : "blue", "U" : "black"}[x] for x in states]
    #print(colors)
    print(len(vectors))
    vectors = list(vectors)[:length]
    X = np.asarray(vectors)
    
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    pca.fit(X)
    Y = pca.transform(X)
    
    pylab.scatter(Y[:, 0], Y[:, 1], 20, color=colors)
    #pylab.scatter(Y[length/2:, 0], Y[length/2:, 1], 20, c="green")
    pylab.savefig('foo_pca_'+str(modes)+'.png') 
    pylab.show()
    
   
