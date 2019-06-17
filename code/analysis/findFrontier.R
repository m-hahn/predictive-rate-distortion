

findFrontier = function(ees, memories) {
   ees = pmax(ees, 0)
   memories = pmax(memories, 0)
   xs = c(0)
   ys = c(0)
#   slopes = c(0)
   lastIndex = 0
   while(lastIndex < length(ees)) {
       lastX = xs[1]
       lastY = ys[1]
        
       bestSlope = 10000000000
       bestIndex = length(ees)
       for(i in (1:length(ees))) {
           x = ees[i]
           if(x > lastX) {
              y = memories[i]
	      cat(y-lastY, "\n")
              slope = (y-lastY)/(x-lastX)
              if(slope < bestSlope) {
                  bestIndex = i
 	          bestSlope = slope
     	      }
#     	      cat(slope, "\n")
	   }
       }
       xs = c(ees[bestIndex], xs)
       ys = c(memories[bestIndex], ys)
 #      slopes = c(bestSlope, slopes)
       lastIndex = bestIndex
       cat(bestSlope,"\n")
   }
   cat(xs, "\n")
   return(data.frame(EE=xs, Memories=ys))
}




findFrontierOld = function(ees, memories) {
   ees = pmax(ees, 0)
   memories = pmax(memories, 0)
   xs = c(0)
   ys = c(0)
   slopes = c(0)
   for(i in (1:length(ees))) {
     x = ees[i]
     if(x < 0) {
   	  next
     }
     y = memories[i]
     
     while(TRUE) {
       lastX = xs[1]
       lastY = ys[1]
   
       slope = (y-lastY)/(x-lastX)
       cat(slope," ", y, lastY, x, lastX,  "..\n", sep=" ")
       if(slope >= slopes[1]) {
   	    xs = c(x, xs)
   	    ys = c(y, ys)
   	    slopes = c(slope, slopes)
   	    break
       } else {
   	    xs = xs[(2:length(xs))]
   	    ys = ys[(2:length(ys))]
   	    slopes = slopes[(2:length(slopes))]
       }
     }
   }
   return(data.frame(EE=xs, Memories=ys))
}



