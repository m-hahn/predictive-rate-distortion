

findFrontier = function(ees, memories) {
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



