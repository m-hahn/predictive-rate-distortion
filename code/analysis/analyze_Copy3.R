# Create plots for the Copy3 Process.


data = read.csv("../../results/results-nprd.tsv", sep="\t")
library(tidyr)
library(dplyr)
library(ggplot2)
data$Horizon = 15
data = data %>% filter(EE <= Memories)



dataU = data %>% filter(Language == "repeat") %>% filter(Memories < 20) 

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() + geom_line(data=data.frame(x=c(0, 16.47), y = c(0, 16.47)), aes(x=x , y=y), size=2)
ggsave("../figures/repeat3-ee-mem.pdf", plot=plot)


dataU = dataU[order(dataU$EE),]
ees = dataU$EE
memories = dataU$Memories

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


plot = ggplot(data=data, aes(x=EE, y=Memories, alpha=0.5)) + theme_classic() + geom_line(data=data.frame(x=c(0, 16.47), y = c(0, 16.47)), aes(x=x , y=y), size=3)+ theme(legend.position="none")  + geom_point(data=dataU, aes(x=EE, y=Memories, alpha=0.5), colour="red") + geom_line(data=findFrontier(dataU$EE, dataU$Memories),aes(x=EE, y=Memories, alpha=0.5), colour="red", size=1.5)
ggsave("../figures/repeat3-ee-mem-frontier.pdf", plot=plot)



