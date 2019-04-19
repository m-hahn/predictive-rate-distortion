library(tidyr)
library(dplyr)
library(ggplot2)




data = read.csv("../../results/results-en-upos-neuralflow.tsv", sep="\t")
data$Horizon = 15
data = data %>% filter(EE <= Memories)
dataU = data %>% filter(Language == "repeat") %>% filter(Memories < 20)

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() + geom_line(data=data.frame(x=c(0, 16.47), y = c(0, 16.47)), aes(x=x , y=y), size=2)
ggsave("../figures/repeat3-info.pdf", plot=plot)


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


plot = ggplot(data=data, aes(x=EE, y=Memories)) 
plot = plot + theme_classic() 
plot = plot + geom_line(data=data.frame(x=c(0, 16.47), y = c(0, 16.47)), aes(x=x , y=y), size=4)
plot = plot + theme(legend.position="none")  
plot = plot + geom_point(data=dataU, aes(x=EE, y=Memories), colour="red") 
plot = plot + geom_line(data=findFrontier(dataU$EE, dataU$Memories),aes(x=EE, y=Memories), colour="red", size=2.5)
plot = plot +    theme(    axis.text.x = element_text(size=20),
		           axis.text.y = element_text(size=20),
			   axis.title.x = element_text(size=25),
			   axis.title.y = element_text(size=25))
plot = plot + xlab("Predictiveness")
plot = plot + ylab("Rate")
ggsave("../figures/repeat3-info.pdf", plot=plot)







