
    data = read.csv("~/CS_SCR/results-en-upos-neuralflow.tsv", sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
data$Horizon = 15

data = data %>% filter(Memories < UpperBound)

plot = ggplot(data, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

plot = ggplot(data, aes(x=avg16, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

    dataD = read.csv("~/CS_SCR/results-en-upos-discrete-sgd.tsv", sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)



dataU = data %>% filter(Language == "even")
dataU$UncondEnt = 0.505
dataU = dataU %>% mutate(MiWithFut = Horizon*(UncondEnt-FutureSurp))
dataDU = dataD %>% filter(Language == "even") %>% filter(Horizon <= 5)
dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 0.636, Horizon == 2 ~ 0.62, Horizon == 3 ~  0.598, Horizon == 4 ~ 0.581, Horizon == 5 ~ 0.568, Horizon == 6 ~ 0.556, Horizon == 7 ~ 0.545, Horizon == 8 ~ 0.536)) 
dataDU = dataDU %>% mutate(MiWithFut = Horizon*(UncondEnt-Surprisal))
dataU = rbind(dataDU %>% select(Language, Beta,Surprisal, Memory, Horizon, UncondEnt, MiWithFut) %>% mutate(Type="Ngrams"), dataU %>% select(Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural"))
dataU = dataU %>% filter(MiWithFut <= Memory)

d = dataU %>% filter(Type == "Neural") %>% select(MiWithFut, Memory)
D = d[order(-d$MiWithFut,d$Memory,decreasing=FALSE),]
front = D[which(!duplicated(cummin(D$Memory))),]

plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlim(0,NA) + ylim(0, 1.1) + geom_line(data=data.frame(x=c(0, 0.6365), y = c(0, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none") + geom_line(data=front, aes(x=MiWithFut, y=Memory, group="neural", color="neural"), size=1.5)
#ggsave("figures/even-info.pdf", plot=plot) 
plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + ylim(0, 1.1) + xlim(0,1) + geom_line(data=data.frame(x=c(0, 1.0), y = c(0.6305, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none")
#ggsave("figures/even-beta-mem.pdf", plot=plot) 

dataU = dataU %>% mutate(Objective = Horizon*Surprisal + Beta * Memory)
plot = ggplot(dataU, aes(x=Beta, y=Objective, group=Type, color=Type)) + geom_point()+ theme_classic() + ylim(0, 2) + xlim(0,1) # + geom_line(data=data.frame(x=c(0, 1.0), y = c(0.6305, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none")
d = dataU %>% filter(Type == "Neural") %>% mutate(BetaD = round(Beta*50))
d2 = d %>% group_by(BetaD) %>% summarize(Objective=min(Objective))
d = merge(d,d2, by=c("BetaD", "Objective"))
plot = ggplot(d, aes(x=Beta, y=Memory)) + geom_point()+ theme_classic() + ylim(0, 2) + xlim(0,1) # + geom_line(data=data.frame(x=c(0, 1.0), y = c(0.6305, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none")
plot = ggplot(d, aes(x=MiWithFut, y=Memory)) + geom_point()+ theme_classic() + ylim(0, 2) + xlim(0,1) # + geom_line(data=data.frame(x=c(0, 1.0), y = c(0.6305, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none")




dataU = data %>% filter(Language == "even") %>% mutate(Type = "Neural")
plot = ggplot(dataU, aes(x=avg16, y=Memories, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memories, y=avg16, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memories, group=Type, color=Type)) + geom_point()+ theme_classic() 



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







dataU = data %>% filter(Language == "even")
dataU$UncondEnt = 0.505
dataU = dataU %>% mutate(MiWithFut = Horizon*(UncondEnt-FutureSurp))

dataDU = dataD %>% filter(Language == "even") %>% filter(Horizon <= 5)
dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 0.636, Horizon == 2 ~ 0.62, Horizon == 3 ~  0.598, Horizon == 4 ~ 0.581, Horizon == 5 ~ 0.568, Horizon == 6 ~ 0.556, Horizon == 7 ~ 0.545, Horizon == 8 ~ 0.536)) 
dataDU = dataDU %>% mutate(MiWithFut = Horizon*(UncondEnt-Surprisal))
dataU = rbind(dataDU %>% select(Language, Beta,Surprisal, Memory, Horizon, UncondEnt, MiWithFut) %>% mutate(Type="Ngrams", EE=NA), dataU %>% select(EE, Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural"))
dataU = dataU %>% filter(MiWithFut <= Memory)

d = dataU %>% filter(Type == "Neural") %>% select(MiWithFut, Memory)
D = d[order(-d$MiWithFut,d$Memory,decreasing=FALSE),]
front = D[which(!duplicated(cummin(D$Memory))),]

dataNU = dataU %>% filter(Type == "Neural")
dataNU = dataNU[order(dataNU$EE),]


plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type))
plot = plot + geom_point()+ theme_classic() + xlim(0,NA) + ylim(0, 1.1) 
plot = plot + geom_line(data=data.frame(x=c(0, 0.6365), y = c(0, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=4) 
plot = plot + theme(legend.position="none") 
plot = plot + geom_line(data=findFrontier(dataNU$EE, dataNU$Memory), aes(x=EE, y=Memories, group=NULL, color=NULL), colour="red", size=2.5)
plot = plot +    theme(    axis.text.x = element_text(size=20),
		           axis.text.y = element_text(size=20),
			   axis.title.x = element_text(size=25),
			   axis.title.y = element_text(size=25))
plot = plot + xlab("Predictiveness")
plot = plot + ylab("Rate")
ggsave("figures/even-info.pdf", plot=plot) 







