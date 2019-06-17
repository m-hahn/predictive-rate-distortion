

library(tidyr)
library(dplyr)
library(ggplot2)


data = read.csv("../../results/results-nprd.tsv", sep="\t") %>% filter(flow_layers>0)
data$Horizon = data$horizon/2
data = data %>% filter(Memories < UpperBound)


dataU = data %>% filter(Language == "rip")
dataU$UncondEnt = 0.505
dataU = dataU %>% mutate(MiWithFut = Horizon*(UncondEnt-FutureSurp))
dataU = dataU %>% select(Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural")
dataU = dataU %>% filter(MiWithFut <= Memory)

d = dataU %>% filter(Type == "Neural") %>% select(MiWithFut, Memory)
D = d[order(-d$MiWithFut,d$Memory,decreasing=FALSE),]
front = D[which(!duplicated(cummin(D$Memory))),]


dataU = dataU %>% mutate(Objective = Horizon*Surprisal + Beta * Memory)
d = dataU %>% filter(Type == "Neural") %>% mutate(BetaD = round(Beta*50))
d2 = d %>% group_by(BetaD) %>% summarize(Objective=min(Objective))
d = merge(d,d2, by=c("BetaD", "Objective"))




dataU = data %>% filter(Language == "rip") %>% mutate(Type = "Neural")

source("findFrontier.R")




dataU = data %>% filter(Language == "rip")
dataU$UncondEnt = 0.47
dataU = dataU %>% mutate(MiWithFut = EE) #Horizon*(UncondEnt-FutureSurp))

dataU = dataU %>% select(EE, Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural")
dataU = dataU %>% filter(MiWithFut <= Memory)

d = dataU %>% filter(Type == "Neural") %>% select(MiWithFut, Memory)
D = d[order(-d$MiWithFut,d$Memory,decreasing=FALSE),]
front = D[which(!duplicated(cummin(D$Memory))),]

dataNU = dataU %>% filter(Type == "Neural")
dataNU = dataNU[order(dataNU$EE),]

dataU$HorizonF = as.character(dataU$Horizon)

plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=HorizonF, color=HorizonF))
plot = plot + geom_point()+ theme_classic() + xlim(0,NA) + ylim(0, 1.2)
# Plot curve as computed from p. 1328-9 in Marzen and Crutchfield (2019).
plot = plot + geom_line(data=data.frame(x=c(0, 0.6365, 0.8595025), y = c(0, 0.6365, 1.05492)), aes(x=x , y=y, group=NA, color=NA), size=4) 
plot = plot + theme(legend.position="none") 
horizons = c(5, 10, 15)
colors = c("blue", "red", "green")
for(i in (1:3)) {
   horizon = horizons[i]
   plot = plot + geom_line(data=findFrontier((dataNU %>% filter(Horizon==horizon))$EE, (dataNU %>% filter(Horizon==horizon))$Memory) %>% mutate(HorizonF=as.character(horizon)), aes(x=EE, y=Memories, group=HorizonF, color=HorizonF), size=2.5) # colour=colors[i], 
}
plot = plot +    theme(    axis.text.x = element_text(size=20),
		           axis.text.y = element_text(size=20),
			   axis.title.x = element_text(size=25),
			   axis.title.y = element_text(size=25))
plot = plot + xlab("Predictiveness")
plot = plot + ylab("Rate")
ggsave("../figures/rip-info-varyM.pdf", plot=plot) 





