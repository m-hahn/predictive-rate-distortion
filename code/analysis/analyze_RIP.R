

library(tidyr)
library(dplyr)
library(ggplot2)


data = read.csv("../../results/results-nprd.tsv", sep="\t")
data$Horizon = 15
data = data %>% filter(Memories < UpperBound)
dataD = read.csv("../../results/results-oce.tsv", sep="\t")



dataU = data %>% filter(Language == "rip")
dataU$UncondEnt = 0.505
dataU = dataU %>% mutate(MiWithFut = Horizon*(UncondEnt-FutureSurp))
dataDU = dataD %>% filter(Language == "rip") %>% filter(Horizon <= 5)
dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 0.636, Horizon == 2 ~ 0.62, Horizon == 3 ~  0.598, Horizon == 4 ~ 0.581, Horizon == 5 ~ 0.568, Horizon == 6 ~ 0.556, Horizon == 7 ~ 0.545, Horizon == 8 ~ 0.536)) 
dataDU = dataDU %>% mutate(MiWithFut = Horizon*(UncondEnt-Surprisal))
dataU = rbind(dataDU %>% select(Language, Beta,Surprisal, Memory, Horizon, UncondEnt, MiWithFut) %>% mutate(Type="Ngrams"), dataU %>% select(Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural"))
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
dataDU = dataD %>% filter(Language == "rip")
dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 0.613, Horizon == 2 ~ 0.612, Horizon == 3 ~  0.594, Horizon == 4 ~ 0.573, Horizon == 5 ~ 0.556, Horizon == 6 ~ 0.541, Horizon == 7 ~ 0.527, Horizon == 8 ~ 0.517, Horizon == 15 ~ 0.47)) 
dataDU = dataDU %>% filter(Horizon <= 5)
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
# Plot curve as computed from p. 1328-9 in Marzen and Crutchfield (2019).
plot = plot + geom_line(data=data.frame(x=c(0, 0.6365, 0.8595025), y = c(0, 0.6365, 1.05492)), aes(x=x , y=y, group=NA, color=NA), size=4) 
plot = plot + theme(legend.position="none") 
plot = plot + geom_line(data=findFrontier(dataNU$EE, dataNU$Memory), aes(x=EE, y=Memories, group=NULL, color=NULL), colour="red", size=2.5)
plot = plot +    theme(    axis.text.x = element_text(size=20),
		           axis.text.y = element_text(size=20),
			   axis.title.x = element_text(size=25),
			   axis.title.y = element_text(size=25))
plot = plot + xlab("Predictiveness")
plot = plot + ylab("Rate")
ggsave("../figures/rip-info.pdf", plot=plot) 





