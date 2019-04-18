
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
ggsave("figures/even-info.pdf", plot=plot) 
plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + ylim(0, 1.1) + xlim(0,1) + geom_line(data=data.frame(x=c(0, 1.0), y = c(0.6305, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none")
ggsave("figures/even-beta-mem.pdf", plot=plot) 

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



dataU = data %>% filter(Language == "rrxor")
dataU$UncondEnt = 0.84
dataU = dataU %>% mutate(MiWithFut = Horizon*(UncondEnt-FutureSurp))
dataDU = dataD %>% filter(Language == "rrxor")
dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 0.955, Horizon == 2 ~ 0.953, Horizon == 3 ~  0.940, Horizon == 4 ~ 0.929, Horizon == 5 ~ 0.918, Horizon == 6 ~ 0.909, Horizon == 7 ~ 0.899)) 
dataDU = dataDU %>% mutate(MiWithFut = Horizon*(UncondEnt-Surprisal))
dataU = rbind(dataDU %>% select(Language, Beta,Surprisal, Memory, Horizon, UncondEnt, MiWithFut) %>% mutate(Type="Ngrams"), dataU %>% select(Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural"))
dataU = dataU %>% filter(MiWithFut <= Memory)
plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlim(0, 1.5) + ylim(0, 5) + geom_line(data=data.frame(x=c(0, 0.6365), y = c(0, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none") 
ggsave("figures/rrxor-info.pdf", plot=plot) 
plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + ylim(0, 1.1) + xlim(0,1) + geom_line(data=data.frame(x=c(0, 1.0), y = c(0.6305, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none")
ggsave("figures/rrxor-beta-mem.pdf", plot=plot) 

dataU = data %>% filter(Language == "rrxor") %>% mutate(Type = "Neural")
plot = ggplot(dataU, aes(x=avg16, y=Memories, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memories, y=avg16, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memories, group=Type, color=Type)) + geom_point()+ theme_classic() 





dataU = data %>% filter(Language == "rip")
dataU$UncondEnt = 0.47
dataU = dataU %>% mutate(MiWithFut = EE) #Horizon*(UncondEnt-FutureSurp))
dataDU = dataD %>% filter(Language == "rip")
dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 0.613, Horizon == 2 ~ 0.612, Horizon == 3 ~  0.594, Horizon == 4 ~ 0.573, Horizon == 5 ~ 0.556, Horizon == 6 ~ 0.541, Horizon == 7 ~ 0.527, Horizon == 8 ~ 0.517, Horizon == 15 ~ 0.47)) 
dataDU = dataDU %>% filter(Horizon <= 5)
dataDU = dataDU %>% mutate(MiWithFut = Horizon*(UncondEnt-Surprisal))
dataU = rbind(dataDU %>% select(Language, Beta,Surprisal, Memory, Horizon, UncondEnt, MiWithFut) %>% mutate(Type="Ngrams"), dataU %>% select(Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural"))
dataU = dataU %>% filter(MiWithFut <= Memory)

d = dataU %>% filter(Type == "Neural") %>% select(MiWithFut, Memory)
D = d[order(-d$MiWithFut,d$Memory,decreasing=FALSE),]
front = D[which(!duplicated(cummin(D$Memory))),]

plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlim(0,NA) + ylim(0, 1.1) + geom_line(data=data.frame(x=c(0, 0.6365, 0.8595025), y = c(0, 0.6365, 1.039721)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none")  + geom_line(data=front, aes(x=MiWithFut, y=Memory, group="neural", color="neural"), size=1.5)
ggsave("figures/rip-info.pdf", plot=plot) 
plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + ylim(0, 1.1) + xlim(0,1) + geom_line(data=data.frame(x=c(0, 0.5, 0.5, 1.0), y = c(1.039721,1.039721, 0.6365, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none")
ggsave("figures/rip-beta-mem.pdf", plot=plot) 


dataU = dataU %>% mutate(Objective = Horizon*Surprisal + Beta * Memory)
plot = ggplot(dataU, aes(x=Beta, y=Objective, group=Type, color=Type)) + geom_point()+ theme_classic() + ylim(0, 2) + xlim(0,1) # + geom_line(data=data.frame(x=c(0, 1.0), y = c(0.6305, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none")
d = dataU %>% filter(Type == "Neural") %>% mutate(BetaD = round(Beta*50))
d2 = d %>% group_by(BetaD) %>% summarize(Objective=min(Objective))
d = merge(d,d2, by=c("BetaD", "Objective"))
plot = ggplot(d, aes(x=Beta, y=Memory)) + geom_point()+ theme_classic() + ylim(0, 2) + xlim(0,1) # + geom_line(data=data.frame(x=c(0, 1.0), y = c(0.6305, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none")
plot = ggplot(d, aes(x=MiWithFut, y=Memory)) + geom_point()+ theme_classic() + ylim(0, 2) + xlim(0,1) # + geom_line(data=data.frame(x=c(0, 1.0), y = c(0.6305, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=2) + theme(legend.position="none")


dataU = data %>% filter(Language == "rip") %>% mutate(Type = "Neural")
plot = ggplot(dataU, aes(x=avg16, y=Memories, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memories, y=avg16, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memories, group=Type, color=Type)) + geom_point()+ theme_classic() 





#
#dataU = rbind(dataD %>% select(Language, Beta,Surprisal, Memory, Horizon) %>% mutate(Type="Ngrams"), data %>% select(Language, Beta,FutureSurp, Memories, Horizon) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural"))
#dataU = dataU %>% filter(Language == "forget2_0_5b")
#plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
#plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
#plot = ggplot(dataU, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
#
#
#dataU = data %>% filter(Language == "forget2_0_5b") %>% mutate(Type = "Neural")
#plot = ggplot(dataU, aes(x=avg16, y=Memories, group=Type, color=Type)) + geom_point()+ theme_classic() 
#plot = ggplot(dataU, aes(x=Memories, y=avg16, group=Type, color=Type)) + geom_point()+ theme_classic() 
#plot = ggplot(dataU, aes(x=Beta, y=Memories, group=Type, color=Type)) + geom_point()+ theme_classic() 




dataU = data %>% filter(Language == "Japanese")
dataU$UncondEnt = 1.895
dataU = dataU %>% mutate(MiWithFut = EE) #Horizon*(UncondEnt-FutureSurp))
#plot = ggplot(dataU, aes(x=Beta, y=Memories/EE)) + geom_point()+ theme_classic()  + xlim(0, 0.4)
dataDU = dataD %>% filter(Language == "Japanese")
dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 2.563, Horizon == 2 ~ 2.18, Horizon == 3 ~  2.15)) 
dataDU = dataDU %>% mutate(MiWithFut = Horizon*(UncondEnt-Surprisal))
dataU = rbind(dataDU %>% select(Language, Beta,Surprisal, Memory, Horizon, UncondEnt, MiWithFut) %>% mutate(Type="Ngrams"), dataU %>% select(Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural"))
dataU = dataU %>% filter(MiWithFut <= Memory)
plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA)
ggsave("figures/japanese-info.pdf", plot=plot) 
plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(0, 0.4) + ylim(0, NA)
ggsave("figures/japanese-beta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=log(Beta), y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/japanese-logbeta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=-log(Beta), y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/japanese-nlogbeta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=log(Beta), y=MiWithFut, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/japanese-logbeta-ee.pdf", plot=plot)
plot = ggplot(dataU, aes(x=-log(Beta), y=MiWithFut, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/japanese-nlogbeta-ee.pdf", plot=plot)




dataU = data %>% filter(Language == "English") %>% filter(model != "REAL")
dataU$UncondEnt = 1.895
dataU = dataU %>% mutate(MiWithFut = EE) #Horizon*(UncondEnt-FutureSurp))
#plot = ggplot(dataU, aes(x=Beta, y=Memories/EE)) + geom_point()+ theme_classic()  + xlim(0, 0.4)
#dataDU = dataD %>% filter(Language == "English")
#dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 2.563, Horizon == 2 ~ 2.18, Horizon == 3 ~  2.15)) 
#dataDU = dataDU %>% mutate(MiWithFut = Horizon*(UncondEnt-Surprisal))
dataU = dataU %>% select(Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut, model, avg16) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural")
dataU = dataU %>% filter(MiWithFut <= Memory)

plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=model)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + xlim(0, NA) + ylim(0, NA)
plot = ggplot(dataU, aes(x=Memory, y=avg16, group=Type, color=model)) + geom_point()+ theme_classic() + xlab("I[Z, Past]") + ylab("Surprisal") + xlim(0, NA)

plot = ggplot(dataU %>% filter(model %in% c("RANDOM_BY_TYPE", "REAL_REAL")), aes(x=Memory, y=avg16, group=model, color=model)) + geom_smooth(se=F)+ theme_classic() + xlab("Memory") + ylab("Future Surprisal") + xlim(0, NA)


plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=model)) + geom_point()+ theme_classic() + xlab("I[Z, Past]") + ylab("Surprisal") + xlim(0, NA)


plot = ggplot(dataU, aes(x=MiWithFut, y=avg16, group=Type, color=model)) + geom_point()+ theme_classic() + xlab("I[Z, Past]") + ylab("Surprisal") + xlim(0, NA)

plot = ggplot(dataU, aes(x=Surprisal, y=Memory-MiWithFut, group=Type, color=model)) + geom_point()+ theme_classic() + xlab("Surprisal") + ylab("Crypticity") 


dataU = data %>% filter(Language == "English") %>% filter(model == "REAL")
dataU$UncondEnt = 1.895
dataU = dataU %>% mutate(MiWithFut = EE) #Horizon*(UncondEnt-FutureSurp))
#plot = ggplot(dataU, aes(x=Beta, y=Memories/EE)) + geom_point()+ theme_classic()  + xlim(0, 0.4)
dataDU = dataD %>% filter(Language == "English")
dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 2.563, Horizon == 2 ~ 2.18, Horizon == 3 ~  2.15)) 
dataDU = dataDU %>% mutate(MiWithFut = Horizon*(UncondEnt-Surprisal))
dataU = rbind(dataDU %>% select(Language, Beta,Surprisal, Memory, Horizon, UncondEnt, MiWithFut) %>% mutate(Type="Ngrams") %>% mutate(model="REAL"), dataU %>% select(Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut, model) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural"))
dataU = dataU %>% filter(MiWithFut <= Memory)

plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=model)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + xlim(0, NA) + ylim(0, NA)
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=model)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("Surprisal") + xlim(0, NA)


plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA)
ggsave("figures/english-info.pdf", plot=plot) 
plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(0, 0.4) + ylim(0, NA)
ggsave("figures/english-beta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=log(Beta), y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/english-logbeta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=-log(Beta), y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/english-nlogbeta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=log(Beta), y=MiWithFut, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/english-logbeta-ee.pdf", plot=plot)
plot = ggplot(dataU, aes(x=-log(Beta), y=MiWithFut, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/english-nlogbeta-ee.pdf", plot=plot)

#xs = (1:1000)/1000
#plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(x=(1-(xs/0.1)^(-0.1)), y=xs), aes(x=x, y=y, group=NULL, color=NULL))
#
#
#D = dataU %>% filter(Type=="Neural") %>% select(MiWithFut, Memory)
#write.csv(D, file="neural-pos.csv")
#D= D %>% mutate(MemoryApprox = round(Memory*100)/100) %>% group_by(MemoryApprox) %>% summarise(MiWithFut = max(MiWithFut))
#xs = D$MiWithFutApprox
#ys = D$Memory
#alpha = 1.0
#beta = 1.0
#gamma = -0.3



dataU1 = rbind(dataD %>% filter(Horizon == 1) %>% select(Language, Beta,Surprisal, Memory) %>% mutate(Type="Ngrams"), data %>% select(Language, Beta,avg16, Memories) %>% rename(Surprisal=avg16, Memory=Memories) %>% mutate(Type="Neural"))
dataU1 = dataU1 %>% filter(Language == "English")
plot = ggplot(dataU1, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlab("Surprisal on First Future Word") + ylab("I[Z, Past]") + theme(legend.position="none") 
ggsave("figures/english-mem-surp-1.pdf", plot=plot)
plot = ggplot(dataU1, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU1, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 


dataU2 = rbind(dataD %>% filter(Horizon == 2) %>% select(Language, Beta,Surprisal, Memory) %>% mutate(Type="Ngrams"), data %>% select(Language, Beta,avg17, Memories) %>% rename(Surprisal=avg17, Memory=Memories) %>% mutate(Type="Neural"))
dataU2 = dataU2 %>% filter(Language == "English")
plot = ggplot(dataU2, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU2, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU2, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 



dataU3 = rbind(dataD %>% filter(Horizon == 3) %>% select(Language, Beta,Surprisal, Memory) %>% mutate(Type="Ngrams"), data %>% select(Language, Beta,avg18, Memories) %>% rename(Surprisal=avg18, Memory=Memories) %>% mutate(Type="Neural"))
dataU3 = dataU3 %>% filter(Language == "English")
plot = ggplot(dataU3, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU3, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU3, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 




dataU = data %>% filter(Language == "PTB")
dataU$UncondEnt = 1.895
dataU = dataU %>% mutate(MiWithFut = EE) #Horizon*(UncondEnt-FutureSurp))
#plot = ggplot(dataU, aes(x=Beta, y=Memories/EE)) + geom_point()+ theme_classic()  + xlim(0, 0.4)
dataU = rbind(dataU %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural"))
dataU = dataU %>% filter(MiWithFut <= Memory)
plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA)
plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() + ylim(4,6) 
plot = ggplot(dataU, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(0, 0.4) + ylim(0, NA)
plot = ggplot(dataU, aes(x=log(Beta), y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none")
plot = ggplot(dataU, aes(x=log(Beta), y=MiWithFut, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none")







    dataD = read.csv("~/CS_SCR/results-en-upos-discrete-sgd.tsv", sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)





dataDU = dataD %>% filter(Language %in% c("English", "EnglishRANDOM_BY_TYPE"))
dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 2.563, Horizon == 2 ~ 2.18, Horizon == 3 ~  2.15)) 
dataDU = dataDU %>% mutate(MiWithFut = Horizon*(UncondEnt-Surprisal))
dataU = rbind(dataDU %>% select(Language, Beta,Surprisal, Memory, Horizon, UncondEnt, MiWithFut) %>% mutate(Type="Ngrams"))
dataU = dataU %>% filter(MiWithFut <= Memory)
plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Language, color=Language)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA)

plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Language, color=Language)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + facet_wrap(~Horizon)


ggsave("figures/random-english-info.pdf", plot=plot) 
plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Language, color=Language)) + geom_point()+ theme_classic()  + facet_wrap(~Horizon)
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Language, color=Language)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memory, group=Language, color=Language)) + geom_point()+ theme_classic()  + xlim(0, 0.4) + ylim(0, NA)
ggsave("figures/random-english-beta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=log(Beta), y=Memory, group=Language, color=Language)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/random-english-logbeta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=-log(Beta), y=Memory, group=Language, color=Language)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/random-english-nlogbeta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=log(Beta), y=MiWithFut, group=Language, color=Language)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/random-english-logbeta-ee.pdf", plot=plot)
plot = ggplot(dataU, aes(x=-log(Beta), y=MiWithFut, group=Language, color=Language)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/random-english-nlogbeta-ee.pdf", plot=plot)

#xs = (1:1000)/1000
#plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(x=(1-(xs/0.1)^(-0.1)), y=xs), aes(x=x, y=y, group=NULL, color=NULL))
#
#
#D = dataU %>% filter(Type=="Neural") %>% select(MiWithFut, Memory)
#write.csv(D, file="neural-pos.csv")
#D= D %>% mutate(MemoryApprox = round(Memory*100)/100) %>% group_by(MemoryApprox) %>% summarise(MiWithFut = max(MiWithFut))
#xs = D$MiWithFutApprox
#ys = D$Memory
#alpha = 1.0
#beta = 1.0
#gamma = -0.3



dataU1 = rbind(dataD %>% filter(Horizon == 1) %>% select(Language, Beta,Surprisal, Memory) %>% mutate(Type="Ngrams"), data %>% select(Language, Beta,avg16, Memories) %>% rename(Surprisal=avg16, Memory=Memories) %>% mutate(Type="Neural"))
dataU1 = dataU1 %>% filter(Language == "EnglishRANDOM_BY_TYPE")
plot = ggplot(dataU1, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlab("Surprisal on First Future Word") + ylab("I[Z, Past]") + theme(legend.position="none") 
ggsave("figures/random-english-mem-surp-1.pdf", plot=plot)
plot = ggplot(dataU1, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU1, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 


dataU2 = rbind(dataD %>% filter(Horizon == 2) %>% select(Language, Beta,Surprisal, Memory) %>% mutate(Type="Ngrams"), data %>% select(Language, Beta,avg17, Memories) %>% rename(Surprisal=avg17, Memory=Memories) %>% mutate(Type="Neural"))
dataU2 = dataU2 %>% filter(Language == "EnglishRANDOM_BY_TYPE")
plot = ggplot(dataU2, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU2, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU2, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 



dataU3 = rbind(dataD %>% filter(Horizon == 3) %>% select(Language, Beta,Surprisal, Memory) %>% mutate(Type="Ngrams"), data %>% select(Language, Beta,avg18, Memories) %>% rename(Surprisal=avg18, Memory=Memories) %>% mutate(Type="Neural"))
dataU3 = dataU3 %>% filter(Language == "EnglishRANDOM_BY_TYPE")
plot = ggplot(dataU3, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU3, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU3, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 



dataDU = dataD %>% filter(Language %in% c("English", "BackwardsEnglish"))
dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 2.563, Horizon == 2 ~ 2.18, Horizon == 3 ~  2.15)) 
dataDU = dataDU %>% mutate(MiWithFut = Horizon*(UncondEnt-Surprisal))
dataU = rbind(dataDU %>% select(Language, Beta,Surprisal, Memory, Horizon, UncondEnt, MiWithFut) %>% mutate(Type="Ngrams"))
dataU = dataU %>% filter(MiWithFut <= Memory)
plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Language, color=Language)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA)
ggsave("figures/backwards-english-info.pdf", plot=plot) 
plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Language, color=Language)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Language, color=Language)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memory, group=Language, color=Language)) + geom_point()+ theme_classic()  + xlim(0, 0.4) + ylim(0, NA)
ggsave("figures/backwards-english-beta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=log(Beta), y=Memory, group=Language, color=Language)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/backwards-english-logbeta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=-log(Beta), y=Memory, group=Language, color=Language)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/backwards-english-nlogbeta-mem.pdf", plot=plot)
plot = ggplot(dataU, aes(x=log(Beta), y=MiWithFut, group=Language, color=Language)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/backwards-english-logbeta-ee.pdf", plot=plot)
plot = ggplot(dataU, aes(x=-log(Beta), y=MiWithFut, group=Language, color=Language)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + ylim(0, NA) + theme(legend.position="none")
ggsave("figures/backwards-english-nlogbeta-ee.pdf", plot=plot)

#xs = (1:1000)/1000
#plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(x=(1-(xs/0.1)^(-0.1)), y=xs), aes(x=x, y=y, group=NULL, color=NULL))
#
#
#D = dataU %>% filter(Type=="Neural") %>% select(MiWithFut, Memory)
#write.csv(D, file="neural-pos.csv")
#D= D %>% mutate(MemoryApprox = round(Memory*100)/100) %>% group_by(MemoryApprox) %>% summarise(MiWithFut = max(MiWithFut))
#xs = D$MiWithFutApprox
#ys = D$Memory
#alpha = 1.0
#beta = 1.0
#gamma = -0.3



dataU1 = rbind(dataD %>% filter(Horizon == 1) %>% select(Language, Beta,Surprisal, Memory) %>% mutate(Type="Ngrams"), data %>% select(Language, Beta,avg16, Memories) %>% rename(Surprisal=avg16, Memory=Memories) %>% mutate(Type="Neural"))
dataU1 = dataU1 %>% filter(Language == "BackwardsEnglish")
plot = ggplot(dataU1, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlab("Surprisal on First Future Word") + ylab("I[Z, Past]") + theme(legend.position="none") 
ggsave("figures/backwards-english-mem-surp-1.pdf", plot=plot)
plot = ggplot(dataU1, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU1, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 


dataU2 = rbind(dataD %>% filter(Horizon == 2) %>% select(Language, Beta,Surprisal, Memory) %>% mutate(Type="Ngrams"), data %>% select(Language, Beta,avg17, Memories) %>% rename(Surprisal=avg17, Memory=Memories) %>% mutate(Type="Neural"))
dataU2 = dataU2 %>% filter(Language == "BackwardsEnglish")
plot = ggplot(dataU2, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU2, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU2, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 



dataU3 = rbind(dataD %>% filter(Horizon == 3) %>% select(Language, Beta,Surprisal, Memory) %>% mutate(Type="Ngrams"), data %>% select(Language, Beta,avg18, Memories) %>% rename(Surprisal=avg18, Memory=Memories) %>% mutate(Type="Neural"))
dataU3 = dataU3 %>% filter(Language == "BackwardsEnglish")
plot = ggplot(dataU3, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU3, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU3, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 






    data = read.csv("~/CS_SCR/results-en-upos-neuralflow.tsv", sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
data$Horizon = 15
data = data %>% filter(EE <= Memories)


# also consider
# https://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf

dataU = data %>% filter(Language == "repeat2") %>% filter(Memories < 15)

#data = data %>% filter(Memories < UpperBound)
#data = data %>% mutate(Objective = Horizon * FutureSurp + Beta * Memories)

plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

plot = ggplot(dataU, aes(x=avg16, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() + geom_line(data=data.frame(x=c(0, 10.39), y = c(0, 10.39)), aes(x=x , y=y), size=2)
ggsave("figures/repeat2-ee-mem.pdf", plot=plot)



dataU = data %>% filter(Language == "repeat") %>% filter(Memories < 20)

#data = data %>% filter(Memories < UpperBound)
#data = data %>% mutate(Objective = Horizon * FutureSurp + Beta * Memories)

plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_line(data=data.frame(x=c(1.1, 0), y = c(0, 16.47)), aes(x=x , y=y), size=2)


plot = ggplot(dataU, aes(x=avg16, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() + geom_line(data=data.frame(x=c(0, 16.47), y = c(0, 16.47)), aes(x=x , y=y), size=2)
ggsave("figures/repeat3-ee-mem.pdf", plot=plot)


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
ggsave("figures/repeat3-ee-mem-frontier.pdf", plot=plot)





data = read.csv("~/CS_SCR/results-en-upos-neuralflow-test.tsv", sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
data$Horizon = 15

data$PastSurp = 15*data$FutureSurp+data$EE

#data = data %>% filter(Memories < UpperBound) #can also take UpperBound2
data = data %>% mutate(Objective = Horizon * FutureSurp + Beta * Memories)

data = data %>% filter(model != "REVERSE")

dataU = data %>% filter(Language == "PTB")
plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-surp-mem.pdf")

plot = ggplot(dataU, aes(x=avg16, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-16-mem.pdf")
plot = ggplot(dataU, aes(x=avg17, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg18, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg19, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

x = 3.2 * (1:100)/100
plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1.0*x^2*exp(1.3/(3.9-x))), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-ee-mem.pdf")


x = (1:500)/1000
plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-log(x))^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-beta-mem.pdf")

x = -6*(10:95)/100
plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-x)^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-logbeta-mem.pdf")

x = 6*(10:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.0*(x)^1.9), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/en-words-nlogbeta-mem-fitted.pdf")





x = -6*(10:95)/100
# this one looks good
plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=0.012*(3.5-(-x-0.1)^(-1.5))^(4.5)), aes(x=x, y=y), color="red") + ylim(0, NA)

plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA) + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-logbeta-ee.pdf")


#plot = ggplot(dataU, aes(x=log(Beta), y=EE, alpha=0.5)) + geom_point()+ theme_classic() + geom_point(data=data.frame(x=x, y=1.3-(-x)^1/(-x-0.1)^2), aes(x=x, y=y), color="red")
#ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=(3.5-(-x-0.1)^(-2))^(1.5)), aes(x=x, y=y), color="red") + ylim(0, NA)





dataU = data %>% filter(Language == "LDC95T8")
dataU$BeatsBound = (dataU$Memories < dataU$UpperBound)

plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-surp-mem.pdf")

plot = ggplot(dataU, aes(x=avg16, y=Memories, color=BeatsBound)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-16-mem.pdf")
plot = ggplot(dataU, aes(x=avg17, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg18, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg19, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

x = 3.2 * (1:100)/100
plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1.0*x^2*exp(1.3/(3.9-x))), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-ee-mem.pdf")


plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5, color=BeatsBound)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")


plot = ggplot(dataU, aes(x=EE, y=Memories, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1)) 


x = (1:500)/1000
plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-log(x))^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-beta-mem.pdf")

x = -6*(10:95)/100
plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-x)^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, color=BeatsBound)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-logbeta-mem.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")


plot = ggplot(dataU, aes(x=log(Beta), y=Objective, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1)) 


x = 6*(10:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1*(x)^2), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/LDC95T8-words-nlogbeta-mem-fitted.pdf")



x = -6*(10:95)/100
# this one looks good
plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=0.012*(3.5-(-x-0.1)^(-1.5))^(4.5)), aes(x=x, y=y), color="red") + ylim(0, NA)

plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-logbeta-ee.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=EE, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))











dataU = data %>% filter(Language == "LDC2012T05")
dataU$BeatsBound = (dataU$Memories < dataU$UpperBound)

plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-surp-mem.pdf")


plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, color=BeatsBound)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")


plot = ggplot(dataU, aes(x=avg16, y=Memories, color=BeatsBound)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-16-mem.pdf")
plot = ggplot(dataU, aes(x=avg17, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg18, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg19, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

x = 3.2 * (1:100)/100
plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1.0*x^2*exp(1.3/(3.9-x))), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-ee-mem.pdf")


plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5, color=BeatsBound)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")


plot = ggplot(dataU, aes(x=EE, y=Memories, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1)) 


x = (1:500)/1000
plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-log(x))^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-beta-mem.pdf")

x = -6*(10:95)/100
plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-x)^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, color=BeatsBound)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-logbeta-mem.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, color=BeatsBound)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")


plot = ggplot(dataU, aes(x=log(Beta), y=Objective, color=BeatsBound)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")



x = 6*(10:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1*(x)^2), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/LDC2012T05-words-nlogbeta-mem-fitted.pdf")



x = -6*(10:95)/100
# this one looks good
plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=0.012*(3.5-(-x-0.1)^(-1.5))^(4.5)), aes(x=x, y=y), color="red") + ylim(0, NA)

plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-logbeta-ee.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=EE, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))









dataU = data %>% filter(Language == "Arabic")
plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-surp-mem.pdf")

dataU$BeatsBound = (dataU$Memories < dataU$UpperBound)

plot = ggplot(dataU, aes(x=avg16, y=Memories, color=BeatsBound, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-16-mem.pdf")
plot = ggplot(dataU, aes(x=avg17, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg18, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg19, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

x = 3.2 * (1:100)/100
plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1.0*x^2*exp(1.3/(3.9-x))), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=EE, y=Memories, color=BeatsBound, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-ee-mem.pdf")


plot = ggplot(dataU, aes(x=EE, y=Memories, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1)) 


x = (1:500)/1000
plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-log(x))^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-beta-mem.pdf")

x = -6*(10:95)/100
plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-x)^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, color=BeatsBound, alpha=0.5)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-logbeta-mem.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")

x = 6*(10:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1*(x)^2), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/ar-words-nlogbeta-mem-fitted.pdf")



x = -6*(10:95)/100
# this one looks good
plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=0.012*(3.5-(-x-0.1)^(-1.5))^(4.5)), aes(x=x, y=y), color="red") + ylim(0, NA)

plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-logbeta-ee.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=EE, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))











dataU = data %>% filter(Language == "Japanese")
plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ja-words-surp-mem.pdf")

plot = ggplot(dataU, aes(x=avg16, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ja-words-16-mem.pdf")
plot = ggplot(dataU, aes(x=avg17, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg18, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg19, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

x = 3.2 * (1:100)/100
plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1.0*x^2*exp(1.3/(3.9-x))), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ja-words-ee-mem.pdf")


plot = ggplot(dataU, aes(x=EE, y=Memories, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1)) 


x = (1:500)/1000
plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-log(x))^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ja-words-beta-mem.pdf")

x = -6*(10:95)/100
plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-x)^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ja-words-logbeta-mem.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")

x = 6*(10:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1*(x)^2), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/ja-words-nlogbeta-mem-fitted.pdf")



x = -6*(10:95)/100
# this one looks good
plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=0.012*(3.5-(-x-0.1)^(-1.5))^(4.5)), aes(x=x, y=y), color="red") + ylim(0, NA)

plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ja-words-logbeta-ee.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=EE, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))








dataU = data %>% filter(Language == "Russian")
plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-surp-mem.pdf")

plot = ggplot(dataU, aes(x=avg16, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-16-mem.pdf")
plot = ggplot(dataU, aes(x=avg17, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg18, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg19, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

x = 3.2 * (1:100)/100
plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1.0*x^2*exp(1.3/(3.9-x))), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-ee-mem.pdf")


plot = ggplot(dataU, aes(x=EE, y=Memories, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1)) 


x = (1:500)/1000
plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-log(x))^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-beta-mem.pdf")

x = -6*(10:95)/100
plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-x)^1.6), aes(x=x, y=y), color="red")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-logbeta-mem.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")

x = 6*(10:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1*(x)^2), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/ru-words-nlogbeta-mem-fitted.pdf")



x = -6*(10:95)/100
# this one looks good
plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=0.012*(3.5-(-x-0.1)^(-1.5))^(4.5)), aes(x=x, y=y), color="red") + ylim(0, NA)

plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-logbeta-ee.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=EE, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))






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




dataU = data %>% filter(Language == "English")
dataU$UncondEnt = 1.895
dataU = dataU %>% mutate(MiWithFut = EE) #Horizon*(UncondEnt-FutureSurp))
#plot = ggplot(dataU, aes(x=Beta, y=Memories/EE)) + geom_point()+ theme_classic()  + xlim(0, 0.4)
dataU = dataU %>% select(Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut, avg16) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural")
dataU = dataU %>% filter(MiWithFut <= Memory)

x = 1.1*(1:100)/100
ggplot(dataU, aes(x=MiWithFut, y=Memory)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(x=x, y=x*exp(1/(1.5-x))), aes(x=x, y=y), color="red")


x = 20*(1:100)/100
ggplot(dataU, aes(x=Memory, y=MiWithFut)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(x=x, y=1.1-exp(1.0-0.6*x)), aes(x=x, y=y), color="red")

x = 1.1*(1:100)/100
ggplot(dataU, aes(x=MiWithFut, y=Memory)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(x=x, y=x*exp(1/(1.5-x))), aes(x=x, y=y), color="red")

x = 20*(1:100)/100
ggplot(dataU, aes(x=Memory, y=MiWithFut)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(x=x, y=1*(1-x^(1.0))), aes(x=x, y=y), color="red")
#based on THEORY
ggplot(dataU, aes(x=Memory, y=MiWithFut)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(x=x, y=1*(1.1-x^(-1.2))), aes(x=x, y=y), color="red")



# very nice fit, based on the idea that \log\lambda -> Memory is a power close to linear
ggplot(dataU, aes(x=Memory, y=MiWithFut)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(x=x, y=1.1-exp(0.8-0.9*x^0.6)), aes(x=x, y=y), color="red")
# the same, done more correctly
plot = ggplot(dataU, aes(x=Memory, y=MiWithFut)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(x=x, y=1.1-exp(-x^0.65) * (1+x^0.65 )), aes(x=x, y=y), color="red")



library(expint)

y = 20*(1:500)/500
# by solving diff eq obtained from superlinear growth
a=1.6
plot = ggplot(dataU, aes(x=MiWithFut, y=Memory)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(y=y), aes(x=1.13-a * gammainc(a, y^(1/a)), y=y), color="red")
ggsave("figures/english-info-fitted.pdf", plot=plot) 

x = 6*(1:500)/500
a=1.65
plot = ggplot(dataU, aes(x=-log(Beta), y=MiWithFut)) + geom_point()+ theme_classic()  + xlim(-log(.4), -log(0.001)) + theme(legend.position="none") + geom_point(data=data.frame(x=x), aes(x=x, y=1.13- a * gammainc(a, x)), color="red") + ylim(0, NA)
ggsave("figures/english-nlogbeta-ee-fitted.pdf", plot=plot) 




plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 

x = (1:100)/100
plot = ggplot(dataU, aes(x=Beta, y=Memory)) + geom_point()+ theme_classic()  + xlim(0, 0.4) + ylim(0, NA) + geom_point(data=data.frame(x=x), aes(x=x, y=(-log(x))^1.6), color="red")
ggsave("figures/english-beta-mem-fitted.pdf", plot=plot)


plot = ggplot(dataU, aes(x=log(Beta), y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none")

x = 6*(1:100)/100

y = 8.8 * x^6.8/(x-1)^0.0187
plot = ggplot(dataU, aes(x=log(Beta), y=MiWithFut)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + ylim(0, NA) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=y), aes(x=x, y=y))

dataU = dataU[order(dataU$Beta),]
write.csv(data.frame(x=log(dataU$Beta), y=dataU$MiWithFut), file="logBeta_EE.csv")

x = -6*(1:100)/100
plot = ggplot(dataU, aes(x=log(Beta), y=Memory)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x), aes(x=x, y=(-x)^1.6), color="red") + ylim(0, NA)
ggsave("figures/english-logbeta-mem-fitted.pdf", plot=plot) 

x = -6*(1:500)/500
plot = ggplot(dataU, aes(x=-log(Beta), y=Memory)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + theme(legend.position="none") + geom_point(data=data.frame(x=x), aes(x=-x, y=(-x)^1.6), color="red") + ylim(0, NA)
ggsave("figures/english-nlogbeta-mem-fitted.pdf", plot=plot) 

x = 20*(1:100)/100
plot = ggplot(dataU, aes(x=Memory, y=log(Beta))) + geom_point()+ theme_classic()  + xlim(0, 20) + theme(legend.position="none") + geom_point(data=data.frame(x=x), aes(x=x, y=-x^0.5), color="red") + ylim(NA, 0)



x = -6*(1:100)/100
y = 1/(x-1) + 1
plot = ggplot(dataU, aes(x=log(Beta), y=MiWithFut)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=y), aes(x=x, y=y))

# best fit obtained
ggplot(dataU, aes(x=log(Beta), y=MiWithFut)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=y), aes(x=x, y=-(-x)^1.0/(-x-0.1)^1.9 + 1.3), color="red") + ylim(0, NA)

# also good
ggplot(dataU, aes(x=log(Beta), y=MiWithFut)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=y), aes(x=x, y=-(-x)^1.0/(-x-0.1)^2.0 + 1.3), color="red") + ylim(0, NA)

# also okay
ggplot(dataU, aes(x=log(Beta), y=MiWithFut)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=y), aes(x=x, y=-(-x)^1.0/(-x-0.1)^2.6 + 1.1), color="red") + ylim(0, NA)

# similar version
ggplot(dataU, aes(x=log(Beta), y=MiWithFut)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=y), aes(x=x, y=-(-x)^1.0/(-x-0.1)^1.6 + 1.5), color="red") + ylim(0, NA)

# also reasonable
ggplot(dataU, aes(x=log(Beta), y=MiWithFut)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=y), aes(x=x, y=(1.1-(-x-0.1)^(-2))^(1.5)), color="red") + ylim(0, NA)

# based on THEORetical thoughts, also a very nice fit!
ggplot(dataU, aes(x=log(Beta), y=MiWithFut)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=y), aes(x=x, y=1.1-exp(0.8+0.9*x)), color="red")




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
plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlim(0,NA) + ylim(0, 1.1) + geom_line(data=data.frame(x=c(0, 0.6365), y = c(0, 0.6365)), aes(x=x , y=y, group=NA, color=NA), size=4) + theme(legend.position="none") + geom_line(data=findFrontier(dataNU$EE, dataNU$Memory), aes(x=EE, y=Memories, group=NULL, color=NULL), colour="red", size=1.5)
ggsave("figures/even-info.pdf", plot=plot) 





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

plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlim(0,NA) + ylim(0, 1.1) + geom_line(data=data.frame(x=c(0, 0.6365, 0.8595025), y = c(0, 0.6365, 1.039721)), aes(x=x , y=y, group=NA, color=NA), size=4) + theme(legend.position="none") + geom_line(data=findFrontier(dataNU$EE, dataNU$Memory), aes(x=EE, y=Memories, group=NULL, color=NULL), colour="red", size=1.5)
ggsave("figures/rip-info.pdf", plot=plot) 





data = read.csv("~/CS_SCR/results-en-upos-neuralflow-test-characters.tsv", sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
data$Horizon = 40

data$PastSurp = 40*data$FutureSurp+data$EE

#data = data %>% filter(Memories < UpperBound)
data = data %>% mutate(Objective = Horizon * FutureSurp + Beta * Memories)

data = data %>% filter(Script != "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE_Chars_PTB_2Flows.py")

dataU = data %>% filter(Language == "PTB") %>% filter(Memories < 200)
plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5, group=Script, color=Script)) + geom_point()+ theme_classic()  + theme(legend.position="none")



plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5, group=Script, color=Script)) + geom_point()+ theme_classic()  + theme(legend.position="none") + geom_point(data=dataU, aes(x=FutureSurp, y=UpperBound), color="black")


plot = ggplot(dataU, aes(x=avg16, y=Memories, alpha=0.5, group=Script, color=Script)) + geom_point()+ theme_classic()  + theme(legend.position="none") + geom_point(data=dataU, aes(x=avg16, y=UpperBound), color="black")
plot = ggplot(dataU, aes(x=avg17, y=Memories, alpha=0.5, group=Script, color=Script)) + geom_point()+ theme_classic()  + theme(legend.position="none")
plot = ggplot(dataU, aes(x=avg18, y=Memories, alpha=0.5, group=Script, color=Script)) + geom_point()+ theme_classic()  + theme(legend.position="none")
plot = ggplot(dataU, aes(x=avg19, y=Memories, alpha=0.5, group=Script, color=Script)) + geom_point()+ theme_classic()  + theme(legend.position="none")

x = 3.2 * (1:100)/100
plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5, group=Script, color=Script)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1.0*x^2*exp(1.3/(3.9-x))), aes(x=x, y=y, group=NULL, color=NULL), color="red") + theme(legend.position="none")

x = (1:500)/1000
plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5, group=Script, color=Script)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-log(x))^1.6), aes(x=x, y=y, group=NULL, color=NULL), color="red") + theme(legend.position="none")

x = -6*(10:95)/100
#plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=-x), aes(x=x, y=y), color="red") 
plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5, group=Script, color=Script)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.8*(-x)^1.6), aes(x=x, y=y, group=NULL, color=NULL), color="red") + theme(legend.position="none") + geom_point(data=dataU, aes(x=log(Beta), y=UpperBound), color="black")


x = -6*(10:95)/100
# this one looks good
ggplot(dataU, aes(x=log(Beta), y=EE, group=Script, color=Script)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=0.012*(3.5-(-x-0.1)^(-1.5))^(4.5)), aes(x=x, y=y, group=NULL, color=NULL), color="red") + ylim(0, NA)
#plot = ggplot(dataU, aes(x=log(Beta), y=EE, alpha=0.5)) + geom_point()+ theme_classic() + geom_point(data=data.frame(x=x, y=1.3-(-x)^1/(-x-0.1)^2), aes(x=x, y=y), color="red")
#ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, y=(3.5-(-x-0.1)^(-2))^(1.5)), aes(x=x, y=y), color="red") + ylim(0, NA)







data = read.csv("~/CS_SCR/results-en-upos-neuralflow-test.tsv", sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
data$Horizon = 15

data$PastSurp = 15*data$FutureSurp+data$EE

data = data %>% filter(Memories < UpperBound) #can also take UpperBound2
data = data %>% mutate(Objective = Horizon * FutureSurp + Beta * Memories)


dataU = data %>% filter(Language == "PTB")
plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, color=model, group=model, alpha=0.5)) + geom_point()+ theme_classic() + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")

plot = ggplot(dataU, aes(x=avg16, y=Memories, color=model, group=model, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")

plot = ggplot(dataU, aes(x=avg16, y=EE, color=model, group=model, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")


plot = ggplot(dataU, aes(x=EE, y=Memories, color=model, group=model, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")



