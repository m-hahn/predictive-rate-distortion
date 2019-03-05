


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

x = (1:100)/100
plot = ggplot(dataU, aes(x=Beta, y=Memory)) + geom_point()+ theme_classic()  + xlim(0, 0.4) + ylim(0, NA) + geom_point(data=data.frame(x=x), aes(x=x, y=(-log(x))^1.6), color="red")
ggsave("figures/english-beta-mem-fitted.pdf", plot=plot)


dataU = dataU[order(dataU$Beta),]
write.csv(data.frame(x=log(dataU$Beta), y=dataU$MiWithFut), file="logBeta_EE.csv")

x = -6*(1:100)/100
plot = ggplot(dataU, aes(x=log(Beta), y=Memory)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + geom_point(data=data.frame(x=x), aes(x=x, y=(-x)^1.6), color="red") + ylim(0, NA)
ggsave("figures/english-logbeta-mem-fitted.pdf", plot=plot) 

x = -6*(1:500)/500
plot = ggplot(dataU, aes(x=-log(Beta), y=Memory)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + theme(legend.position="none") + geom_point(data=data.frame(x=x), aes(x=-x, y=(-x)^1.6), color="red") + ylim(0, NA)
ggsave("figures/english-nlogbeta-mem-fitted.pdf", plot=plot) 


