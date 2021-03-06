library(tidyr)
library(dplyr)
library(ggplot2)



data = read.csv("../../results/results-nprd.tsv", sep="\t")
dataD = read.csv("../../results/results-oce.tsv", sep="\t")

data$Horizon = 15
data = data %>% filter(Memories < UpperBound)


dataU = data %>% filter(Language == "English")
dataU$UncondEnt = 1.895
dataU = dataU %>% mutate(MiWithFut = EE)
dataU = dataU %>% select(Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut, avg16) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural")
dataU = dataU %>% filter(MiWithFut <= Memory)



library(expint)

y = 20*(1:500)/500
# by solving diff eq obtained from superlinear growth
a=1.6
plot = ggplot(dataU, aes(x=MiWithFut, y=Memory)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(y=y), aes(x=1.13-a * gammainc(a, y^(1/a)), y=y), color="red")
plot = plot +    theme(    axis.text.x = element_text(size=20),
                           axis.text.y = element_text(size=20),
                           axis.title.x = element_text(size=25),
                           axis.title.y = element_text(size=25))
plot = plot + xlab("Predictiveness")
plot = plot + ylab("Rate")
plot = plot + theme(legend.position="none")
ggsave("../figures/english-info-fitted.pdf", plot=plot) 

x = 6*(1:500)/500
a=1.65
plot = ggplot(dataU, aes(x=-log(Beta), y=MiWithFut)) + geom_point()+ theme_classic()  + xlim(-log(.4), -log(0.001)) + theme(legend.position="none") + geom_point(data=data.frame(x=x), aes(x=x, y=1.13- a * gammainc(a, x)), color="red") + ylim(0, NA)
plot = plot +    theme(    axis.text.x = element_text(size=20),
                           axis.text.y = element_text(size=20),
                           axis.title.x = element_text(size=25),
                           axis.title.y = element_text(size=25))
plot = plot + xlab("log(1/\u03BB)")
plot = plot + ylab("Predictiveness")
plot = plot + theme(legend.position="none")
ggsave("../figures/english-nlogbeta-ee-fitted.pdf", plot=plot, device=cairo_pdf) 

x = (1:100)/100
plot = ggplot(dataU, aes(x=Beta, y=Memory)) + geom_point()+ theme_classic()  + xlim(0, 0.4) + ylim(0, NA) + geom_point(data=data.frame(x=x), aes(x=x, y=(-log(x))^1.6), color="red")
plot = plot +    theme(    axis.text.x = element_text(size=20),
                           axis.text.y = element_text(size=20),
                           axis.title.x = element_text(size=25),
                           axis.title.y = element_text(size=25))
plot = plot + xlab("Beta")
plot = plot + ylab("Rate")
plot = plot + theme(legend.position="none")
ggsave("../figures/english-beta-mem-fitted.pdf", plot=plot, device=cairo_pdf)


dataU = dataU[order(dataU$Beta),]
write.csv(data.frame(x=log(dataU$Beta), y=dataU$MiWithFut), file="logBeta_EE.csv")

x = -6*(1:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memory)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + theme(legend.position="none") + geom_point(data=data.frame(x=x), aes(x=-x, y=(-x)^1.6), color="red") + ylim(0, NA)
plot = plot +    theme(    axis.text.x = element_text(size=20),
                           axis.text.y = element_text(size=20),
                           axis.title.x = element_text(size=25),
                           axis.title.y = element_text(size=25))
plot = plot + xlab("log(1/\u03BB)")
plot = plot + ylab("Rate")
plot = plot + theme(legend.position="none")
ggsave("../figures/english-logbeta-mem-fitted.pdf", plot=plot, device=cairo_pdf) 

x = -6*(1:500)/500
plot = ggplot(dataU, aes(x=-log(Beta), y=Memory)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + theme(legend.position="none") + geom_point(data=data.frame(x=x), aes(x=-x, y=(-x)^1.6), color="red") + ylim(0, NA)
plot = plot +    theme(    axis.text.x = element_text(size=20),
                           axis.text.y = element_text(size=20),
                           axis.title.x = element_text(size=25),
                           axis.title.y = element_text(size=25))
plot = plot + xlab("log(1/\u03BB)")
plot = plot + ylab("Rate")
plot = plot + theme(legend.position="none")

ggsave("../figures/english-nlogbeta-mem-fitted.pdf", plot=plot, device=cairo_pdf) 





















library(tidyr)
library(dplyr)
library(ggplot2)




data = read.csv("../../results/results-nprd.tsv", sep="\t")
dataD = read.csv("../../results/results-oce.tsv", sep="\t")

data$Horizon = 15

data = data %>% filter(Memories < UpperBound)

plot = ggplot(data, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

plot = ggplot(data, aes(x=avg16, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 




dataU = data %>% filter(Language == "English") %>% filter(model == "REAL")
dataU$UncondEnt = 1.895
dataU = dataU %>% mutate(MiWithFut = EE) #Horizon*(UncondEnt-FutureSurp))
#plot = ggplot(dataU, aes(x=Beta, y=Memories/EE)) + geom_point()+ theme_classic()  + xlim(0, 0.4)
dataDU = dataD %>% filter(Language == "English")
dataDU = dataDU %>% mutate(UncondEnt = case_when(Horizon == 1 ~ 2.563, Horizon == 2 ~ 2.18, Horizon == 3 ~  2.15)) 
dataDU = dataDU %>% mutate(MiWithFut = Horizon*(UncondEnt-Surprisal))
dataU = rbind(dataDU %>% select(Language, Beta,Surprisal, Memory, Horizon, UncondEnt, MiWithFut) %>% mutate(Type="Ngrams") %>% mutate(model="REAL"), dataU %>% select(Language, Beta,FutureSurp, Memories, Horizon, UncondEnt, MiWithFut, model) %>% rename(Surprisal=FutureSurp, Memory=Memories) %>% mutate(Type="Neural"))
dataU = dataU %>% filter(MiWithFut <= Memory)



plot = ggplot(dataU, aes(x=MiWithFut, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA)
plot = plot +    theme(    axis.text.x = element_text(size=20),
                           axis.text.y = element_text(size=20),
                           axis.title.x = element_text(size=25),
                           axis.title.y = element_text(size=25))
plot = plot + xlab("Predictiveness")
plot = plot + ylab("Rate")
plot = plot + theme(legend.position="none")
ggsave("../figures/english-info.pdf", plot=plot, device=cairo_pdf) 

plot = ggplot(dataU, aes(x=Surprisal, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Memory, y=Surprisal, group=Type, color=Type)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=Beta, y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(0, 0.4) + ylim(0, NA)
plot = plot +    theme(    axis.text.x = element_text(size=20),
                           axis.text.y = element_text(size=20),
                           axis.title.x = element_text(size=25),
                           axis.title.y = element_text(size=25))
plot = plot + xlab("Beta")
plot = plot + ylab("Rate")
plot = plot + theme(legend.position="none")
ggsave("../figures/english-beta-mem.pdf", plot=plot, device=cairo_pdf)

plot = ggplot(dataU, aes(x=-log(Beta), y=Memory, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + ylim(0, NA) + theme(legend.position="none")
plot = plot +    theme(    axis.text.x = element_text(size=20),
                           axis.text.y = element_text(size=20),
                           axis.title.x = element_text(size=25),
                           axis.title.y = element_text(size=25))
plot = plot + xlab("log(1/\u03BB)")
plot = plot + ylab("Rate")
ggsave("../figures/english-nlogbeta-mem.pdf", plot=plot, device=cairo_pdf)

plot = ggplot(dataU, aes(x=-log(Beta), y=MiWithFut, group=Type, color=Type)) + geom_point()+ theme_classic()  + xlim(-log(0.4), -log(0.001)) + ylim(0, NA) + theme(legend.position="none")
plot = plot +    theme(    axis.text.x = element_text(size=20),
                           axis.text.y = element_text(size=20),
                           axis.title.x = element_text(size=25),
                           axis.title.y = element_text(size=25))
plot = plot + xlab("log(1/\u03BB)")
plot = plot + ylab("Predictiveness")
ggsave("../figures/english-nlogbeta-ee.pdf", plot=plot, device=cairo_pdf)




