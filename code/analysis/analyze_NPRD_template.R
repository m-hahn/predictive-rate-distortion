

library(tidyr)
library(dplyr)
library(ggplot2)


source("findFrontier.R")

data = read.csv("../../results/results-nprd.tsv", sep="\t")
data$Horizon = 15
data = data %>% filter(Memories < UpperBound)
dataU = data %>% filter(Language == "LANGUAGE")
dataU = dataU %>% mutate(MiWithFut = EE)
dataU = dataU %>% select(EE, Language, Beta,FutureSurp, Memories, Horizon, MiWithFut) %>% rename(Surprisal=FutureSurp, Memory=Memories) 
dataU = dataU %>% filter(MiWithFut <= Memory)


plot = ggplot(dataU, aes(x=MiWithFut, y=Memory))
plot = plot + geom_point()+ theme_classic() + xlim(0,NA) + ylim(0, 1.1) 
plot = plot + theme(legend.position="none") 
plot = plot + geom_line(data=findFrontier(dataNU$EE, dataNU$Memory), aes(x=EE, y=Memories, group=NULL, color=NULL), colour="red", size=2.5)
plot = plot +    theme(    axis.text.x = element_text(size=20),
		           axis.text.y = element_text(size=20),
			   axis.title.x = element_text(size=25),
			   axis.title.y = element_text(size=25))
plot = plot + xlab("Predictiveness")
plot = plot + ylab("Rate")
ggsave("../figures/LANGUAGE-info.pdf", plot=plot) 





