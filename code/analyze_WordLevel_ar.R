
data = read.csv("~/CS_SCR/results-en-upos-neuralflow-test.tsv", sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
data$Horizon = 15

data$PastSurp = 15*data$FutureSurp+data$EE

data = data %>% filter(Memories < UpperBound) #can also take UpperBound2
data = data %>% mutate(Objective = Horizon * FutureSurp + Beta * Memories)

data = data %>% filter(model != "REVERSE")



#####################################################3



dataU = data %>% filter(Language == "Arabic")
plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-surp-mem.pdf")

plot = ggplot(dataU, aes(x=avg16, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-16-mem.pdf")

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-ee-mem.pdf")

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-beta-mem.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-logbeta-mem.pdf")


plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-logbeta-ee.pdf")

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-ee-mem.pdf")

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-beta-mem.pdf")

plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-logbeta-mem.pdf")



plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA) + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-logbeta-ee.pdf")


x = 6*(10:100)/100
alpha=1
beta=2.3
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=alpha*(x)^beta), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/ar-words-nlogbeta-mem-fitted.pdf")



library(expint)

y = 60*(1:500)/500
# by solving diff eq obtained from superlinear growth
E0=3.1
plot = ggplot(dataU, aes(x=EE, y=Memories)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(y=y), aes(x=E0-alpha*beta * gammainc(beta, (y/alpha)^(1/beta)), y=y), color="red")
ggsave("figures/ar-words-info-fitted.pdf", plot=plot) 

x = 6*(1:500)/500 # this is for log(lambda)
plot = ggplot(dataU, aes(x=-log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(-log(.4), -log(0.001)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, rate=alpha*x^beta), aes(x=x, y=E0-1.6*alpha*beta * gammainc(beta, (rate/alpha)^(1/beta))), color="red") + ylim(0, NA)
ggsave("figures/ar-words-nlogbeta-ee-fitted.pdf", plot=plot) 





