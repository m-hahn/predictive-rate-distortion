
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

dataU = data %>% filter(Language == "PTB")
plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-surp-mem.pdf")

plot = ggplot(dataU, aes(x=avg16, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-16-mem.pdf")
plot = ggplot(dataU, aes(x=avg17, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg18, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg19, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-ee-mem.pdf")

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-beta-mem.pdf")

plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-logbeta-mem.pdf")

# a=2.0
# b=1.9
x = 6*(10:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2.0*(x)^1.9), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/en-words-nlogbeta-mem-fitted.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA) + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/en-words-logbeta-ee.pdf")





library(expint)

y = 60*(1:500)/500
# by solving diff eq obtained from superlinear growth
alpha=2.0
beta=1.9
E0=3.5
plot = ggplot(dataU, aes(x=EE, y=Memories)) + geom_point()+ theme_classic() + xlab("I[Z, Future]") + ylab("I[Z, Past]") + theme(legend.position="none") + xlim(0, NA) + ylim(0, NA) + geom_point(data=data.frame(y=y), aes(x=E0-3.5*alpha/beta * gammainc(beta, (y/alpha)^(1/beta)), y=y), color="red")
ggsave("figures/en-words-info-fitted.pdf", plot=plot) 

x = 6*(1:500)/500 # this is for log(lambda)
a=1.65
plot = ggplot(dataU, aes(x=-log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(-log(.4), -log(0.001)) + theme(legend.position="none") + geom_point(data=data.frame(x=x, rate=alpha*x^beta), aes(x=x, y=E0-3.5*alpha/beta * gammainc(beta, (rate/alpha)^(1/beta))), color="red") + ylim(0, NA)
ggsave("figures/en-words-nlogbeta-ee-fitted.pdf", plot=plot) 






###############################################################3



dataU = data %>% filter(Language == "LDC95T8")
dataU$BeatsBound = (dataU$Memories < dataU$UpperBound)

plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-surp-mem.pdf")

plot = ggplot(dataU, aes(x=avg16, y=Memories, color=BeatsBound)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-16-mem.pdf")
plot = ggplot(dataU, aes(x=avg17, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg18, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 
plot = ggplot(dataU, aes(x=avg19, y=Memories, alpha=0.5)) + geom_point()+ theme_classic() 

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-ee-mem.pdf")


plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5, color=BeatsBound)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")

plot = ggplot(dataU, aes(x=EE, y=Memories, group=flow_layers, color=flow_layers)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1)) 

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-beta-mem.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, color=BeatsBound)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-logbeta-mem.pdf")

x = 6*(10:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2*(x)^2), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/LDC95T8-words-nlogbeta-mem-fitted.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC95T8-words-logbeta-ee.pdf")

#########################################


dataU = data %>% filter(Language == "LDC2012T05")
dataU$BeatsBound = (dataU$Memories < dataU$UpperBound)

plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-surp-mem.pdf")


plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, color=BeatsBound)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")


plot = ggplot(dataU, aes(x=avg16, y=Memories, color=BeatsBound)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-16-mem.pdf")

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-ee-mem.pdf")


plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-beta-mem.pdf")


plot = ggplot(dataU, aes(x=log(Beta), y=Memories, color=BeatsBound)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-logbeta-mem.pdf")

x = 6*(10:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1*(x)^2.3), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/LDC2012T05-words-nlogbeta-mem-fitted.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/LDC2012T05-words-logbeta-ee.pdf")



##############################


dataU = data %>% filter(Language == "Arabic")
plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-surp-mem.pdf")

dataU$BeatsBound = (dataU$Memories < dataU$UpperBound)

plot = ggplot(dataU, aes(x=avg16, y=Memories, color=BeatsBound, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-16-mem.pdf")

plot = ggplot(dataU, aes(x=EE, y=Memories, color=BeatsBound, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-ee-mem.pdf")

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-beta-mem.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, color=BeatsBound, alpha=0.5)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-logbeta-mem.pdf")

x = 6*(10:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=2*(x)^2), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/ar-words-nlogbeta-mem-fitted.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ar-words-logbeta-ee.pdf")



################################



dataU = data %>% filter(Language == "Russian")
plot = ggplot(dataU, aes(x=FutureSurp, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-surp-mem.pdf")

plot = ggplot(dataU, aes(x=avg16, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-16-mem.pdf")

plot = ggplot(dataU, aes(x=EE, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()   + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-ee-mem.pdf")

plot = ggplot(dataU, aes(x=Beta, y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-beta-mem.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()    + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-logbeta-mem.pdf")

x = 6*(10:100)/100
plot = ggplot(dataU, aes(x=-log(Beta), y=Memories, alpha=0.5)) + geom_point()+ theme_classic()  + geom_point(data=data.frame(x=x, y=1*(x)^2), aes(x=x, y=y), color="red")
ggsave(plot, file="figures/ru-words-nlogbeta-mem-fitted.pdf")

plot = ggplot(dataU, aes(x=log(Beta), y=EE)) + geom_point()+ theme_classic()  + xlim(log(0.001), log(0.4)) + theme(legend.position="none") + ylim(0, NA)  + theme(text = element_text(size=20), axis.text.x = element_text(angle=90, hjust=1))  + theme(legend.position="none")
ggsave(plot, file="figures/ru-words-logbeta-ee.pdf")


