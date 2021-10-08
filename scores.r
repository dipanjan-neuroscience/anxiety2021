library(ggplot2)
library(plotly)
#library(gridExtra)
#X=(1:164)
#p1<-ggplot(FSAFAO, mapping = aes(x = X, y = FAO)) +geom_point(color='darkred') +geom_line(color='darkred')+xlab("Participants")+ylab("Mean centered scores")+theme_classic()+ggtitle("Fear Somatic Arousal Scores")+theme(plot.title = element_text(hjust = 0.5))
#p2<-ggplot(FSAFAO, mapping = aes(x = X, y = FA)) +geom_point(color='darkblue') +geom_line(color='darkblue')+xlab("Participants")+ylab("Mean centered scores")+theme_classic()+ggtitle("Fear Affect Scores")+theme(plot.title = element_text(hjust = 0.5))
#p3<-ggplot(FSAFAO, mapping = aes(x = X, y = FAO)) +geom_point(color='blue') +geom_line(color='blue')+xlab("Participants")+ylab("Mean centered scores")+theme_classic()+ggtitle("Fear Affect Scores (Orthogonalized)")+theme(plot.title = element_text(hjust = 0.5))
#grid.arrange(p1, p2, p3, nrow = 3)

PScores <- read_excel("Downloads/PScores.xlsx")

p<- ggplot(PScores, mapping = aes(x = Sub, y = Val, shape = Scores, colour = Scores)) + 
  geom_point() + geom_line()+facet_grid(factor(Scores, levels=c('FSA','FA','FAO'))~.)+
  xlab("Participants")+ylab("Mean centered scores")+theme(legend.position = "none")

print(p)






