#연습문제 p174
#1.
#2.

#연습문제 p.185
#1.
library(gapminder)
library(dplyr)
y<-gapminder%>%group_by(continent)%>%summarize(mean(gdpPercap))
y
plot(y)
legend("topleft", legend=levels((y$continent)))

#2.
gapminder
x<-gapminder%>%group_by(continent)%>%filter(year=="1952")%>%summarize(gdppercap=mean(gdpPercap),lifeexp=mean(lifeExp))
x
plot(x$gdppercap, x$lifeexp, pch=c(1:5))

legend("topleft",legend=levels((x$continent)),pch=c(1:length(levels(x$continent))))
ggplot(x,aes(x=gdppercap, lifeexp, col=continent))+geom_point(size=5)

#연습문제 p197
#1. 정렬한 것과 정렬하지 않은 것의 차이이
gapminder %>% filter(year==1952&continent=="Asia")%>%ggplot(aes(reorder(country,pop),pop))+geom_bar(stat="identity")+coord_flip()
gapminder %>% filter(year==1952&continent=="Asia")%>%ggplot(aes(country,pop))+geom_bar(stat="identity")+coord_flip()

#2.1번은 분포를 상자로 나타내고 2번은 점으로 분포를 나타냄
library(ggplot2)
x%>%ggplot(aes(continent,lifeexp))+geom_boxplot()
ggplot(gapminder,aes(continent,lifeExp))+geom_point(alpha=0.2,size=1.0,position="jitter")

#연습문제 p211
#1.
head(cars)
plot(cars, type="p", main= "cars")
ggplot(cars, aes(x=speed, y=dist))+geom_point()

#2.?
iris[,1:4]
matplot(iris[, 1:4], type='l')
legend("topleft", names(iris)[1:4], lty=c(1,2,3,4), col=c(1,2,3,4))
ggplot(iris[1:4], aes(x=rownames(iris), y=Sepal.Length))+geom_line(aes(group=rownames(iris)))

#3.?
gapminder %>% filter(continent == "Africa") %>% ggplot(aes(x=country, y=reorder(-lifeExp))) + geom_bar(stat="identity")+coord_flip()
                                                       
#연습문제 p228
#1.
x=c(5,3)
y=c(50000000,4000000)
lm(y~x)

#연습문제 p237
#1.
x=c(10.0,12.0,9.5,22.2,8.0)
y=c(360.2,420.0,359.5,679.0,315.3)
m=lm(y~x)
plot(x,y)
abline(m,col="red")
deviance(m) #잔차제곱합
deviance(m)/length(x) #평균제곱오차차
newx=data.frame(x= c(10.5,25.0,15.0))
predict(m, newdata=newx)

#2.
newx= data.frame(x= c(10.5, 25.0, 15.0))
newy=predict(m, newdata=newx)
newx= unlist(data.frame(x= c(10.5, 25.0, 15.0)))
newx
newy
plot(newx,newy)
abline(m,col="green")

#연습문제 p246
#1.
women
women=rbind(women,c(65.5,121))
women
wm=lm(weight~height, data=women)
plot(women)
abline(wm, col="red")
summary(wm)

#2.
cars
cars=cars[-c(20,22,23),]
cars
cm=lm(dist~speed, data=cars)
plot(cars)
abline(cm, col="red")
summary(cm)

#연습문제 p250
#1.
x = c(3.0, 6.0, 3.0, 6.0, 7.7)
u = c(10.0, 10.0, 20.0, 20.0, 14.8)
y = c(4.65, 5.9, 6.7, 8.02, 7.7)
m = lm(y ~ x + u)
coef(m)
s = scatterplot3d(x, u, y, xlim = 2:7, ylim = 7:23, zlim = 0:10, pch = 20, type = 'h')
s$plane3d(m)

nx = c(7.5, 5.0)
nu = c(15.0, 12.0)
new_data = data.frame(x = nx, u = nu)          
new_data
ny = predict(m, new_data)
ny
s = scatterplot3d(nx, nu, ny, xlim = 4:10, ylim = 7:23, zlim = 0:10, pch = 20, type = 'h')
s$plane3d(m)

#연습문제 p253
#1.?
nx=matrix(c(3.0,10.0),c(6.0,10.0),c(3.0,20.0),c(6.0,20.0),c(7.5,5.0),c(7.5,10.0),c(15.0,12.0),nrow=6)
ny=c(4.65,5.9,8.02,7.7,8.1,6.1)
new_data=data.frame(x=nx, y=ny)
m=lm(y~x, new_data)
class(trees)
class(new_data)

#2.
