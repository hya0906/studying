#2단원

#number 1
?plot
Speed <- cars$speed
Distance <- cars$dist
plot(Speed, Distance, panel.first = grid(8, 8),
     pch = 0, cex = 1.2, col = "blue")
plot(Speed, Distance,
     panel.first = lines(stats::lowess(Speed, Distance), lty = "dashed"),
     pch = 0, cex = 1.2, col = "blue")

?summary
summary(attenu, digits = 4) #-> summary.data.frame(...), default precision

#number 2
data()
plot(women)
plot(women, col = 'blue')
plot(women, col = 'blue', xlab = '높이', ylab = '무게')
plot(women, col = 'blue', xlab = '높이', ylab = '무게', pch = 18)


#3단원

#number 1
x<-1:100
x = x[x%%3==0]
x

y<-1:100
y = y[y%%4==0]
y


#number 2
z= intersect(x,y)
sum(z)

#number  3
summary(airquality)
# data from New York in 1973

#number 4
#화씨온도

#number  5
max_wind = max(airquality$Wind)
airquality[airquality$Wind == max_wind, ]
#6월 17일

#number 6
table(is.na(airquality))
#44개

#number 7
#피지섬

#number 8
summary(quakes)
max_mag=max(quakes$mag)
quakes[quakes$mag==max_mag, ]
#6.4규모


#4단원

#number 1

x<-1:100
x
x = x[x%%3==0 & x%%4!=0]
sum(x)

#number 2
func = function(x,n){
  num<-1:n
  num = num[num%%x==0]
  return (sum(num))
}
func(3,20)

#number 3
install.packages("hflights")
library(hflights)
table(is.na(hflights))

#number 4
summary("hflights")
str("hflights")
table(is.na(hflights))
hflights = na.omit(hflights)

summary(hflights)
max_airtime = max(hflights$AirTime)
hflights$AirTime[hflights$AirTime== max_airtime]
#549시간

#number 5
hflights$Distance[hflights$Distance== max(hflights$Distance)]
#3904

#number 6
sum(hflights$Cancelled)
hflights$Cancelled

