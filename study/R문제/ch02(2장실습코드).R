getwd()

setwd('C:/Users/admin/Desktop/Bigdata_Practice/TEST')
getwd()

setwd('C:/Users/admin/Desktop/Bigdata_Practice')
getwd()

# package install, add library 

install.packages("dplyr")
install.packages("ggplot2")

library(dplyr)
library(ggplot2)

str(cars) #데이터의 구조를 보여주는 함수 str
head(cars) #6개의 데이터와 라벨을 보여주는 함수
plot(Cars) #데이터 및 대상의 그래프를 그려주는 함수

str(cars)  # structure of data:cars
head(cars)  # 6 head data of data: cars
tail(cars)  # 6 tail data of data: cars
plot(cars)
cars

women
plot(women)

plot(cars)
plot(cars, col='blue', xlab='속도', ylab='거리', pch=18)

?str
?plot




data #기본데이터 목록보여줌
str(iris)
head(iris, 10)
plot(iris)
summary(iris)

plot(iris$Petal.Width, iris$Petal.Length, col = iris$Species)

tips = read.csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
str(tips)
head(tips, 10)


summary(tips)                


library(dplyr)
library(ggplot2)

#  pic 2-19 (a)~(d)
tips%>%ggplot(aes(size)) + geom_histogram()                                           # # pic 2-19 (a)
tips%>%ggplot(aes(total_bill, tip)) + geom_point()                                    # # pic 2-19 (b)
tips%>%ggplot(aes(total_bill, tip)) + geom_point(aes(col = day))                      # # pic 2-19 (c)
tips%>%ggplot(aes(total_bill, tip)) + geom_point(aes(col = day, pch = sex), size = 3) # # pic 2-19 (d)



rm(list=ls())   # Delete all loaded data and variables
                # CTRL+L : Clear console window