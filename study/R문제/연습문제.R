#3단원

#p79
x=1
y=2
temp=y
y=x
x=temp
x
y

#p83
#1)
xinf=Inf
yinf=-Inf
xinf/yinf

#2)
1<2
#True
4<3
#False
#비교연산 할때 논리형으로 결과값 나옴.

#3)
str="hello world"
blood.type=factor(c('A',"B","O","AB"))
#문자형은 작은 따옴표나 큰따옴표로 묶어서 표기한다. 범주형은 factor레벨에 따라 분류한다.

#p84
5%/%2
5/2
#1)?
#a>1 & a<5

#2)
!(x%%3==0&x%%4==0)
x%%3!=0|x%%4!=0

#p90
#1)
x=c(1:5)
x=c(x,6:10)
x

#2
x=c(1:10)
x=x[x%%2==0]
x

#p95
#1)??
Titanic
plot(Titanic)
summary(Titanic)
str(Titanic)

#2)
x=array(1:24,c(4,6))
d=dim(x)
for(i in 1:d[1])
  for(j in 1:d[2])
    if(j%%2==0)
      x[i,j]=0
x

#p104
#1)
table(is.na(airquality$Ozone))
mean(na.omit(airquality$Ozone))

#2)
name=c("철수","춘향","길동")
age=c(22,20,25)
gender=factor(c("M","F","M"))
blood.type=factor(c("A","O","B"))
patients=data.frame(name,age,gender,blood.type)
patients
str(patients)
#name은 chr형으로 표현됨. 순서가 필요없음

#p109
#1)
patients = data.frame(name = c("철수", "춘향", "길동"), age = c(22, 20, 25), gender = factor(c("M", "F", "M")), blood.type = factor(c("A", "O", "B")))
no.patients = data.frame(day = c(1:6), no = c(50, 60, 55, 52, 65, 58))
room=30
listPatients=list(patients = patients, no.patients = no.patients, room = room)
listPatients

#2)
listPatients["room"]=NULL
listPatients


#4단원
#p122
#1)
students = read.csv("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students1.csv")
students
str(students)
ave = apply(students[,2:4],1,mean)
students[,"average"] = as.data.frame(ave) #same with => students = cbind(students,ave)
students
write.csv(students, file="C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students_result.csv")

#p131
#1)
total=0
for (i in c(1:10)){
  if(i%%2==0){
    total=total+i 
  }
}
total

#2)
seq(from=1, to=10, by=2)

#3)???모름
c=TRUE
for(i in 1:10){
  for(j in 2:i-1){
    if(i%%j==0){
      c=FALSE
      break
    }
  }
  if(c==TRUE){
    print(i) 
  }
  c=TRUE
}

#p134
#1)
fact = function(x){
  if(x==1){
    return(1) 
  }
  else{
    fact(x-1)*x
  }
}
fact(10)

#2)#아직덜함함
prime_num = function(x){
    check = 0
    for(j in 1:x){
      if(i%%j==0){
        check=check+1
        print(paste(j,check))
      }
    }
    if(check==2){
      return(TRUE)
    }
    else{
      return(FALSE)
    }
  
}
prime_num(4)

#5장

#p155
#1)
apply(gapminder[gapminder$country == "Korea, Rep.", c("pop","year")], 2, max)

#2)
apply(gapminder[gapminder$year == 2007 & gapminder$continent == "Asia", "pop"], 2, sum)

#p160
#1)
gapminder[gapminder$country == "Korea, Rep." | gapminder$country == "China" | gapminder$country == "Japan", c("country","year","gdpPercap", "lifeExp")]
data = data.frame(gapminder)
data%>% filter(country %in% c("Korea, Rep.", "Japan", "China"))%>% arrange(country, year)%>% select(country, year, lifeExp, gdpPercap)
  
#######
data = data.frame(gapminder)
data
data %>%
  dplyr::filter(country %in% c("Korea, Rep.", "Japan", "China")) %>%
  dplyr::arrange(country, desc(year)) %>%
  dplyr::select(-continent, -pop)
#######

#2)
data = gapminder[gapminder$continent=="Africa" | gapminder$continent=="Europe",]
data
comp =  summarize(group_by(data, year, continent), all_pop =sum(pop))
str(comp)
count=1
while(count<=length(comp$year)){
  if(comp$all_pop[count] > comp$all_pop[count+1]){
    print(comp$year[count])
  }
  count=count+2
}

####
a <- gapminder %>% filter(continent == 'Africa') %>% group_by(year) %>% summarize(population_sum = sum(pop))
b <- gapminder %>% filter(continent == 'Europe') %>% group_by(year) %>% summarize(population_sum = sum(pop))

c <- merge(a, b, by = 'year')
colnames(c) <- c('year', 'Africa', 'Europe')

c %>% filter(Africa >= Europe) %>% select(year) %>% pull()
####

#3)
library(gapminder)
library("dplyr")
data = gapminder_unfiltered 
#summarize(group_by(data, country, year))

d <- data %>% summarize(group_by(data, country, year))%>%group_by(country)%>%summarise(n=n())%>%filter(n>=12)%>%arrange(desc(n))





