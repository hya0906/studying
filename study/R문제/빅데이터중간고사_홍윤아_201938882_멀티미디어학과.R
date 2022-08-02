#이름:홍윤아 학번:20193882

#1번
a=4
b=5
c=a+b
c

#2번
pi=3.14
a=10*10*pi
b=12*12*pi
c=15*15*pi
a
b
c

#3번
#val.a에만 값이 할당되어있고 val.b에는 값이 할당되어있지 않은데 아래줄에서
#sum.ab<-val.a+val.b로 val.b를 이용하는 식이 있으므로 에러가 난다.
#이를 해결하기 위해서 val.b에 대한 값을 할당해 주고 sum.ab를 구하면 
#에러가 나지 않는다.

#4번
#시간=거리/속력
dist=60L
v1=20L
v2=30L
avg=dist/v1+dist/v2
avg

#5번
even = c(1:100)
even = even[even%%2==0]
sum=0
for(i in 1:length(even)){
  sum = sum + even[i]
}
sum

#6번
shoes = 76000
pants=52000
shirts=36000
sum=0
sale = function(x,y,z){
  shoes = x*0.95
  pants = y*0.9
  shirts = z*0.95
  sum = shoes+pants+shirts
  return(sum)
}
sale(shoes,pants,shirts)

#7번
d=c(100:200)
#1)
tail(d,10)

#2
d.20=head(d,20)
d.20

#3)
d.20[-c(5,7,9)]

#8번
point <- c(7, 7, 8, 8, 9, 10, 10, 10, 11, 20)
answer_sheet <- c(1, 3, 2, 2, 4, 1, 5, 2, 3, 1)
mark <- c(1, 3, 1, 2, 4, 2, 5, 2, 5, 1)
#1)
count=0
for(i in 1:length(mark)){
  if(answer_sheet[i]==mark[i]){
    count=count+1
  }
}
count
#7개

#2)
score=0
for(i in 1:length(mark)){
  if(answer_sheet[i]==mark[i]){
    score=score+point[i]
  }
}
score
#71점

#9번
#1)
st<-data.frame(state.x77)
st

#2)
names(st)

#3)
dim(st)

#4)
summary(st)

#5)
library(dplyr)
max(subset(st, select = Life.Exp))


#10번
#1)
corp <- matrix(c(12289, 1460, 
                 8921, 1060, 
                 17589, 2091, 
                 5389, 652, 
                 12642, 1502, 
                 9463, 1125 ),
               ncol = 2, byrow = T)

colnames(corp) <- c('USD', 'KRW')
rownames(corp) <- c('마이크로소프트', '구글', '사우디 아람코', '알리바바', '애플', '아마존')
corp_sum = apply(corp, 2, sum)
corp_mean = apply(corp, 2, mean)
corp_sum
corp_mean
#열별합계답  
#USD   KRW 
#66293  7890 
#열별평균답
#USD      KRW 
#11048.83  1315.00

#2)
corp.rank <- matrix(c("마이크로소프트", "미국", 
                 "구글", "미국", 
                 "사우디 아람코", "사우디", 
                 "알리바바", "중국", 
                 "애플", "미국", 
                 "아마존", "미국" ),
               ncol = 2, byrow = T)
colnames(corp.rank) <- c('기업', '국가')
ranks = rank(corp[,1])
corp.rank = cbind(ranks, corp.rank)

#11번
for(i in 2:1000){
  count=0
  for(j in 1:i){
    if(i%%j==0){
      count=count+1
    }
  }
  if(count==2){
    print(i)
  }
}

#12번
#1)
apply(iris[,1:4], 1, sum)

#2)
apply(iris[,1:4], 2, max)

#13번
res = ifelse(n<0, -n, n)

#14번
tmp=airquality
tmp$Ozone = ifelse(is.na(tmp$Ozone), 0, tmp$Ozone)
tmp$Solar.R = ifelse(is.na(tmp$Solar.R), 0, tmp$Solar.R)
tmp
