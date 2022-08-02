# CH 3 : R의 데이터형과 연산

# 02:변수 #

# 76P~ #

x = 1   # x에 1을 할당 
y = 2   # y에 2를 할당
z = x + y
z
x + y = z
z <- x + y
z

# 77 페이지, 그림 3-1 #

x = 1
y = 2
temp = x
x = y
y = temp
x
y

# 77 페이지, 변수의 과거와 현재 #

x = 1
x
x = 2
x
x = x + 1
x
x = x +2
x

# 78 페이지, NOTE : 대입 연산자의 차이 #

# 함수에 <- 사용하여 값 할당 시도

x = function(){a <- 1}
x()
a

# <- : cannot be used within function

y = function(){b <<- 2}
y()
b     # 함수에 선언된 변수이지만 함수 호출 후 외부에서 읽을 수 있음 

# <<- : can be used within function, hardly used


# 78~79 페이지 #

a = 1
a
A

# distinguish capital

a@ <- 2

# special character cannot be used as variable

initial_value = 1
initial_value

blood.type = c("A", "B", "O", "AB")
blood.type

1a = 1
_a1 = 1
a_1 = 1
a_1
b2 = 2
b2

if = 1
for = 2

# if, for, while..etc cannot be used as variable

# 03 데이터형 #

# 80P~ #

x = 5
y = 2
x/y

xi = 1 + 2i
yi = 1 - 2i
xi+yi

str = "Hello, World!"
str

blood.type = factor(c('A', 'B', 'O', 'AB'))
blood.type

T  # CAPITAL 로 쓰여야 함
F

xinf = Inf  # I 가 대문자여야 무한대로 인식을 함
yinf = -Inf


# 82P~ #

is.integer(1)
is.numeric(1)

is.integer(1L)  # L: Literal : 고정된 값(fixed value) 표기법 
is.numeric(1L)

x = 1 		# x에 단순히 1을 넣은 경우 x는 숫자형 
x
is.integer(x)
x = 1L 		# x에 1L을 입력한 경우 x는 정수형 
x
is.integer(x)
x = as.integer(1) 	# x에 1을 as.integer 함수로 변환하여 입력한 경우 x는 정수형 
x
is.integer(x)


# 05 벡터 vector #

# 86~87 페이지 #

1:7 		# 1부터 7까지 1씩 증가시켜 요소가 7개인 벡터 생성
7:1 		# 7부터 1까지 1씩 감소시켜 요소가 7개인 벡터 생성 

vector(length = 5)
c(1:5)	 	# 1~5 요소로 구성된 벡터 생성. 1:5와 동일
c(1, 2, 3, c(4:6)) 	# 1~3 요소와 4~6 요소를 결합한 1~6 요소로 구성된 벡터 생성

x = c(1, 2, 3) 	# 1~3 요소로 구성된 벡터를 x에 저장
x 		# x 출력

y = c() 		# y를 빈 벡터로 생성
y = c(y, c(1:3)) 	# 기존 y 벡터에 c(1:3) 벡터를 추가해 생성
y 		# y 출력 

seq(from = 1, to = 10, by = 2) 	# 1부터 10까지 2씩 증가하는 벡터 생성
seq(1, 10, by = 2) 			# 1부터 10까지 2씩 증가하는 벡터 생성 
seq(0, 1, by = 0.1) 			# 0부터 1까지 0.1씩 증가하는 요소가 11개인 벡터 생성
seq(0, 1, length.out = 11) 		# 0부터 1까지 요소가 11개인 벡터 생성 

rep(c(1:3), times = 2)		# (1, 2, 3) 벡터를 2번 반복한 벡터 생성
rep(c(1:3), each = 2) 		# (1, 2, 3) 벡터의 개별 요소를 2번 반복한 벡터 생성

# 88~89P , 벡터 연산 vector calculation # 

x = c(2, 4, 6, 8, 10)
length(x) 		
x[1] 		
x[1, 2, 3] 		# Error
x[c(1, 2, 3)] 	
x[-c(1, 2, 3)] 	
x[c(1:3)] 		

x = c(1, 2, 3, 4)
y = c(5, 6, 7, 8)
z = c(3, 4)
w = c(5, 6, 7)

# calculation

x+2 		
x + y 		
x + z 		
x + w 

x = 1:10
x >5 		
all(x>5) 	
any(x>5) 	
x = 1:10

head(x) 	
tail(x) 		
head(x, 3) 	
tail(x, 3) 		

x = c(1, 2, 3)
y = c(3, 4, 5)
z = c(3, 1, 2)

# Set calculation btw vectors

union(x, y) 	
intersect(x, y) 	

setdiff(x, y) 
setdiff(y, x) 

setequal(x, y) 	
setequal(x, z) 	


# 91P~ 배열(행렬) , array

x = array(1:5, c(2, 4)) 
x

x[1, ] 
x[, 2] 

dimnamex = list(c("1st", "2nd"), c("1st", "2nd", "3rd", "4th")) 
x = array(1:5, c(2, 4), dimnames = dimnamex)

x
x["1st", ]
x[, "4th"]

# 2차원 배열 생성 (92P)

x = 1:12
x

matrix(x, nrow = 3) # 행의 갯수를 지정해 줌
matrix(x, nrow = 3, byrow = T)

# 벡터를 묶어 배열 생성

v1 = c(1, 2, 3, 4)
v2 = c(5, 6, 7, 8)
v3 = c(9, 10, 11, 12)

cbind(v1, v2, v3) # 열 단위로 묶어 배열 생성

rbind(v1, v2, v3) # 행 단위로 묶어 배열 생성

# 배열 연산 (94P)

# 표 3-7 연산자를 활용한 다양한 행렬 연산
# 2X2 행렬 2개를 각각 x,y에 저장


x = array(1:4, dim = c(2, 2))
y = array(5:8, dim = c(2, 2))
x
y

x + y
x - y

x * y  # 각 열별 곱셈

x %*% y # 수학적 행렬 곱셈

t(x) # x의 전치 행렬

x

solve(x) # x의 역행렬

det(x) # x의 행렬식


# 배열에 유용한 함수 (95P~)

x = array(1:12, c(3, 4))
x

apply(x, 1, mean) # 1 이면 함수를 행별로 적용

apply(x, 2, mean) # 2 이면 함수를 열별로 적용

x = array(1:12, c(3, 4))
dim(x)

x = array(1:12, c(3, 4))

sample(x) # 배열 요소를 임의로 섞어 추출

sample(x, 10) # 배열 요소 중 10개를 골라 추출

sample(x, 10, prob = c(1:12)/24) # 각 요소별 추출 확률을 달리 할 수 있음

sample(10) # 단순히 숫자만 사용하여 샘플을 만들 수 있음



# 07 DATA frame   97P~
name = c("철수", "춘향", "길동")
age = c(22, 20, 25)
gender = factor(c("M", "F", "M"))
blood.type = factor(c("A", "O", "B"))

patients = data.frame(name, age, gender, blood.type)
patients

# 다음과 같이 한 행으로도 작성 가능

patients1 = data.frame(name = c("철수", "춘향", "길동"), age = c(22, 20, 25), gender = factor(c("M", "F", "M")), blood.type = factor(c("A", "O", "B")))
patients1
patients1$name # name 속성 출력

patients1[1, ] # 1행 값 출력

patients1[, 2] # 2열 값 출력

patients1[3, 1] # 3행 1렬 값 출력

patients1[patients1$name=="철수", ] # 환자 중 철수에 대한 정보 출력

patients1[patients1$name=="철수", c("name", "age")] # 철수 이름과 나이만 출력

# 99P

head(cars) 
speed

attach(cars) # attach 함수를 통해 cars의 각 속성을 변수로 이용하게 함
speed 

detach(cars) # detach 함수를 통해 cars의 각 속성을 변수로 사용하는 걸 해제함
speed # speed 변수에 접근하지만, 해당 변수가 없음


# 데이터 속성을 이용해 함수 적용

mean(cars$speed)
max(cars$speed)

# with 함수를 이용해 함수적용
with(cars, mean(speed))
with(cars, max(speed))

# 속도가 20 초과인 데이터만 추출
subset(cars, speed > 20)

# 속도가 20 초과인 dist 데이터만 추출, 여러 열 선택은 c() 안을 , 로 구분
subset(cars, speed > 20, select = c(dist))

# 속도가 20 초과인 데이터 중 dist를 제외한 데이터만 추출

subset(cars, speed > 20, select = -c(dist))

# na.omit function

head(airquality) # airquality 데이터에는 NA 포함되어 있음 
head(na.omit(airquality)) # NA가 포함된 값을 제외하여 추출함

# merge : 102P~

name = c("철수", "춘향", "길동")
age = c(22, 20, 25)
gender = factor(c("M", "F", "M"))
blood.type = factor(c("A", "O", "B"))
patients1 = data.frame(name, age, gender)
patients1

patients2 = data.frame(name, blood.type)
patients2

patients = merge(patients1, patients2, by = "name")
patients


# 이름이 같은 열 변수가 없다면, merge 함수의 by.x와 by.y가 합칠 때
# 사용할 열의 속성명을 각각 기입해주어야 함

name1 = c("철수", "춘향", "길동")
name2 = c("민수", "춘향", "길동")
age = c(22, 20, 25)
gender = factor(c("M", "F", "M"))
blood.type = factor(c("A", "O", "B"))
patients1 = data.frame(name1, age, gender)
patients1

patients2 = data.frame(name2, blood.type)
patients2

patients = merge(patients1, patients2, by.x = "name1", by.y = "name2")
patients

patients = merge(patients1, patients2, by.x = "name1", by.y = "name2", all = TRUE)
patients

# 104p~

x = array(1:12, c(3, 4))
x
is.data.frame(x)  # 현재는 X가 DF 아님 
as.data.frame(x)

# is.data.frame 함수를 호출하는 것만으로 x가 DF으로 바뀌지 않음
is.data.frame(x)
# as.data.frame 함수로 x를 데이터 프레임 형식으로 변환

x = as.data.frame(x)
x

is.data.frame(x)

# is.data.frame 함수 호출만으로는 x가 DF로 바뀌지 않음

names(x) = c("1st", "2nd", "3rd", "4th")
x


# 08 LIST #

patients = data.frame(name = c("철수", "춘향", "길동"), age = c(22, 20, 25), gender = factor(c("M", "F", "M")), blood.type = factor(c("A", "O", "B")))
no.patients = data.frame(day = c(1:6), no = c(50, 60, 55, 52, 65, 58))


# 데이터를 단순 추가

listPatients = list(patients, no.patients) 
listPatients


# 각 데이터에 이름을 부여하면서 추가

listPatients = list(patients=patients, no.patients = no.patients) 
listPatients

listPatients$patients		# 요소명 입력

listPatients[[1]]				# 인덱스 입력

listPatients[["patients"]]			# 요소명을 " "에 입력


listPatients[["no.patients"]]		# 요소명을 " "에 입력


# 108P~
# no.patients 요소의 평균을 구해줌. 

lapply(listPatients$no.patients, mean) 

# patients 숫자 형태가 아닌 것은 평균이 구해지지 않음
lapply(listPatients$patients, mean) 

sapply(listPatients$no.patients, mean) 

# sapply() 의 simplify 옵션을 F로 하면 lapply() 결과와 동일한 결과를 반환함

sapply(listPatients$no.patients, mean, simplify = F) 
