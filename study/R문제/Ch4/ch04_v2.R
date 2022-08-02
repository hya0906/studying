# 01 파일 읽고 쓰기 #


# ENTER 관계없음

# 파일 마지막 행에서 [Enter]를 누르지 않은 경우
students = read.table("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students1.txt", header = T) 

# 파일 마지막 행에서 [Enter]를 누른 경우
students = read.table("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students1.txt",  header = T) 
students

# 읽은 파일의 구조 확인
str(students) 

# 파일을 있는 형태 그대로 읽음 (as.is 비수치형데이터를 factor로 자동해석)
students = read.table("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students1.txt", header = T, as.is = T) 
str(students)

# 파일을 읽을 때 문장을 요인으로 인식하지 않도록 설정
students = read.table("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students1.txt", header = T, stringsAsFactors = F) 
str(students)

# 구분 기호는 쉼표(,), 첫 행은 header로 인식하여 파일을 있는 그대로 읽어들이면 
# NA로 인해 math 요소가 문장으로 인식됨
students = read.table("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students1.txt", sep = ",", header = T, as.is = T) 
str(students)

# "NA" 문장을 결측값 NA로 처리하라고 해도 처리가 안됨. 정확한 문장은 NA 앞에 빈 칸이 있어야 하기 때문
students = read.table("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students1.txt", sep = ",", header = T, as.is = T, na.strings = "NA")  
str(students)

# "NA"로 정확하게 입력하자 결측값 NA로 처리되면서 math 요소가 모두 숫자로 인식됨
students = read.table("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students1.txt", sep = ",", header = T, as.is = T, na.strings = " NA") 
str(students)

# strip.white에서 빈칸을 제거하면 na.string의 기본값이 "NA"로 설정되어 math 요소가 모두 숫자로 인식됨.
students = read.table("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students1.txt", sep = ",", header = T, as.is = T, strip.white = T) 
str(students)

# 첫 행이 header이므로 header 옵션을 지정할 필요가 없음
students = read.csv("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students1.csv") 
students

# 읽은 파일의 구조 확인
str(students) 

# name 속성을 요인에서 문장으로 변경
students$name = as.character(students$name) 
str(students)

# 파일을 읽을 때 문장을 요인으로 인식하지 않도록 설정함
students = read.csv("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students1.csv", stringsAsFactors = FALSE) 
str(students)

students = read.table("C:/Sources/students1.txt", header = T, as.is = T)

# 문장에 큰따옴표가 표시됨.
write.table(students, file = "C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/output.txt") 

# 문장에 큰따옴표되지 않음.
write.table(students, file = "C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4//output.txt", quote = F) 


# 02 데이터 정제를 위한 조건문과 반복문 #

test = c(15, 20, 30, NA, 45)	# 벡터인 경우
test[test<40]	# 값이 40 미만인 요소 추출

test[test%%3!= 0]	# 값이 3으로 나누어 떨어지지 않는 요소 추출


test[is.na(test)]	# NA인 요소 추출


test[!is.na(test)]			# NA가 아닌 요소 추출

test[test%%2==0&!is.na(test)]	# 2의 배수면서 NA가 아닌 요소 추출

characters = data.frame(name = c("길동", "춘향", "철수"), age = c(30, 16, 21), gender = factor(c("M", "F","M")))  
# 데이터 프레임인 경우

characters
                        
                      
characters[characters$gender =="F",]  # 성별이 여성인 행 추출
                
characters[characters$age<30&characters$gender =="M",] # 30살 미만의 남성 행 추출                    
                        

## if else 

x = 5
if(x %% 2 ==0) {
print('x는 짝수')	# 조건식이 참일 때 수행
}   else {
print('x는 홀수')	# 조건식이 거짓일 때 수행
}

# if else if   else

x = -1
if(x>0) {
  print('x is a positive value.')	# x가 0보다 크면 출력
} else if(x<0) {
  print('x is a negative value.')	# 위 조건을 만족하지 않고 x가 0보다 작으면 출력
} else {
  print('x is zero.')		# 위 조건을 모두 만족하지 않으면 출력
}


x = c(-5:5)
options(digits = 3)		# 숫자 표현 시 유효자릿수를 3자리로 설정
sqrt(x)


sqrt(ifelse(x>=0, x, NA))	# NaN이 발생하지 않게 음수면 NA로 표시

options(digits = 5)	

students = read.csv("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students.csv")
students 	     # 데이터에 100 초과 값과 음수 값이 포함되어 있음.

students[, 2] = ifelse(students[, 2]>= 0 & students[, 2]<= 100, students[, 2], NA)
students[, 3] = ifelse(students[, 3]>= 0 & students[, 3]<= 100, students[, 3], NA)
students[, 4] = ifelse(students[, 4]>= 0 & students[, 4]<= 100, students[, 4], NA)
students 	     # ifelse 문으로 2~4열 값 중 0~100 외의 값은 NA로 처리함.

                        
# repeat 문을 이용해 1부터 10까지 숫자 증가시키기
i = 1	             # i의 시작값은 1
repeat {
  if(i>10) {         # i가 10을 넘으면 반복을 중단(break)함
    break
  } else {
    print(i)
    i = i+1           # i를 1 증가시킴.
  }
}



# while 문을 이용해 1부터 10까지 숫자 증가시키기
i = 1 # i의 시작값은 1임.
while(i <= 10){ # i가 10 이하인 동안에 반복함
  print(i)
  i = i+1 # i를 1 증가시킴.
}


# while 문을 이용해 구구단 2단 만들기
i = 1
while(i<10) {
  print(paste(2, "X", i, "=", 2*i))
  i = i+1
}

# for 문을 이용한 1부터 10까지 숫자 증가시키기
for(i in 1:10) {
  print(i)
}  

# for 문을 이용해 구구단 2단 만들기
for(i in 1:9) {
  print(paste(2, "X", i, "=", 2*i))
}

# for 문을 이용해 구구단 2~9단 만들기
for(i in 2:9) {
  for(j in 1:9) {
    print(paste(i, "X", j, "=", i*j))
  }
}

# 1부터 10까지의 수 중 짝수만 출력하기
for(i in 1:10) {
  if(i%%2 == 0) {
    print(i)
  }
}

# 1부터 10까지의 수 중 소수 출력하기
for(i in 1:10) {
  check = 0
  for(j in 1:i) {
    if(i%%j ==0) {
      check = check+1
    }
  }
  if(check ==2) { 
    print(i)
  }
}

students = read.csv("C:/Users/USER/OneDrive - bu.ac.kr/바탕 화면/R문제/Ch4/students.csv")
students	    # 데이터에 100 초과 값과 음수 값이 포함되어 있음


for(i in 2:4) {
  students[, i] = ifelse(students[, i]>= 0 & students[, i]<= 100, students[, i], NA)
}


students	    # ifelse 문으로 2~4열 값 중 0~100 외의 값은 NA로 처리함


# 03 사용자 정의 함수 : 원하는 기능 묶기 # 

fact = function(x) {   # 함수의 이름은 fact, 입력은 x
  fa = 1  # 계승값을 저장할 변수
  while(x>1) {  # x가 1보다 큰 동안 반복
    fa = fa*x   # x 값을 fa에 곱한 후 fa에 다시 저장
    x = x-1  # x 값을 1 감소
  }  
  return(fa)   # 최종 계산된 fa 반환
}
fact(5)	  # 5!을 계산한 결과 출력


my.is.na<-function(x) {	# table(is.na()) 함수를 하나로 묶은 my.is.na 함수를 만듦
  table(is.na(x))
}

my.is.na(airquality)	# 이 결과는 table(is.na(airquality))와 같음.


table(is.na(airquality))


# 04 데이터 정제 예제 1 : 결측값 처리 # 

# is.na 함수를 이용해 결측값 처리하기
str(airquality)	# airquality 데이터의 구조를 살펴봄.

# airquality 데이터에서 NA인 것은 TRUE, 아니면 FALSE로 나타냄. 데이터가 많아 head 함수로 추려냄.
head(is.na(airquality))	
table(is.na(airquality))	# NA가 총 44개 있음.

table(is.na(airquality$Temp))	# Temp에는 NA가 없음을 확인함.

table(is.na(airquality$Ozone))	# Ozone에는 NA가 37개 발견됨.

mean(airquality$Temp)		# NA가 없는 Temp는 평균이 구해짐.

mean(airquality$Ozone)		# NA가 있는 Ozone은 평균이 NA로 나옴.

air_narm = airquality[!is.na(airquality$Ozone), ] # Ozone 속성에서 NA가 없는 값만 추출함. 
air_narm
mean(air_narm$Ozone)	# 결측값이 제거된 데이터에서는 mean 함수가 정상적으로 동작함.

# na.omit 함수를 이용해 결측값 처리하기
air_narm1 = na.omit(airquality)
mean(air_narm1$Ozone)

# 함수 속성인 na.rm을 이용해 결측값 처리하기
mean(airquality$Ozone, na.rm = T)


table(is.na(airquality))

table(is.na(airquality$Ozone))

table(is.na(airquality$Solar.R))

air_narm = airquality[!is.na(airquality$Ozone) & !is.na(airquality$Solar.R), ]
mean(air_narm$Ozone)


# 05 데이터 정제 예제 2 : 이상값 처리 # 

# 이상값이 포함된 환자 데이터
patients = data.frame(name = c("환자1", "환자2", "환자3", "환자4", "환자5"), age = c(22, 20, 25, 30, 27), gender=factor(c("M", "F", "M", "K", "F")), blood.type = factor(c("A", "O", "B", "AB", "C")))
patients

# 성별에서 이상값 제거
patients_outrm = patients[patients$gender=="M"|patients$gender=="F", ]
patients_outrm	


# 성별과 혈액형에서 이상값 제거
patients_outrm1 = patients[(patients$gender == "M"|patients$gender == "F") & (patients$blood.type == "A"|patients$blood.type == "B"|patients$blood.type == "O"|patients$blood.type == "AB"), ]
patients_outrm1	 

# 이상값이 포함된 환자 데이터
patients = data.frame(name = c("환자1", "환자2", "환자3", "환자4", "환자5"), age = c(22, 20, 25, 30, 27), gender = c(1, 2, 1, 3, 2), blood.type = c(1, 3, 2, 4, 5))
patients	

# 성별에 있는 이상값을 결측값으로 변경
patients$gender = ifelse((patients$gender<1|patients$gender>2), NA, patients$gender)
patients	

# 형액형에 있는 이상값도 결측값으로 변경
patients$blood.type = ifelse((patients$blood.type<1|patients$blood.type>4), NA, patients$blood.type)
patients

# 결측값을 모두 제거
patients[!is.na(patients$gender)&!is.na(patients$blood.type), ]

boxplot(airquality[, c(1:4)])    # Ozone, Solar.R, Wind, Temp에 대한 boxplot
boxplot(airquality[, 1])$stats   # Ozone의 boxplot 통계값 계산

air = airquality                 # 임시 저장 변수로 airquality 데이터 복사
table(is.na(air$Ozone))          # Ozone의 현재 NA 개수 확인

# 이상값을 NA로 변경
air$Ozone = ifelse(air$Ozone<1|air$Ozone>122, NA, air$Ozone) 
table(is.na(air$Ozone)) # 이상값 처리 후 NA 개수 확인(2개 증가)

# NA 제거
air_narm = air[!is.na(air$Ozone), ] 
mean(air_narm$Ozone) # 이상값 두 개 제거로 is.na 함수를 이용한 결과보다 값이 줄어듦
