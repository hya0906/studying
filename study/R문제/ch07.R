# 03 단순 선형 회귀 #

x = c(3.0, 6.0, 9.0, 12.0)	# 설명 변수 x
y = c(3.0, 4.0, 5.5, 6.5)	# 반응 변수 y
m = lm(y ~ x)		# 모델 적합(학습)
m			# m 출력


plot(x, y)
abline(m, col = 'red')

coef(m)       # 매개변수(계수) 값을 알려줌
fitted(m)     # 훈련집합에 있는 샘플에 대한 예측값
residuals(m)  # 잔차를 알려줌
deviance(m)   # 잔차제곱합를 알려줌
deviance(m)/length(x) # 잔차제곱합을 평균제곱오차로 변환하여 출력

summary(m)	# 모델의 상세 분석

newx = data.frame(x = c(1.2, 2.0, 20.65))	 # 3개의 새로운 값
predict(m, newdata = newx)		 # 예측 수행

nx=c(1.2,2.0,20.65)
ny=c(2.23,2.55,10.01)
plot(nx,ny,col='red', cex=2,pch=20)
abline(m)


# 04 단순 선형 회귀의 적용 : cars 데이터 #
str(cars)
head(cars)
plot(cars)


car_model = lm(dist~speed, data = cars)
coef(car_model)
abline(car_model, col = 'red')

fitted(car_model)
residuals(car_model)


nx1 = data.frame(speed = c(21.5))
predict(car_model, nx1)
nx2 = data.frame(speed = c(25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0))
predict(car_model, nx2)



nx = data.frame(speed = c(21.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0))
plot(nx$speed, predict(car_model, nx), col ='red', cex = 2, pch = 20)
abline(car_model)

plot(cars, xlab ='속도', ylab = '거리')      # cars 데이터를 그림
x = seq(0, 25, length.out = 200)            # 예측할 지점  0부터 25까지 200개 숫자 뽑음
for(i in 1:4) {  
  m = lm(dist~poly(speed, i), data = cars)
  assign(paste('m', i, sep = '.'), m)        # i차 모델 m을 m.i라 부름
  lines(x, predict(m, data.frame(speed = x)), col = i) # m으로 예측한 결과를 겹쳐 그림 
}

# 분산 분석
anova(m.1, m.2, m.3, m.4)                 


# 05 모델의 통계량 해석 #
# women 데이터에 적용
str(women)
women


women_model = lm(weight ~ height, data = women)
coef(women_model)

plot(women)
abline(women_model, col = 'red')

summary(women_model)


car_model = lm(dist ~ speed, data = cars)
summary(car_model)


# 06 다중 선형 회귀 #

install.packages('scatterplot3d')
library(scatterplot3d)
x = c(3.0, 6.0, 3.0, 6.0)
u = c(10.0, 10.0, 20.0, 20.0)
y = c(4.65, 5.9, 6.7, 8.02)
scatterplot3d(x, u, y, xlim = 2:7, ylim = 7:23, zlim = 0:10, pch = 20, type = 'h')


m = lm(y ~ x + u)
coef(m)

s = scatterplot3d(x, u, y, xlim = 2:7, ylim = 7:23, zlim = 0:10, pch = 20, type = 'h')
s$plane3d(m)

fitted(m)
residuals(m) 
deviance(m)           # 잔차 제곱합
deviance(m)/length(x) # 평균 제곱 오차 : MSE(Mean Squared Error)

nx = c(7.5, 5.0)
nu = c(15.0, 12.0)
new_data = data.frame(x = nx, u = nu)          
new_data
ny = predict(m, new_data)
ny


s = scatterplot3d(trees$Girth, trees$Height, trees$Volume, pch = 20, type = 'h', angle = 55)
s$plane3d(m)

s = scatterplot3d(nx, nu, ny, xlim=0:10, ylim=7:23, zlim=0:10, pch = 20, type = 'h', angle = 55)
s$plane3d(m)

# 07 다중 선형 회귀의 적용 : trees 데이터 #
str(trees)
summary(trees)
scatterplot3d(trees$Girth, trees$Height, trees$Volume)
m = lm(Volume ~ Girth + Height, data = trees)
m
s = scatterplot3d(trees$Girth, trees$Height, trees$Volume, pch = 20, type = 'h', angle = 55)
s$plane3d(m)

ndata = data.frame(Girth = c(8.5, 13.0, 19.0), Height = c(72, 86, 85))
predict_trees=predict(m, newdata = ndata)
predict_trees


# [더 알아보기] t-검정과 분산 분석 #
# 세 종류의 데이터
data1 = c(30, -5, 55, -30, -20, 45)
data2 = c(12, 13, 12, 13, 12, 13)
data3 = c(30, -5, 55, -30, -20, 45, 30, -5, 55, -30, -20, 45)
# 평균과 표준편차
mean(data1)
sd(data1)
mean(data2)
sd(data2)
mean(data3)
sd(data3)
t.test(data1)
t.test(data2)
t.test(data3)

