#홍윤아_20193882 (2시까지)

#1
#1-1)
plot(cars)

#1-2)
m = lm(dist~speed, data=cars)
coef(m)
#회귀식=> dist=3.932409 * speed - 17.579095

#1-3)
plot(cars)
abline(m, col="red")

#1-4)
coef(m)

#1-5)
newdata=data.frame(speed=c(57))
predict(m, newdata)
#주행속도가 57일때 제동거리: 206.5682


#2
#2-1)
wm = lm(weight~height, data=women)
coef(wm)
#회귀식=> weight=3.45 * height - 87.51667

#2-2)
nhdata=data.frame(height=c(130,141,153))
predict(wm, nhdata)


#3
#3-1)
n_women=rbind(women,c(65.5, 121))
n_women

#3-2)
w_m=lm(weight~height, n_women)

#3-3)
plot(n_women)
abline(w_m, col="red")

#3-4)
summary(w_m)
summary(wm) #문제 2번의 원본데이터 wm의 모델 사용.
#t-value가 크고 p-value가 8.788e-09으로 0.05보다 매우 작게 나왔으므로 
#통계적으로 유의미한 모델임을 알 수 있다.
#원래의 women데이터에서는 p-value는 1.09-14이고 R-squared가 0.9903으로 약 99퍼센트를 설명할 수 있지만만
#이상값을 넣은 데이터에서는 p-value가 8.79e-09이고 R-squared가 0.9059이므로 약 90퍼센트를 설명할 수 있다.
#이상값으로 인해 p-value가 원래 데이터보다 더 커지고 R-squared가 꽤 떨어진 것을 알 수 있다.


#4
#4-1)
n_cars=cars[-c(20,22,23),]
n_cars

#4-2)
cm=lm(dist~speed, data=n_cars)

#4-3)
plot(n_cars)
abline(cm, col="red")

#4-4)
cars
ccm=lm(dist~speed, data=cars) #원본데이터로 훈련시킨 모델
summary(ccm)
summary(cm)
#제거한 데이터들의 모델의 p-value가 1.05e-13으로 0.05보다 작게 나왔으므로 통계적으로 유의미한 모델임을 알 수 있다.
#원본 cars데이터의 p-value는 1.49e-12이고 R-squared값은 0.6438이지만 3개의 샘플을 제거한 모델은 R-squared의 값은 0.7043으로
#나왔고 p-value도 1.053e-13으로 원본데이터의 p-value보다 작으므로 제거를 한 데이터가 더
#유의미한 모델임을 알 수 있다.


#5
#t통계량은 분석하는데 필요한 평균,분산,데이터의 크기의 값을 하나로 묶는 값이다.
#평균이 클수록, 분산이 작을수록, 데이터의 크기는 클수록 신뢰도가 높아진다.
#t의 값이 클수록 효과를 입증하는데 좋다.
#p value는 t-검정을 할때 사용한다. 귀무가설을 가정했을 때 어떤 통계량이
#관측 데이터와 같거나 클 확률을 의미한다. t값이 진짜로 맞는지 t값의 안에 들어올 확률이
#얼마나 되는지에 대한 확률을 의미한다.이 t값은 작아져야 좋다. 그래야 p값보다 커지는 
#경우가 적다는 의미이기 때문이다.


#6
#과잉적합이란 모델이 훈련 집합에 과도하게 적응하여 일반화 능력을 상실하는 현상이다.
#과잉적합에 이르게 되면 훈련데이터에서만 성능이 잘 나오고 새로운 데이터에서는 성능이 
#훈련데이터로 성능측정을 해봤을 때만큼 잘 나오지 않게 된다.


#7
#ANOVA은은 서로 다른 그룹의 평균 또는 산술평균에서 분산값을 비교하는데 사용되는 통계공식이다.
#다양한 시나리오에서 ANOVA를 사용해서 서로 다른 그룹의 평균간의 차이가 있는지 확인할 수 있다.
#결과값은 F통계량으로 나오는데 이 비율은 그룹 내 분산과 그룹간 분산간의 차이를 보여준다. 그룹간의
#유의한 차이가 있는 경우에는 귀무가설이 지원되지 않고 F비율이 더 커진다.
#ANOVA의 유형은 일원분산분석과 이원분산분석이 있다.일원분산분석은 수준이 두개 이상인 독립 변수가 하나만 있는 실험에 적합하다.
#이원분산분석은 두개 이상의 독립변수가 존재할 때 사용한다.
#ANOVA의 한계는 최소 두 그룹의 평균 간에 유의한 차이가 있는지 여부는 알 수 있지만 
#어떤 쌍에서 평균이 다른지는 설명을 하지 못한다.


#8
#8-1)
mtcars
mod = lm(mpg~. ,data=mtcars)
coef(mod)
mod
#mod = (-0.11144048*cyl) + (0.01333524*disp) + (-0.02148212*hp)+ (0.78711097*drat)
# + (-3.71530393*wt) + (0.82104075*qsec) + (0.31776281*vs) + (2.52022689*am) 
# + (0.65541302*gear) + (-0.19941925*carb) + 12.30337

#8-2)
library(MASS)
mod2 <- stepAIC(mod)           
summary(mod2)
summary(mod)
#mod의 R-squared는 0.8066이고 mod2의 R-squared는 0.8336이므로 mod2가 설명력이 더 좋다고 볼 수 있다.


#9
library(dplyr)
library(ggplot2)
trees
tree=select(trees, Girth)
ggplot(tree,aes(Girth), stat="identity")+geom_histogram(fill="steelblue", binwidth = 3.0)+labs(title="트리데이터셋의 나무둘레에 대한 히스토그램", x="나무둘레", y="개수")


#10
library(ggplot2)
mtcars
ggplot(mtcars, aes(x=mpg, y=wt, col=gear))+geom_point()
