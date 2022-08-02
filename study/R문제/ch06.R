# 01 데이터 시각화란? #

# 평균
apply(anscombe, 2, mean)

# 분산
apply(anscombe, 2, var)


# 상관관계(상관계수)
cor(anscombe$x1, anscombe$y1)

cor(anscombe$x2, anscombe$y2)

cor(anscombe$x3, anscombe$y3)

cor(anscombe$x4, anscombe$y4)

library(gapminder)
library(dplyr)
gapminder
y <- gapminder %>% group_by(year, continent) %>% summarize(c_pop = sum(pop)) 
head(y, 20)

plot(y$year, y$c_pop)
plot(y$year, y$c_pop, col = y$continent)
plot(y$year, y$c_pop, col = y$continent, pch = c(1:5))
plot(y$year, y$c_pop, col = y$continent, pch = c(1:length(levels(y$continent))))

# 범례 개수를 숫자로 지정
legend("topleft", legend = 5, pch = c(1:5), col = c(1:5)) 

# 범례 개수를 데이터 개수에 맞게 지정
legend("topleft", legend = levels((y$continent)), pch = c(1:length(levels(y$continent))), col = c(1:length(levels(y$continent))))


# 02 시각화의 기본 기능 #
plot(gapminder$gdpPercap, gapminder$lifeExp, col = gapminder$continent)
legend("bottomright", legend = levels((gapminder$continent)), pch = c(1:length(levels(gapminder$continent))), col = c(1:length(levels(y$continent))))

plot(log10(gapminder$gdpPercap), gapminder$lifeExp, col = gapminder$continent)
legend("bottomright", legend  = levels((gapminder$continent)), pch = c(1:length(levels(gapminder$continent))), col = c(1:length(levels(y$continent))))

install.packages("ggplot2")
library(ggplot2)
ggplot(gapminder, aes(x =  gdpPercap, y = lifeExp, col = continent)) + geom_point() + scale_x_log10()

ggplot(gapminder, aes(x =  gdpPercap, y = lifeExp, col = continent, size = pop)) + geom_point() + scale_x_log10()

ggplot(gapminder, aes(x = gdpPercap, y = lifeExp, col = continent, size = pop)) + geom_point(alpha = 0.5) + scale_x_log10()

ggplot(gapminder, aes(x=gdpPercap, y=lifeExp, col=continent, size=pop)) + geom_point(alpha=0.5) + scale_x_log10() + facet_wrap(~year)

gapminder %>% filter(year == 1952 & continent =="Asia") %>% ggplot(aes(reorder(country, pop), pop)) + geom_bar(stat = "identity") + coord_flip()

gapminder %>% filter(year==1952 & continent== "Asia") %>% ggplot(aes(reorder(country, pop), pop)) + geom_bar(stat  = "identity") + scale_y_log10() + coord_flip()

gapminder %>% filter(country == "Korea, Rep.") %>% ggplot(aes(year, lifeExp, col = country)) + geom_point() + geom_line()

gapminder %>% ggplot(aes(x = year, y = lifeExp, col = continent)) + geom_point(alpha = 0.2) + geom_smooth()

x = filter(gapminder, year == 1952)
hist(x$lifeExp, main = "Histogram of lifeExp in 1952")

x %>% ggplot(aes(lifeExp)) + geom_histogram()

x %>% ggplot(aes(continent, lifeExp)) + geom_boxplot()

plot(log10(gapminder$gdpPercap), gapminder$lifeExp)



# 03 시각화 도구 #

head(cars)

# type = "p"는 점 플롯, main = "cars"는 그래프의 제목
plot(cars, type  = "p", main  = "cars")

plot(cars, type = "l", main = "cars")       # type ="l"은 선을 사용한 플롯
plot(cars, type="b", main="cars")   # type ="b"는 점과 선을 모두 사용한 플롯
plot(cars, type = "h", main = "cars")  # type ="h"는 히스토그램과 같은 막대 그래프

x = gapminder %>% filter(year == 1952 & continent == "Asia") %>% mutate(gdp = gdpPercap*pop) %>% select(country, gdp) %>% arrange(desc(gdp)) %>% head()
pie(x$gdp, x$country)
barplot(x$gdp, names.arg = x$country)


x = gapminder %>% filter(year == 2007 & continent == "Asia") %>% mutate(gdp  = gdpPercap*pop) %>% select(country, gdp) %>% arrange(desc(gdp)) %>% head()
pie(x$gdp, x$country)
barplot(x$gdp, names.arg = x$country)


matplot(iris[, 1:4], type = "l")
legend("topleft", names(iris)[1:4], lty = c(1, 2, 3, 4), col = c(1, 2, 3, 4))



hist(cars$speed)


ggplot(gapminder, aes(x = gdpPercap, y = lifeExp, col = continent)) + geom_point(alpha = 0.2)

gapminder %>% filter(lifeExp>70) %>% group_by(continent) %>% summarize(n = n_distinct(country)) %>% ggplot(aes(x = continent, y = n)) + geom_bar(stat = "identity")


gapminder %>% filter(year == 2007) %>%  ggplot(aes(lifeExp, col = continent)) + geom_histogram()
gapminder %>% filter(year == 2007) %>% ggplot(aes(lifeExp, col = continent)) + geom_histogram(position = "dodge")

gapminder %>% filter(year == 2007) %>% ggplot(aes(continent, lifeExp, col = continent)) + geom_boxplot()

ggplot(gapminder, aes(x = gdpPercap, y = lifeExp, col = continent)) + geom_point(alpha = 0.2)
ggplot(gapminder, aes(x = gdpPercap, y = lifeExp, col = continent)) + geom_point(alpha = 0.2) + scale_x_log10() # 가로축을 로그 스케일로 변환함.


gapminder %>% filter(continent == "Africa") %>% ggplot(aes(country, lifeExp)) + geom_bar(stat  =  "identity")                  # [그림 6-35(a)]

gapminder %>% filter(continent == "Africa") %>% ggplot(aes(country, lifeExp)) + geom_bar(stat  =  "identity") + coord_flip()    # [그림 6-35(b)] 플롯의 방향을 전환함. 


install.packages("RColorBrewer")
library(RColorBrewer)

display.brewer.all()


# [그림 6-37(a)] : 기본 팔레트를 적용한 그래프
gapminder %>% filter(lifeExp>70) %>% group_by(continent) %>% summarize(n  = n_distinct(country)) %>% ggplot(aes(x = continent, y = n)) + geom_bar(stat = "identity", aes(fill = continent)) 

# [그림 6-37(b)] : Spectral 팔레트를 적용한 그래프
gapminder %>% filter(lifeExp>70) %>% group_by(year, continent) %>% summarize(n = n_distinct(country)) %>% ggplot(aes(x = continent, y = n)) + geom_bar(stat = "identity", aes(fill = continent)) + scale_fill_brewer(palette = "Spectral")

# [그림 6-37(c)] Blues 팔레트를 적용한 그래프
gapminder %>% filter(lifeExp>70) %>% group_by(continent) %>% summarize(n = n_distinct(country)) %>% ggplot(aes(x = continent, y = n)) + geom_bar(stat = "identity", aes(fill = continent)) + scale_fill_brewer(palette = "Blues")

# [그림 6-37(d)] Oranges 팔레트를 적용한 그래프
gapminder %>% filter(lifeExp>70) %>% group_by(continent) %>% summarize(n =  n_distinct(country)) %>% ggplot(aes(x = continent, y = n)) + geom_bar(stat = "identity", aes(fill =  continent)) + scale_fill_brewer(palette = "Oranges")


# reorder(continent, -n)은 continent를 n을 기준으로 내림차 순으로 정렬하라는 의미
gapminder %>% filter(lifeExp >70) %>% group_by(continent) %>% summarize(n  =  n_distinct(country)) %>% ggplot(aes(x = reorder(continent, -n), y =  n)) + geom_bar(stat = "identity", aes(fill =  continent)) + scale_fill_brewer(palette  = "Blues")


# 04 시각화를 이용한 데이터 탐색 #

gapminder %>% ggplot(aes(gdpPercap, lifeExp, col = continent)) + geom_point(alpha  =  0.2) + facet_wrap(~year) + scale_x_log10()

gapminder %>% filter(year == 1952 & gdpPercap > 10000 & continent == "Asia") 

gapminder %>% filter(country == "Kuwait") %>% ggplot(aes(year, gdpPercap)) + geom_point() + geom_line()             # [그림 6-40(a)]
gapminder %>% filter(country == "Kuwait") %>% ggplot(aes(year, pop)) + geom_point() + geom_line()                   # [그림 6-40(b)]

gapminder %>% filter(country == "Korea, Rep.") %>% ggplot(aes(year, gdpPercap)) + geom_point() + geom_line()        # [그림 6-41(a)]
gapminder %>% filter(country == "Korea, Rep.") %>% ggplot(aes(year, pop)) + geom_point() + geom_line()              # [그림 6-41(b)]

gapminder %>% filter(country == "Kuwait" | country == "Korea, Rep.") %>% mutate(gdp = gdpPercap*pop) %>% ggplot(aes(year, gdp, col = country)) + geom_point() + geom_line()

# [그림 6-43(a)] gdpPercap의 변화 비교 
gapminder %>% filter(country == "Kuwait"|country == "Saudi Arabia"|country == "Iraq"|country == "Iran"|country == "Korea, Rep."|country == "China"|country == "Japan")  %>% ggplot(aes(year, gdpPercap, col = country)) + geom_point() + geom_line()

# [그림 6-43(b)] pop의 변화 비교 
gapminder %>% filter(country == "Kuwait"|country=="Saudi Arabia"|country == "Iraq"|country == "Iran"|country == "Korea, Rep."|country == "China"|country == "Japan")  %>% ggplot(aes(year, pop, col=country)) + geom_point() + geom_line()

# [그림 6-43(c)] gdp의 변화 비교 
gapminder %>% filter(country == "Kuwait"|country == "Saudi Arabia"|country == "Iraq"|country == "Iran"|country == "Korea, Rep."|country == "China"|country == "Japan")  %>% mutate(gdp=gdpPercap*pop) %>% ggplot(aes(year, gdp, col = country)) + geom_point() + geom_line() + scale_y_log10()

