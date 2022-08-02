# 02 베이스 R을 이용한 데이터 가공 # 
install.packages("gapminder")
library("gapminder") 
library("dplyr")
glimpse(gapminder)


gapminder[, c("country", "lifeExp")]

gapminder[, c("country", "lifeExp", "year")]

gapminder[1:15, ]

gapminder[gapminder$country == "Croatia", ]


gapminder[gapminder$country == "Croatia", "pop"]


gapminder[gapminder$country == "Croatia", c("lifeExp","pop")]

gapminder[gapminder$country == "Croatia" & gapminder$year > 1990, c("lifeExp","pop")]


apply(gapminder[gapminder$country == "Croatia", c("lifeExp","pop")], 2, mean)


# 03 dplyr 라이브러리를 이용한 데이터 가공 # 
select(gapminder, country, year, lifeExp)

filter(gapminder, country == "Croatia")

summarise(gapminder, pop_avg = mean(pop))

summarise(group_by(gapminder, continent), pop_avg = mean(pop))

summarise(group_by(gapminder, continent, country), pop_avg = mean(pop))

gapminder %>% group_by(continent, country) %>% summarise(pop_avg = mean(pop))


temp1 = filter(gapminder, country == "Croatia")      
temp2 = select(temp1, country, year, lifeExp)  
temp3 = apply(temp2[ , c("lifeExp")], 2, mean)
temp3

gapminder %>% filter(country == "Croatia") %>% select(country, year, lifeExp) %>% summarise(lifeExp_avg = mean(lifeExp))
