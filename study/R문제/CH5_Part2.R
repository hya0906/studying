library (dplyr)


# 04 데이터 가공의 실제 # 

############################
## Avocado data in Kaggle
############################

avocado <- read.csv("D:/ch5_part2_data/avocado.csv", header=TRUE, sep = ",")

str(avocado)

#(x_avg = avocado %>% summarize(V_avg = mean(Total.Volume), P_avg = mean(AveragePrice)))
(x_avg = avocado %>% group_by(region) %>% summarize(V_avg = mean(Total.Volume), P_avg = mean(AveragePrice)))
#x_avg = avocado %>% group_by(region, year)
#print(x_avg)
(x_avg = avocado %>% group_by(region, year) %>% summarize(V_avg = mean(Total.Volume), P_avg = mean(AveragePrice)))

x_avg = avocado %>% group_by(region, year, type) %>% summarize(V_avg = mean(Total.Volume), P_avg = mean(AveragePrice))
#print(x_avg)
install.packages("ggplot2")
library(ggplot2)

x_avg %>% filter(region != "TotalUS") %>% ggplot(aes(year, V_avg, col = type)) + geom_line() + facet_wrap(~region)
#x_avg %>% filter(region != "TotalUS")
#print(x_avg)

x_avg_arrange=arrange(x_avg, desc(V_avg))
print(x_avg)
x_avg1 = x_avg %>% filter(region != "TotalUS")

# TotalUS를 제외하고 나면 통계 함수를 직접 사용하여 처리할 수 있음. 

install.packages("lubridate")
library(lubridate)

lubridate_example=(x_avg = avocado %>% group_by(region, year, month(Date), type) %>% summarize(V_avg = mean(Total.Volume), P_avg = mean(AveragePrice)))

################################
### wine Data Manipulation
################################

wine <- read.table("D:/ch5_part2_data/wine.data.txt", header = TRUE, sep = ",")

head(wine)

n = readLines("D:/ch5_part2_data//wine.name.txt")
n

names(wine)[2:14] <- substr(n, 4, nchar(n))
names(wine)


train_set = sample_frac(wine, 0.6)
str(train_set)

test_set = setdiff(wine, train_set)
str(test_set)


###############################################
## Electricity generation data manipulation
###############################################


elec_gen = read.csv("D:/ch5_part2_data/electricity_generation_per_person.csv", header = TRUE, sep = ",")

names(elec_gen)

names(elec_gen)[2:33] = substr(names(elec_gen)[2:33], 2, nchar(names(elec_gen)))
dim(elec_gen)
names(elec_gen)

elec_use = read.csv("D:/ch5_part2_data/electricity_use_per_person.csv", header = TRUE, sep = ",")

names(elec_use)[2:56] = substr(names(elec_use)[2:56], 2, nchar(names(elec_use)[2:56]))

install.packages("tidyr")
library(tidyr)
elec_gen_df = gather(elec_gen, -country, key = "year", value = "ElectricityGeneration")
#print(elec_gen_df)
elec_use_df = gather(elec_use, -country, key = "year", value = "ElectricityUse")
#print(elec_use_df)
elec_gen_use = merge(elec_gen_df, elec_use_df)
#print(elec_gen_use)
