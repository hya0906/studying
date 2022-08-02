# 4장 단원문제 
# 4-1 

total = 0
for(i in c(1:100)) {
  if (i%%3 == 0 & i%%4 != 0) {
    total=total + i }
} 

total

# 4-2

times = function(x, n) {
  total = 0
  for(i in c(1:n)) {
    if (i%%x == 0) {
      total=total + i }
  } 
  return(total)
}
times(3, 100) # 1부터 100까지 수 중 3의 배수의 합을 구함