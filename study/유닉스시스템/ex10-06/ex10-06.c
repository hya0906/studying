#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>

int main()
{
 pid_t pid;
 int count=5;

 if((pid==fork()) > 0) {
  sleep(2);
  kill(pid,SIGINT);
  raise(SIGINT);
  printf("\nbye!!\n");
  fflush(stdout);
 } else if(pid==0) {
   printf("\n[Child] count is %d", count--);
   sleep(1);
 } else {
  printf("\nfail to fork()\n");
 }
return 0;
}
