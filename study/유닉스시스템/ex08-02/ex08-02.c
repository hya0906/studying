#include <unistd.h>
#include <sys/wait.h>
#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h>

int main()
{
 pid_t pid;
 int status;

 pid = fork();
 
 if (pid>0) {
  printf("\nParent: waiting\n");
  fflush(stdout);
  wait(&status);
  printf("\nParent: status is %d\n", (status >> 8));
  fflush(stdout);
 } else if (pid == 0) {
   sleep(1);
   printf("\nChild: bye~\n");
   exit(127);
 } else {
   printf("\nFail to fork()\n");
   fflush(stdout);
 }
 printf("\nbye!");
 fflush(stdout);
return 0;
}
