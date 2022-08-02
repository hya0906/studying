#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main()
{
 pid_t pid;

 printf("\nHello");
 fflush(stdout);

 pid=fork();
 
 if(pid>0) { //parent process
  printf("\nparent");
  sleep(1);
 } else if (pid==0) {//child process
   printf("\nchild");
   fflush(stdout);
   execl("/bin/ls", "ls", "-l", (char *)0);
   printf("\nfail to execute ls -l");
 } else
   printf("\nparent: fail to fork");

printf("\nBye");
}
