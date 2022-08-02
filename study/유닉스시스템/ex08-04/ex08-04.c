#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main()
{
 pid_t pid;
 int status = 0;

 if((pid=fork())> 0) { //parent
  while (!waitpid(pid, &status, WNOHANG)) {
    printf("\nParent: %d", status++);
    fflush(stdout);
    sleep(1);
  }
  printf("\nParent: child - exit(%d)", status>>8);
 } else if (pid ==0) { // child
   sleep(5);
   printf("\nbye");
   fflush(stdout);
   exit(9);
 } else printf("\nfail to fork()");
 fflush(stdout);
 printf("\n");
}

