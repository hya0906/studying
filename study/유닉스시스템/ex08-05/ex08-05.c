#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

int main()
{
 pid_t pid;

 if (( pid=fork())>0) { //parent      mecro->this is replaced to this file name
   printf("[%s] PPID:%d, PID:%d\n", __FILE__, getppid(), getpid());
   fflush(stdout);
   sleep(1);
 } else if (pid == 0) { //child
   printf("[%s] PPID:%d, PID:%d\n", __FILE__, getppid(), getpid());
   fflush(stdout);
   execl("./ex08-06", "./ex08-06", (char *)0);
 } else printf("\nfail to fork()");

 return 0;
}
