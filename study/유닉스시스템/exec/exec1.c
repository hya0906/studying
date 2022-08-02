#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdlib.h>

int main()
{
 pid_t pid;
 int status;
 
 printf("\nHello\n");
 pid = fork();

 if(pid>0) { // parent process(pid: child process id)
   printf("\nParent\n");
   wait(&status);
   printf("\nstatus: %d\n", status >>8);//to make it right
   //sleep(1);

  } else if (pid == 0) { //child process
    printf("\nChild\n");
    //execl("/bin/ls", "ls","-l", (char *)0);
    execl("/bin/ls1", "ls1","-l", (char *)0);
    printf("\nFail to execute ls\n");
    exit(20); // 20<<8 shift left
   } else { // error: fork()
    printf("\nFail to fork\n");
   }
   printf("\nBye\n");
}
