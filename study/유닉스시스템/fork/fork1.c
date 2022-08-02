#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

int main()
{
 int i=1;
 
 pid_t pid = fork();
 
 if (pid==0) {
  printf("\nI am Child process(%d, parent: %d)\n", getpid(), getppid());
 } else if(pid>0) { // parent process(pid<- process id of child)
  printf("\nI am Parent process(%d, parent: %d)\n", getpid(), getppid());
  sleep(2);
 } else { // pid<0
  printf("\nfork() error!!!!\n");
 }
printf("\nEnd(%d)\n", getpid());
}
