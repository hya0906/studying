#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

#define SIZE 512

int main()
{
 char *msg[] = {"Apple is red", "Banana is yellow", "Cherry is red"};
 char buffer[SIZE];
 int filedes[2], nread, i;
 pid_t pid;
 
 if(pipe(filedes) == -1) {
  printf("\nFail to call pipe()\n");
  exit(1);
 }
 if ((pid = fork() ) == -1) {
  printf("\nFail to call fork()\n");
  exit(1);
 } else if (pid > 0) { //parent
   for(i=0;i<3;i++) {
    strcpy(buffer, msg[i]);
    write(filedes[1], buffer, SIZE);
  }
  nread = read(filedes[0], buffer, SIZE);
  printf("\n[Parent] %s (%d)", buffer, nread);
  fflush(stdout);
 
  write(filedes[1], buffer, SIZE);
  printf("\n[Parent] bye!\n");
  fflush(stdout);
 } else { //child
   sleep(3);
   for(i=0;i<3;i++) {
   nread = read(filedes[0], buffer, SIZE);
   printf("\n[Child] %s (%d)", buffer, nread);
   fflush(stdout);
 }
 printf("\n[Child] bye!\n");
 fflush(stdout);
 }
}
