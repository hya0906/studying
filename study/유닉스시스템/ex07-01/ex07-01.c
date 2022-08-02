#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

void cleanupaction(void);

int main()
{
 pid_t pid;
 int i;

 for (i=0;i<3;i++) {
  printf("\nbefore fork [%d]", i);
  fflush(stdout);
  sleep(1);
 } // end for
 //i <- 3

 pid = fork(); // <-------create child

 if(pid>0) { //Parent
  for (;i<7;i++) { // i <- 3,4,5,6
   printf("\nParent [%d]", i);
   fflush(stdout);
   sleep(1);
  } //end for
  atexit(cleanupaction);

 } else if (pid == 0) { //Child
  for(;i<5;i++) {
   printf("\nChild [%d]", i);
   fflush(stdout);
   sleep(1);
   execl("/bin/ls", "ls", "-l", (char *)0);
  } // end for

 } else { //fail to fail
  printf("\nFail to fork child process");
 }
 exit(0);
} // end main

void cleanupaction(void)
{
 printf("\nclean-up-action\n");
}

