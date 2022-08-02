#include <stdio.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>

#define MSGSIZE 16

int main()
{
 int p1[2],p2[2]; //for pipe
 char msg[MSGSIZE];
 int i;
 pid_t pid1, pid2; //for fork
 fd_set initset, newset; //for select
 
 pid1 = pid2=0;
 
 if(pipe(p1) == -1) {
  printf("\nfail to call pipe() #1\n");
  exit(1);
 }
 if(pipe(p2) == -1) {
  printf("\nfail to call pipe() #1\n");
  exit(1);
 }
 
 if((pid1=fork()) == -1) { //P -> C1
  printf("\nfail to call fork() #1\n");
  exit(1);
 }
 if(pid1>0)
 if((pid2==fork()) == -1) { // P -> C2
  printf("\nfail to call fork() #2\n");
  exit(1);
 }

 if((pid1>0)&&(pid2>0)) { //Parent
  printf("\nParent: %d\n", getpid());
  close(p1[1]); close(p2[1]);

  FD_ZERO(&initset);       // initset = {     }
  FD_SET(p1[0], &initset); // initset = { p1[0] }
  FD_SET(p2[0], &initset); // initset = { p1[0], p2[0] }
  
  newset = initset; // newset = { p1[0], p2[0] }
  while (select(p2[0]+1, &newset, NULL, NULL, NULL) > 0) {
   if (FD_ISSET(p1[0], &newset)) // newset = { p1[0] }
      if( read(p1[0], msg, MSGSIZE)>0)
         printf("\n[Parent] %s from child1\n", msg);
   if (FD_ISSET(p2[0], &newset)) // newset = { p2[0] }
      if( read(p2[0], msg, MSGSIZE)>0)
         printf("\n[Parent] %s from child2\n", msg);
   newset = initset;
  }// end while
 } // end if
 else if ((pid1 == 0) && (pid2 == 0)) { //C1(first child)
  printf("\nChild1: %d\n", getpid());
  close(p1[0]); close(p2[0]); close(p2[1]);
 
  for (i=0;i<3;i++) {
   sleep((i+1)%4);
   printf("\nChild1: send message %d\n", i);
   write(p1[1], "I am child1", MSGSIZE);
  } //end for
  printf("\nChild1; bye!\n");
  exit(0);
  }
  else if((pid1>0)&&(pid2==0)) { // C2(second child)
  printf("\nChild2: %d\n", getpid());
  close(p1[0]); close(p2[0]); close(p1[1]);
 
  for (i=0;i<3;i++) {
   sleep((i+3)%4);
   printf("\nChild2: send message %d\n", i);
   write(p2[1], "I am child2", MSGSIZE);
  } //end for
  printf("\nChild2; bye!\n");
  exit(0);
 }
}
