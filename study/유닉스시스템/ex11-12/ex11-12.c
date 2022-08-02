#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

int main()
{
 int p[2];
 pid_t pid;

 if (pipe(p) == -1) {
  printf("\nfail to call pipe()\n");
  exit(1);
 }
 if ((pid=fork()) == -1) {
  printf("\nfail to call fork()\n");
  exit(1);
 }
 else if(pid>0) { //Parent
  printf("\n[Parent]\n");
  close(p[0]);
  dup2(p[1], 1);
  execlp("ls", "ls", "-al", (char *)0);
  printf("\n[Parent] fail to call execlp()\n");
 }
 else { //child
  printf("\n[Child]\n");
  close(p[1]);
  dup2(p[0], 0);
  execlp("wc", "wc", (char *)0);
  printf("\n[Child] fail to call execlp()\n");
 }
return 0;
}
