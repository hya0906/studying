#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>

void timeover(int signum) //signal handler function
{
 printf("\n\ntime over!!!\n\n");
 fflush(stdout);
 exit(0);
}

int main()
{
 char buf[1024];
 char *alpha = "abcdefghijklmnopqrstuvwxyz";

 int timelimit;
 struct sigaction act;

 act.sa_handler = timeover;
 sigaction(SIGALRM, &act, NULL);

 printf("\nInput timelimit(sec)..\n");
 scanf("%d", &timelimit);

 alarm(timelimit);

 printf("\nSTART!!\n");
 scanf("%s", buf);

 if(!strcmp(buf, alpha))
  printf("\nWell done.. you succeed!!\n");
 else
  printf("\nSorry.. you fail!\n");
return 0;
}
