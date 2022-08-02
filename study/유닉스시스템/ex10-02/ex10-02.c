#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

int main()
{
 sigset_t set;
 int result;

 sigemptyset(&set);
 result = sigismember(&set,SIGALRM);
 printf("\nSIGALRM is %s a member", result?"":"not");

 sigaddset(&set,SIGALRM);
 result = sigismember(&set,SIGALRM);
 printf("\nSIGALRM is %s a member", result?"":"not");

 sigfillset(&set);
 result = sigismember(&set,SIGCHLD);
 printf("\nSIGCHLD is %s a member", result?"":"not");

 sigdelset(&set,SIGCHLD);
 result = sigismember(&set,SIGCHLD);
 printf("\nSIGCHLD is %s a member", result?"":"not");
 printf("\n");
}


