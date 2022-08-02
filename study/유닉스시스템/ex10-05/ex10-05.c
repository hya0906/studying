#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>

int main()
{
 sigset_t set;
 int count = 3;

 sigemptyset(&set);
 sigaddset(&set, SIGINT);

 sigprocmask(SIG_BLOCK, &set, NULL);

 while (count) {
  printf("\ndon't distrub me (%d)", count--);
  fflush(stdout);
  sleep(1);
 }
 sigprocmask(SIG_UNBLOCK, &set, NULL);
 sleep(1);
 printf("\nyou did not disturb me!!\n");
 fflush(stdout);
}
