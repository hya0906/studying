#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

int num = 0;
int main()
{
 static struct sigaction act;
 void int_handle(int);

 act.sa_handler = int_handle;
 sigfillset(&(act.sa_mask));
 sigaction(SIGINT, &act, NULL);

 while(1){
  printf("\nI am sleepy..");
  sleep(1);
  if (num>=3){
    printf("\n");
    exit(0);
  }
 }
return 0;
}
void int_handle(int signum)
{
 printf("\nSIGINT: %d", signum);
 printf("\nint_handle called %d times", ++num);
}
