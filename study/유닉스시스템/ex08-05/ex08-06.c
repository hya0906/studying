#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

int main()
{
 printf("\n[%s] PPID:%d, PID:%d", __FILE__, getppid(), getpid());
 printf("\n");
 fflush(stdout);
}
