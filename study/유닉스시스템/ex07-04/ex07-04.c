#include <stdio.h>
#include <unistd.h>

int main()
{
 char *arg[] = {"ls", "-l", (char *)0};

 printf("\nbefore executing ls -l\n");

 execv("/bin/ls", arg);

 printf("\nafter executing ls -l\n");

 return 0;
}
