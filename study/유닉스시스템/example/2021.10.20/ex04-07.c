#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
 if(link(argv[1], argv[2]))
   printf("hard-link failed\n");
}
