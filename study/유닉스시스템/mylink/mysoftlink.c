#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
 if (symlink(argv[1], argv[2]))
    printf("\nsoft-link failed\n");
}

