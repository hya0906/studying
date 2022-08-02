#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main()
{
 printf("\ngetpgrp(): %d", getpgrp());
 printf("\ngetpgid(0): %d", getpgid(0));
 printf("\ngetpgid(getpid()): %d",getpgid(getpid()));
 printf("\n");

} // end main

