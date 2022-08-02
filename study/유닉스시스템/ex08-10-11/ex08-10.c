#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(){
 putenv("APPLE=BANNA");
 printf("\n%s) %s",__FILE__, getenv("APPLE"));
 printf("\n");

 execl("./ex08-11", "./ex08-11", (char *)0);
}
