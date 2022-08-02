#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


int main(){
 printf("\n%s) %s",__FILE__, getenv("APPLE"));
 unsetenv("APPLE");

 if(!getenv("APPLE"))
   printf("\n%s) APPLE not found",__FILE__);

 printf("\n");

 return 0;
}




