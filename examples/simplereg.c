#define CML_IMPLEMENTATION
#define CML_STRIP_PREFIX
#include "cml.h"

int main(void){
  float x_arr[] =  {1.0f, 2.0f, 3.0f, 5.0f, 10000.0f};
  float y_arr[] = {2.0f, 4.0f, 6.0f, 10.0f, 20000.0f};
  
  train_linear(fit_linear(x_arr, y_arr), 100);
}
