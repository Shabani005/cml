#define CML_IMPLEMENTATION
#define CML_STRIP_PREFIX
#include "cml.h"

#define EPOCHS 100000

int main(){
  float x_arr[] =  {1.0f, 2.0f, 3.0f, 5.0f, 10000.0f};
  float y_arr[] = {4.0f, 8.0f, 12.0f, 20.0f, 40000.0f};
  train_linear(fit_linear(x_arr, y_arr), EPOCHS);
}
