#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// #include <math.h>

// make an allocator for this later
typedef struct{
  float* x;
  float* y;
  size_t size;
} sample;

float normalizef(float array[], size_t size){
  float highest = array[0];
  float lowest = array[0];

  for( size_t i=0; i < size; ++i){
   if (array[i] > highest){
      highest = array[i];
    }
    if (array[i] < lowest){
      lowest = array[i];
    }
  }
    for (size_t i = 0; i < size; ++i) {
        array[i] = (array[i] - lowest) / (highest - lowest);
    }
        return *array;
  }

#define normalize(arr) normalizef((arr), sizeof(arr) / sizeof((arr)[0]))
// there are probably better ways to do this but it is what it is 

#define LEARNING_R 0.1f
#define OUTPUT 10

float random_float(){
    return (float) rand() / (float) RAND_MAX;
}

int main(int argc, char** argv){
  int EPOCHS;
  if (argc < 2){
    EPOCHS = 100;
    printf("Using EPOCH default: %d\n", EPOCHS);
    printf("<%s> EPOCH_COUNT\n", argv[0]);
  } else {
    EPOCHS = atoi(argv[1]);
  }

  float x_arr[] =  {1.0f, 2.0f, 3.0f, 5.0f, 10000.0f};
  float y_arr[] = {2.0f, 4.0f, 6.0f, 10.0f, 20000.0f};
  normalize(x_arr);
  normalize(y_arr);

  sample timestwo = {
    .x = x_arr,
    .y = y_arr,
    .size = sizeof(x_arr)/sizeof(x_arr[0])
  };

  srand(time(0));
  float weight = random_float();
  for (int epoch = 0; epoch < EPOCHS; ++epoch){
    float total_loss = 0.0f;
    float grad = 0.0f;

    for (int i=0; i < timestwo.size; ++i){
      float x = timestwo.x[i]; // make a variadic macro to generalize the way this works
      float y = timestwo.y[i];
      float y_hat = x * weight;
      float error = y_hat - y; 
      total_loss += error*error;
      grad += 2*error*x;
    }
    total_loss /= timestwo.size;
    grad /= timestwo.size;
    
    weight -= LEARNING_R*grad;
    if (epoch % (EPOCHS/OUTPUT) == 0){
      printf("Epoch: %d Loss: %.6e Weight: %f\n", epoch, total_loss, weight);
    }
    
  }
    printf("\nTrained weight: %f\n", weight);
  
  for (int i = 0; i < timestwo.size; ++i) {
      float x = timestwo.x[i];
      float y = timestwo.y[i];
      float y_pred = weight * x;
      printf("x = %f, y = %f, y_pred = %f\n", x, y, y_pred);
  }
  return 0;
}
