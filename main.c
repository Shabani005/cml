#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


typedef float sample[5];

sample and_gate[] =
{
  {1.0f, 2.0f, 3.0f, 5.0f, 10000.0f}, // x
  {2.0f, 4.0f, 6.0f, 10.0f, 20000.0f}, // y
  };

// I may not need to specify size and use the sizeof trick inside the function
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

// normalize data later. allow for arbirtary input size

#define N_SAMPLES 4 // auto later 
#define LEARNING_R 0.1f
#define EPOCHS 100000
#define OUTPUT 10

float random_float(){
    return (float) rand() / (float) RAND_MAX;
}

int main(void){
  normalize(and_gate[0]);
  normalize(and_gate[1]);
  srand(time(0));
  float weight = random_float();
  for (int epoch = 0; epoch < EPOCHS; ++epoch){
    float total_loss = 0.0f;
    float grad = 0.0f;

    for (int i=0; i < N_SAMPLES; ++i){
      float x = and_gate[0][i]; // make a variadic macro to generalize the way this works
      float y = and_gate[1][i];
      float y_hat = x * weight;
      float error = y_hat - y; //maybe do abs later
      total_loss += error*error;
      grad += 2*error*x;
    }
    total_loss /= N_SAMPLES;
    grad /= N_SAMPLES;
    
    weight -= LEARNING_R*grad;
    if (epoch % (EPOCHS/OUTPUT) == 0){
      printf("Epoch: %d Loss: %.6e Weight: %f\n", epoch, total_loss, weight);
    }
    
    //if ( epoch+1/EPOCHS == 1 ){
      //printf("Last Epoch\n=======\nEpoch: %d Loss: %f Weight: %f\n", epoch, total_loss, weight);
      //break;
    //}
  }
    printf("\nTrained weight: %f\n", weight);
  
  for (int i = 0; i < N_SAMPLES; ++i) {
      float x = and_gate[0][i];
      float y = and_gate[1][i];
      float y_pred = weight * x;
      printf("x = %f, y = %f, y_pred = %f\n", x, y, y_pred);
  }
  return 0;
}


