#include <stdio.h>
#include <stdlib.h>
#include <time.h>


typedef float sample[2];

sample and_gate[] =
{
//  x     y
  {1.0f, 2.0f},
  {2.0f, 4.0f},
  {3.0f, 6.0f},
  {5.0f, 10.0f}
};

// normalize data later. allow for arbirtary input size

#define N_SAMPLES (sizeof(and_gate)/sizeof(and_gate[0]))
#define LEARNING_R 0.01f
#define EPOCHS 100

float random_float(){
    return (float) rand() / (float) RAND_MAX;
}

int main(void){
  srand(time(0));
  float weight = random_float();
  for (int epoch = 0; epoch < EPOCHS; ++epoch){
    float total_loss = 0.0f;
    float grad = 0.0f;

    for (int i=0; i < N_SAMPLES; ++i){
      float x = and_gate[i][0];
      float y = and_gate[i][1];
      float y_hat = x * weight;
      float error = y_hat - y; //maybe do abs later
      total_loss += error*error;
      grad += 2*error*x;
    }
    total_loss /= N_SAMPLES;
    grad /= N_SAMPLES;
    
    weight -= LEARNING_R*grad;
    if (epoch % EPOCHS/10 == 0){
      printf("Epoch: %d Loss: %f Weight: %f\n", epoch, total_loss, weight);
    }
  }
  printf("\nTrained weight: %f\n", weight);
  for (int i = 0; i < N_SAMPLES; ++i) {
      float x = and_gate[i][0];
      float y = and_gate[i][1];
      float y_pred = weight * x;
      printf("x = %f, y = %f, y_pred = %f\n", x, y, y_pred);
  }
  return 0;
}
