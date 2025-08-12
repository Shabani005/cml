#include <stddef.h>
#include <math.h>
#include <stdio.h>

typedef struct{
  float x;
  float y;
  int label;
} point;

typedef struct{
  point* points;
  size_t size;
} dataset;

float distance(point a, point b){
  return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}



int main(){
  point data[] = {
    {1.0, 2.0, 0},
    {2.0, 3.0, 0},
    {3.0, 3.0, 1},
    {5.0, 1.0, 1}
  };
  
  dataset df = {data, sizeof(data)/sizeof(data[0])};
  
  point test_pt = {2.5, 2.5, 1};

  for (size_t i=0; i < df.size; ++i){
    printf("distance between test_pt and point %zu = %f\n",i+1, distance(test_pt, df.points[i]));
  }
}
