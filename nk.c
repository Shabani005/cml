#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define K_NUM 5


typedef struct{
  float *features;
  int label;
} point;

typedef struct{
  point* points;
  size_t size;
  size_t dim;
} dataset;

typedef struct{
  point point;
  float dist;
} tosort;

int compare_tosort(const void *a, const void *b) {
    float d1 = ((tosort *)a)->dist;
    float d2 = ((tosort *)b)->dist;
    return (d1 > d2) - (d1 < d2);
}

float distance_vec(point a, point b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a.features[i] - b.features[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

void shuffle_dataset(point *arr, size_t n) {
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        point tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

int knn_predict(dataset train, point test_pt, int num_classes){
  int *label_count = calloc(num_classes, sizeof(int));
  tosort* sarray = malloc(sizeof(tosort) * train.size);

  for (int i=0; i < train.size; ++i){
    sarray[i].point = train.points[i];
    sarray[i].dist = distance_vec(test_pt, train.points[i], train.dim);
  }
  qsort(sarray, train.size, sizeof(tosort), compare_tosort);
  
  for (size_t i = 0; i < K_NUM; ++i){
    label_count[sarray[i].point.label]++;
  }

  int predicted_label = -1;
  int max_count = -1;

  for (int lbl = 0; lbl < num_classes; ++lbl){
    if (label_count[lbl] > max_count){
    max_count = label_count[lbl];
    predicted_label = lbl;
    }
  }
  return predicted_label;
}

/* int main(){
  point data[] = {
    {2.0, 4.0, 0},
    {2.0, 3.0, 0},
    {6.0, 1.0, 0},
    {9.0, 1.0, 0},
    {7.0, 2.0, 1},
    {6.0, 2.0, 1},
    {3.0, 4.0, 1},
    {5.0, 4.0, 1},
    {4.0, 4.0, 1},
    {12.0, 1.0, 1}
  };
  
  dataset df = {data, sizeof(data)/sizeof(data[0])};
  
  point test_pt = {2, 1, -1};

  for (size_t i=0; i < df.size; ++i){
    printf("distance between test_pt and point %zu = %f\n",i+1, distance_vec2(test_pt, df.points[i]));
  }
}*/

int main() {
    srand(time(NULL));

    size_t dim = 4; // number of features for Iris

float iris_data[][4] = {
    {5.1, 3.5, 1.4, 0.2}, {4.9, 3.0, 1.4, 0.2}, {4.7, 3.2, 1.3, 0.2},
    {4.6, 3.1, 1.5, 0.2}, {5.0, 3.6, 1.4, 0.2}, {5.4, 3.9, 1.7, 0.4},
    {4.6, 3.4, 1.4, 0.3}, {5.0, 3.4, 1.5, 0.2}, {4.4, 2.9, 1.4, 0.2},
    {4.9, 3.1, 1.5, 0.1}, {5.4, 3.7, 1.5, 0.2}, {4.8, 3.4, 1.6, 0.2},
    {4.8, 3.0, 1.4, 0.1}, {4.3, 3.0, 1.1, 0.1}, {5.8, 4.0, 1.2, 0.2},
    {5.7, 4.4, 1.5, 0.4}, {5.4, 3.9, 1.3, 0.4}, {5.1, 3.5, 1.4, 0.3},
    {5.7, 3.8, 1.7, 0.3}, {5.1, 3.8, 1.5, 0.3}, {5.4, 3.4, 1.7, 0.2},
    {5.1, 3.7, 1.5, 0.4}, {4.6, 3.6, 1.0, 0.2}, {5.1, 3.3, 1.7, 0.5},
    {4.8, 3.4, 1.9, 0.2}, {5.0, 3.0, 1.6, 0.2}, {5.0, 3.4, 1.6, 0.4},
    {5.2, 3.5, 1.5, 0.2}, {5.2, 3.4, 1.4, 0.2}, {4.7, 3.2, 1.6, 0.2},
    {4.8, 3.1, 1.6, 0.2}, {5.4, 3.4, 1.5, 0.4}, {5.2, 4.1, 1.5, 0.1},
    {5.5, 4.2, 1.4, 0.2}, {4.9, 3.1, 1.5, 0.1}, {5.0, 3.2, 1.2, 0.2},
    {5.5, 3.5, 1.3, 0.2}, {4.9, 3.1, 1.5, 0.1}, {4.4, 3.0, 1.3, 0.2},
    {5.1, 3.4, 1.5, 0.2}, {5.0, 3.5, 1.3, 0.3}, {4.5, 2.3, 1.3, 0.3},
    {4.4, 3.2, 1.3, 0.2}, {5.0, 3.5, 1.6, 0.6}, {5.1, 3.8, 1.9, 0.4},
    {4.8, 3.0, 1.4, 0.3}, {5.1, 3.8, 1.6, 0.2}, {4.6, 3.2, 1.4, 0.2},
    {5.3, 3.7, 1.5, 0.2}, {5.0, 3.3, 1.4, 0.2},

    {7.0, 3.2, 4.7, 1.4}, {6.4, 3.2, 4.5, 1.5}, {6.9, 3.1, 4.9, 1.5},
    {5.5, 2.3, 4.0, 1.3}, {6.5, 2.8, 4.6, 1.5}, {5.7, 2.8, 4.5, 1.3},
    {6.3, 3.3, 4.7, 1.6}, {4.9, 2.4, 3.3, 1.0}, {6.6, 2.9, 4.6, 1.3},
    {5.2, 2.7, 3.9, 1.4}, {5.0, 2.0, 3.5, 1.0}, {5.9, 3.0, 4.2, 1.5},
    {6.0, 2.2, 4.0, 1.0}, {6.1, 2.9, 4.7, 1.4}, {5.6, 2.9, 3.6, 1.3},
    {6.7, 3.1, 4.4, 1.4}, {5.6, 3.0, 4.5, 1.5}, {5.8, 2.7, 4.1, 1.0},
    {6.2, 2.2, 4.5, 1.5}, {5.6, 2.5, 3.9, 1.1}, {5.9, 3.2, 4.8, 1.8},
    {6.1, 2.8, 4.0, 1.3}, {6.3, 2.5, 4.9, 1.5}, {6.1, 2.8, 4.7, 1.2},
    {6.4, 2.9, 4.3, 1.3}, {6.6, 3.0, 4.4, 1.4}, {6.8, 2.8, 4.8, 1.4},
    {6.7, 3.0, 5.0, 1.7}, {6.0, 2.9, 4.5, 1.5}, {5.7, 2.6, 3.5, 1.0},
    {5.5, 2.4, 3.8, 1.1}, {5.5, 2.4, 3.7, 1.0}, {5.8, 2.7, 3.9, 1.2},
    {6.0, 2.7, 5.1, 1.6}, {5.4, 3.0, 4.5, 1.5}, {6.0, 3.4, 4.5, 1.6},
    {6.7, 3.1, 4.7, 1.5}, {6.3, 2.3, 4.4, 1.3}, {5.6, 3.0, 4.1, 1.3},
    {5.5, 2.5, 4.0, 1.3}, {5.5, 2.6, 4.4, 1.2}, {6.1, 3.0, 4.6, 1.4},
    {5.8, 2.6, 4.0, 1.2}, {5.0, 2.3, 3.3, 1.0}, {5.6, 2.7, 4.2, 1.3},
    {5.7, 3.0, 4.2, 1.2}, {5.7, 2.9, 4.2, 1.3}, {6.2, 2.9, 4.3, 1.3},
    {5.1, 2.5, 3.0, 1.1}, {5.7, 2.8, 4.1, 1.3},

    {6.3, 3.3, 6.0, 2.5}, {5.8, 2.7, 5.1, 1.9}, {7.1, 3.0, 5.9, 2.1},
    {6.3, 2.9, 5.6, 1.8}, {6.5, 3.0, 5.8, 2.2}, {7.6, 3.0, 6.6, 2.1},
    {4.9, 2.5, 4.5, 1.7}, {7.3, 2.9, 6.3, 1.8}, {6.7, 2.5, 5.8, 1.8},
    {7.2, 3.6, 6.1, 2.5}, {6.5, 3.2, 5.1, 2.0}, {6.4, 2.7, 5.3, 1.9},
    {6.8, 3.0, 5.5, 2.1}, {5.7, 2.5, 5.0, 2.0}, {5.8, 2.8, 5.1, 2.4},
    {6.4, 3.2, 5.3, 2.3}, {6.5, 3.0, 5.5, 1.8}, {7.7, 3.8, 6.7, 2.2},
    {7.7, 2.6, 6.9, 2.3}, {6.0, 2.2, 5.0, 1.5}, {6.9, 3.2, 5.7, 2.3},
    {5.6, 2.8, 4.9, 2.0}, {7.7, 2.8, 6.7, 2.0}, {6.3, 2.7, 4.9, 1.8},
    {6.7, 3.3, 5.7, 2.1}, {7.2, 3.2, 6.0, 1.8}, {6.2, 2.8, 4.8, 1.8},
    {6.1, 3.0, 4.9, 1.8}, {6.4, 2.8, 5.6, 2.1}, {7.2, 3.0, 5.8, 1.6},
    {7.4, 2.8, 6.1, 1.9}, {7.9, 3.8, 6.4, 2.0}, {6.4, 2.8, 5.6, 2.2},
    {6.3, 2.8, 5.1, 1.5}, {6.1, 2.6, 5.6, 1.4}, {7.7, 3.0, 6.1, 2.3},
    {6.3, 3.4, 5.6, 2.4}, {6.4, 3.1, 5.5, 1.8}, {6.0, 3.0, 4.8, 1.8},
    {6.9, 3.1, 5.4, 2.1}, {6.7, 3.1, 5.6, 2.4}, {6.9, 3.1, 5.1, 2.3},
    {5.8, 2.7, 5.1, 1.9}, {6.8, 3.2, 5.9, 2.3}, {6.7, 3.3, 5.7, 2.5},
    {6.7, 3.0, 5.2, 2.3}, {6.3, 2.5, 5.0, 1.9}, {6.5, 3.0, 5.2, 2.0},
    {6.2, 3.4, 5.4, 2.3}, {5.9, 3.0, 5.1, 1.8}
};

int iris_labels[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0,

    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1,

    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2
};
    size_t total_size = sizeof(iris_labels) / sizeof(iris_labels[0]);

    point *points = malloc(total_size * sizeof(point));
    if (!points) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < total_size; i++) {
        points[i].features = malloc(dim * sizeof(float));
        for (size_t j = 0; j < dim; j++) {
            points[i].features[j] = iris_data[i][j];
        }
        points[i].label = iris_labels[i];
    }

    // Shuffle before splitting
    shuffle_dataset(points, total_size);

    size_t train_size = (size_t)(total_size * 0.7);
    dataset train = {points, train_size, dim};
    dataset test = {points + train_size, total_size - train_size, dim};

    int correct = 0;
    for (size_t i = 0; i < test.size; ++i) {
        int pred = knn_predict(train, test.points[i], 3);
        printf("Test point (");
        for (size_t j = 0; j < dim; j++) {
            printf("%.1f%s", test.points[i].features[j],
                   (j < dim - 1) ? ", " : "");
        }
        printf(") true=%d pred=%d\n", test.points[i].label, pred);
        if (pred == test.points[i].label) correct++;
    }

    printf("\nAccuracy: %.2f%%\n", (100.0 * correct) / test.size);

    // Free memory
    for (size_t i = 0; i < total_size; i++) {
        free(points[i].features);
    }
    free(points);

    return 0;
}
