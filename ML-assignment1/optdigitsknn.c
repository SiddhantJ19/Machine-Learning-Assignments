#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define NUMBER_OF_ATTRIBUTES 65
#define NUMBER_OF_EXAMPLES 3823
#define NUMBER_OF_CLASSES 10
#define FILENAME "optdigits.tra"
#define NUMBER_OF_VALIDATION_SETS 10
#define ELEMENTS_IN_A_SET ceil(NUMBER_OF_EXAMPLES / NUMBER_OF_VALIDATION_SETS)
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) > 0) ? (x) : (-1 * (x)))
#define K_range 10
#define P_range 5
typedef struct data_node
{
    double attrs[NUMBER_OF_ATTRIBUTES];
    int class;
}data;

typedef struct distance_and_class
{
    double distance;
    int class;
}dac;


void file_to_array(char* filename, data *arr){
    int i, j;
    FILE *fptr = fopen(filename, "r");
    for (i = 0; i < NUMBER_OF_EXAMPLES; i++){
        for (j = 0; j < NUMBER_OF_ATTRIBUTES; j++){
            fscanf(fptr, "%lf", &arr[i].attrs[j]);
        }
    }
    
}

double minkowski_distance(data a, data b, int p){
    double dist_sq = 0;
    int i;
    for (i = 0; i < (NUMBER_OF_ATTRIBUTES - 1); i++){
        double diff = a.attrs[i] - b.attrs[i];
        dist_sq += pow(ABS(diff), p);
    }
    return pow(dist_sq, 1.0/p);
}


int comparator(const void *p, const void *q)  
{ 
    double l = ((dac *)p)->distance;
    double r = ((dac *)q)->distance;
    return (l - r) < 0 ? -1 : 1; 
} 



int predict_class(dac *arr, int k, int size){
    int i, classes[NUMBER_OF_CLASSES + 1];
    for (i = 0; i < NUMBER_OF_CLASSES + 1; i++){
        classes[i] = 0;
    }
    /*sorting to get min distances at the top*/
    qsort((void*)arr, size, sizeof(arr[0]), comparator); 
   
    /*saving the frequency of the classes*/
    for (i = 0; i < k; i++){
        classes[arr[i].class]++;
    }
    
    int max = 0;
    for (i = 0; i < NUMBER_OF_CLASSES + 1; i++){
        if (classes[i] > classes[max]){
            max = i;
        }
    }
    return max;
}

/*The knn_classifier function returns the error for a specific value of k and p.
    The ith loop is for changing the validation set. The jth loop iterates over
    the validation set and the kth loop iterates over training set.
    We calculate the minkowski distance between each vector in the validation set
    and every vector in the training set.*/

int knn_classifier(data *arr, int K, int P){
    int i, j, k;
    int total_error = 0;
    int total = 0;
    for (i = 0; i < NUMBER_OF_VALIDATION_SETS; i++){
        /* The beggining and end of the validation set */
        int begin = i * ELEMENTS_IN_A_SET;
        int end = MIN(begin + ELEMENTS_IN_A_SET, NUMBER_OF_EXAMPLES);

        for (j = begin; j < end; j++){/*predict for every j*/
            int size = NUMBER_OF_EXAMPLES - (end - begin);/* size of the training set*/
            dac dists[size];/* Struct to store minkowski distance and corresponding vector's class in training set*/
            int index = 0;
            for (k = 0; k < begin; k++){
                dists[index].distance = minkowski_distance(arr[j], arr[k], P);
                dists[index].class = arr[k].class;
                index++;
            }
            
            for (k = end; k < NUMBER_OF_EXAMPLES; k++){
                dists[index].distance = minkowski_distance(arr[j], arr[k], P);
                dists[index].class = arr[k].class;
                index++;
            }
            
            /* Comparing the orijnal class and predicted class*/
            int predicted_class = predict_class(dists, K, size);
            total_error += predicted_class == arr[j].class ? 0 : 1;
        }
    }
    return total_error;
}

void shuffle_array(data *arr){
    int i;
    for (i = NUMBER_OF_EXAMPLES - 1; i > 0; i--){
        int swap_pos = rand() % (i + 1);
        data temp = arr[i];
        arr[i] = arr[swap_pos];
        arr[swap_pos] = temp;
    }
}

int main(){
    int i, j, k, p;
    srand((unsigned int)time(NULL));
    data data_array[NUMBER_OF_EXAMPLES];

    file_to_array(FILENAME, data_array); /*importing data from file seeds_dataset.txt*/

    /* Shuffling the dataset 3 times*/   
    shuffle_array(data_array);
    shuffle_array(data_array);
    shuffle_array(data_array);

    /* Storing the corresponding class attribute in the 'class' variable of the struct 'data'*/ 
    for (i = 0; i < NUMBER_OF_EXAMPLES; i++){
        data_array[i].class = (int)data_array[i].attrs[NUMBER_OF_ATTRIBUTES - 1];
    }
    
   
    printf("The following code will run for K=%d and P=%d\n\n", K_range,P_range);
    int error[K_range][P_range];
    /* The following code will loop over for different values of k and p.
        For each pair of k and p, we calculate the error occurs in 
        missclassification. Finally we save the minimum error.*/
    int min_error = NUMBER_OF_EXAMPLES, min_k, min_p;
    for (k = 1; k <= K_range; k++){
        for (p = 1; p <= P_range; p++){
            error[k - 1][p - 1] = knn_classifier(data_array, k, p);
            if (error[k - 1][p - 1] < min_error){
                min_error = error[k - 1][p - 1];
                min_k = k;
                min_p = p;
            }
        }
    }
    /*Displays the accuracy matrix and values of k and p for which accuracy is max.*/
    for (k = 1; k <= K_range; k++){
        for (p = 1; p <= P_range; p++){
            printf("%f\t",100 - error[k - 1][p - 1]*100.0/NUMBER_OF_EXAMPLES);
        }
        printf("\n");
    }
    printf("K = %d\nP = %d\n", min_k, min_p);

    return 0;
}
/*
98.273607       98.587497       98.456709       98.195135       98.195135
97.645828       98.012032       98.168977       97.985875       97.907403
98.352080       98.613654       98.718284       98.665969       98.482867
98.168977       98.299765       98.430552       98.325922       98.352080
97.959717       98.482867       98.613654       98.509024       98.404394
97.828930       98.404394       98.561339       98.325922       98.247450
97.907403       98.509024       98.456709       98.430552       98.195135
97.593513       98.404394       98.247450       98.221292       98.195135
97.698143       98.456709       98.482867       98.299765       98.273607
97.593513       98.273607       98.221292       98.064347       98.195135
K = 3
P = 3
*/