
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <omp.h>

int maxthreads = 14;

void cdist(int d,int na,int nb,float a[na][d],float b[nb][d],float result[na][nb]) {
    int n = na*nb;
#pragma omp parallel for num_threads (maxthreads)
    for(int job=0;job<n;job++) {
        int i = job/nb;
        int j = job%nb;
        double total = 0.0;
        for(int k=0;k<d;k++) {
            float delta = a[i][k]-b[j][k];
            total += delta*delta;
        }
        result[i][j] = sqrt(total);
    }
}
