#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 18

__global__ void sum(double *a, double *b, double *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}


void printSchema(int threadPerBlock[]) {
	int schema = 1;

	do {
		int block = 0;
		int thread = 0;

		printf("\n  Schema with %i Thread/Block", threadPerBlock[schema]);

		printf("\n |");
		for (int i = 0; i < N; i++) {
			printf("====|");
		}

		printf("\n | ");
		for (int i = 0; i < N; i++) {
			printf("%2.1d | ", i);
		}
		
		printf("\n |");
		for (int i = 0; i < N; i++) {
			printf("====|");
		}

		printf("\n | ");
		for (int i = 0; i < N; i++) {
			if (thread < threadPerBlock[schema]) {
				printf("%2.1d | ", thread);
				thread = thread + 1;
			}
			if (thread == threadPerBlock[schema]) {
				thread = 0;
				block = block + 1;
			}
		}

		printf("\n |");
		for (int i = 0; i < N; i++) {
			printf("====|");
		}

		printf("\n\n");
		schema = schema + 1;
	} while (schema <= 3);

}

int main() {
	int size = N * sizeof(double*);
	double *a = (double*)malloc(size);
	double *b = (double*)malloc(size);
	double *c = (double*)malloc(size);
	double *d_a, *d_b, *d_c;

	int schema = 3;
	int threadPerBlock[4];
	threadPerBlock[1] = 3;
	threadPerBlock[2] = 6;
	threadPerBlock[3] = 9;

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	for (int i = 0; i < N; i++) {
		a[i] = i * (3. / 4.);
		b[i] = i * (1. / 4.);
	}

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	sum << < N / threadPerBlock[schema], threadPerBlock[schema] >> > (d_a, d_b, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	printSchema(threadPerBlock);

	printf(" Results Summation \n");
	printf(" |======|=============|=============|=============| \n");
	printf(" |  No  |      A      |      B      |  C = A + B  | \n");
	printf(" |======|=============|=============|=============| \n");
	for (int i = 0; i < N; i++) {
		printf(" | %4.1d | %8.2f    | %8.2f    | %8.2f    | \n", i, a[i], b[i], c[i]);
	}
	printf(" |======|=============|=============|=============| \n");

	free(a); cudaFree(d_a);
	free(b); cudaFree(d_b);
	free(c); cudaFree(d_c);

	getchar();

}