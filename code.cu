#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define BLOCK_SIZE 512
#define MAT_DIM 10000

__constant__ float c_weights[9]; // The weights for the convolution.

__device__ float calculateNewValue(float* matrix, int i) {
	// Perform the convolution.
	float sum = 0;
	int row = i / MAT_DIM;
	int col = i % MAT_DIM;
	
	// Add the upper row.
	if (row > 0) {
		if (col > 0)
			sum += matrix[i - MAT_DIM - 1] * c_weights[0];

		sum += matrix[i - MAT_DIM] * c_weights[1];

		if (col < MAT_DIM-1)
			sum += matrix[i - MAT_DIM + 1] * c_weights[2];
	}

	// Add the middle row.
	if (col > 0)
		sum += matrix[i-1] * c_weights[3];

	sum += matrix[i] * c_weights[4];

	if (col < MAT_DIM-1)
		sum += matrix[i+1] * c_weights[5];

	// Add the bottom row.
	if (row < MAT_DIM-1) {
		if (col > 0)
			sum += matrix[i + MAT_DIM - 1] * c_weights[6];

		sum += matrix[i + MAT_DIM] * c_weights[7];

		if (col < MAT_DIM-1)
			sum += matrix[i + MAT_DIM + 1] * c_weights[8];
	}

	return sum;
}

__global__ void convolution(float* inputMatrix, float* outputMatrix, float* diagonal) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= MAT_DIM * MAT_DIM)
		return;
	
	float newValue = calculateNewValue(inputMatrix, i);
	outputMatrix[i] = newValue;

	// If the element is in the diagonal, add it to the array.
	if ( i % (MAT_DIM+1) == 0 )
		diagonal[i / (MAT_DIM+1)] = newValue;
}

__global__ void getMaxDiagonalElement(float* matrix, float* maxArr, int* maxIdx, int problemSize) {

	// Create the shared memory for the current block.
	__shared__ float sArr[BLOCK_SIZE*2];	// The matrix elements for the current block.
	__shared__ int sIdx[BLOCK_SIZE*2];		// The element indices for the current block.

	// Generate the ids.
	int gId = blockIdx.x * blockDim.x + threadIdx.x;
	int tId = threadIdx.x;

	// Check if this thread has any job to do.
	if (gId*2 >= problemSize)
		return;

	// Populate the shared memory.
	sIdx[tId*2] = maxIdx[gId*2];
	sArr[tId*2] = matrix[gId*2];
	if (gId+1 < problemSize) {
		sArr[tId*2+1] = matrix[gId*2+1];
		sIdx[tId*2+1] = maxIdx[gId*2+1];
	}

	// Synchronize threads for this block after filling the shared memory.
	__syncthreads();
	
	for (unsigned int s = 1; s < blockDim.x*2; s*=2) {
		int index = 2 * s * tId;
		
		if (index < blockDim.x*2) {

			// Check if the element after the stride is bigger and if so, store it at the current spot along with its index.
			if (sArr[index] < sArr[index+s]) {
				sArr[index] = sArr[index+s];
				sIdx[index] = sIdx[index+s];
			}

		}

		// Synchronize threads before increasing the stride.
		__syncthreads();
	}
	
	// Have one thread update the global memory for the CPU to read.
	if (tId == 0) {
		maxArr[blockIdx.x] = sArr[0];
		maxIdx[blockIdx.x] = sIdx[0];
	}

}

void getCPUmax(float*, float*, int*);
void fillMatrix(float*);

int main() {

	// Timing setup.
	cudaEvent_t start,stop;
    float elapsedTime, elapsedTime2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	float* d_matrix = NULL;
	float* d_matrix2 = NULL;
	float* d_diagonal = NULL;

	float* h_matrix = (float*)malloc(MAT_DIM * MAT_DIM * sizeof(float));
	if (h_matrix == NULL) {
		fprintf(stderr, "Unable to allocate memory for the matrix.\n");
		return 1;
	}
	float* h_diagonal = (float*)malloc(MAT_DIM * sizeof(float));
	if (h_diagonal == NULL) {
		fprintf(stderr, "Unable to allocate memory for the diagonal.\n");
		free(h_matrix);
		return 2;
	}

	// Populate the matrix with random data.
	fillMatrix(h_matrix);

	// Create the weights.
	float weights[9];
	weights[0] = 0.2;	weights[1] = 0.5;	weights[2] = -0.8;
	weights[3] = -0.3;	weights[4] = 0.6;	weights[5] = -0.9;
	weights[6] = 0.4;	weights[7] = 0.7;	weights[8] = 0.1;

	// Move the weights to the GPU's constant memory.
	cudaMemcpyToSymbol(c_weights, weights, 9 * sizeof(float)); 

	// Allocate memory on the GPU.
	if (cudaMalloc((void**)&d_matrix, MAT_DIM*MAT_DIM * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Unable to allocate memory on the GPU.\n");
		free(h_matrix);
		free(h_diagonal);
		return 3;
	}
	if (cudaMalloc((void**)&d_matrix2, MAT_DIM*MAT_DIM * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Unable to allocate memory on the GPU.\n");
		cudaFree(d_matrix);
		free(h_matrix);
		free(h_diagonal);
		return 4;
	}
	if (cudaMalloc((void**)&d_diagonal, MAT_DIM * sizeof(float)) != cudaSuccess) {
		fprintf(stderr, "Unable to allocate memory on the GPU.\n");
		cudaFree(d_matrix);
		free(h_matrix);
		free(h_diagonal);
		return 5;
	}

	// Copy data to GPU memory.
	cudaMemcpy(d_matrix, h_matrix, MAT_DIM*MAT_DIM*sizeof(float), cudaMemcpyHostToDevice);
	
	// Call the convolution kernel.
	int gridSize = (MAT_DIM*MAT_DIM-1) / BLOCK_SIZE + 1;

	cudaEventRecord(start,0);
	convolution<<<gridSize, BLOCK_SIZE>>>(d_matrix, d_matrix2, d_diagonal);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);

	// Copy the result to the CPU and deallocate the matrices on the GPU since they are no longer needed.
	cudaMemcpy(h_matrix, d_matrix2, MAT_DIM*MAT_DIM * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_matrix);
	cudaFree(d_matrix2);

	int problemSize = MAT_DIM; // The size of the current reduction problem.
	gridSize = (problemSize/2-1) / BLOCK_SIZE + 1;

	// Allocate an array on the GPU for keeping the max element.
	float* d_maxArr = NULL;
	int* d_maxIdx = NULL;
	cudaMalloc((void**)&d_maxArr, gridSize * sizeof(float));
	cudaMalloc((void**)&d_maxIdx, MAT_DIM * sizeof(int));
	
	// Create a temporary array to copy to the GPU.
	int* temp = (int*)malloc(MAT_DIM * sizeof(int));
	for (int i = 0; i < MAT_DIM; i++)
		temp[i] = i;
	cudaMemcpy(d_maxIdx, temp, MAT_DIM * sizeof(int), cudaMemcpyHostToDevice);
	free(temp);

	// Start the reduction of the diagonal for the max element.
	cudaEventRecord(start, 0);
	while(1) {
		gridSize = (problemSize/2-1) / BLOCK_SIZE + 1;

		getMaxDiagonalElement<<<gridSize, BLOCK_SIZE>>>(d_diagonal, d_maxArr, d_maxIdx, problemSize);

		problemSize = gridSize;

		// If the problem wasn't solved, copy the data to the diagonal array and repeat the procedure.
		if (problemSize > 1)
			cudaMemcpy(d_diagonal, d_maxArr, gridSize * sizeof(float), cudaMemcpyDeviceToDevice);
		else
			break;
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime2, start, stop);

	// Find the maximum element and its index using the CPU (used for evaluation only).
	int cpuIdx;
	float cpuMax;
	getCPUmax(h_matrix, &cpuMax, &cpuIdx);

	// Retrieve the maximum element and its index from the GPU.
	float max;
	int idx;
	cudaMemcpy(&max, d_maxArr, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&idx, d_maxIdx, sizeof(int), cudaMemcpyDeviceToHost);

	// Print results.
	printf("Convolution time: %.3fms\n", elapsedTime);
	printf("Max value time: %.3fms\n", elapsedTime2);
	printf("GPU max value: %.1f\n", max);
	printf("GPU max value index: %d\n", idx);
	printf("CPU max value: %.1f\n", cpuMax);
	printf("CPU max value index: %d\n", cpuIdx);


	// Deallocate memory.
	cudaFree(d_maxArr);
	cudaFree(d_maxIdx);
	cudaFree(d_diagonal);
	free(h_matrix);
	free(h_diagonal);
	return 0;
}

void fillMatrix(float* mat) {
	srand(time(0));
	for (int i = 0; i < MAT_DIM * MAT_DIM; i++)
		mat[i] = (float)(rand() % 100 + 1);
}

void getCPUmax(float* m, float* max, int* idx) {
	*max = m[0];
	*idx = 0;
	for (int i = 1; i < MAT_DIM; i++) {
		int index = i * (MAT_DIM+1);
		if (m[index] > *max) {
			*max = m[index];
			*idx = i;
		}
	}
}