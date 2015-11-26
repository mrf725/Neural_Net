/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Student: Mike Fehr
 * Instructor: Bryan Mills, University of Pittsburgh
 * MPI neural net.
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "timer.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define TAG 7
#define TRAINING_SAMPLES 8
#define TRAINING_TARGETS 9


/* weights between neurons for each layer
   # Hidden Layers, # Neurons, #Neurons inputs from
   previous layer
 i = index for each layer
 j = index of each neuron for that layer
 k = index of weight from neuron k to i */
float* hiddenWeights;

/* weight of the bias term per each hidden layer */
float*  hiddenNodeBias;

/* output at each hidden node
   # Hidden Layer, # Neurons*/
float* hiddenNodeOutput;


/* weight between last hidden layer and output
   #Neuron, #outputs */
/* i - ith output node
   j - jth input from last hidden layer */
float*  outputWeights;
float*  outputBias;

/* weight from input to hidden layer
 * i = output node (hidden layer)
 * j = input node */
float*  inputWeights;
float numInputNodes;

/* the output value at each output neuron */
float* outputOutputs;
int numOutputNodes;
int neuronsPerLayer;
int numHiddenLayers;
float learningRate;


int* trainingSamples;
int* trainingTargets;
int* testSamples;
int numTrainingSamples, numTestSamples;

int myRank;// Rank of process

// Reads training or test data from a text file
int ReadFile(char *file_name, int valuesPerLine, int numLines, int* arr){
	FILE *ifp;
	int i, j, val;
	char *mode = "r";
	ifp = fopen(file_name, mode);

	if (ifp == NULL) {
		return 1;
	}

	i = 0;
	while((!feof(ifp)) && (i < (valuesPerLine*numLines)))
	{
		fscanf(ifp, "%d ", &val);

		arr[i] = val;

		i++;
	}

	// closing file
	fclose(ifp);

	return 0;
}

int getIndex3d (int i, int j, int k, int dimB, int dimC)
{
	return (i*dimB*dimC + j*dimC + k);
}

int getIndex2d (int i, int j, int dimB)
{
	return (i*dimB + j);
}

void InitNeuralNet(void){
	int  index, size;
	time_t t;

	/* seed rand with time */
	srand((unsigned) time(&t));
	
	size = numInputNodes * neuronsPerLayer;

	/* allocate and initialize inputWeights*/
	for(index = 0; index < size; index++)
	{
		inputWeights = (float *) malloc(numInputNodes * neuronsPerLayer * sizeof(float));

	}

	size = (numHiddenLayers - 1) * neuronsPerLayer * neuronsPerLayer;

	/* allocate and initialize hiddenWeights, hiddenNodeOutput,
     hiddenNodeBias */
	hiddenWeights = (float *) malloc((numHiddenLayers - 1) * neuronsPerLayer * neuronsPerLayer * sizeof(float));
	hiddenNodeOutput = (float *) malloc(numHiddenLayers * neuronsPerLayer * sizeof(float));
	hiddenNodeBias = (float *) malloc(numHiddenLayers * neuronsPerLayer * sizeof(float));


	for(index = 0; index < size; index++)
	{
		if(index < numHiddenLayers * neuronsPerLayer)

		{
			hiddenNodeOutput[index] = ((-2)*((float)rand()/RAND_MAX)) + 1;

			hiddenNodeBias[index] = ((-2)*((float)rand()/RAND_MAX)) + 1;

			hiddenWeights[index] = ((-2)*((float)rand()/RAND_MAX)) + 1;
		}
		else
		{
			hiddenWeights[index] = ((-2)*((float)rand()/RAND_MAX)) + 1;
		}


	}

	size = numOutputNodes * neuronsPerLayer;

	/* allocate and initialize outputWeights, outputOutputs,
     outputNodeBias */
	outputWeights = (float *) malloc(numOutputNodes * neuronsPerLayer * sizeof(float));
	outputOutputs = (float *) malloc(numOutputNodes * sizeof(float));
	outputBias = (float *) malloc(numOutputNodes * sizeof(float));

	for(index = 0; index < size; index++)
	{
		if(index < numOutputNodes)
		{
			outputOutputs[index] = ((-2)*((float)rand()/RAND_MAX)) + 1;

			outputBias[index] = ((-2)*((float)rand()/RAND_MAX)) + 1;

			outputWeights[index] = ((-2)*((float)rand()/RAND_MAX)) + 1;
		}
		else
		{
			outputWeights[index] = ((-2)*((float)rand()/RAND_MAX)) + 1;
		}


	}


	return;
}

void InitSamples(numTraining, numTest)
{
	trainingSamples = (int *) calloc(numTraining * numInputNodes, sizeof(int));
	trainingTargets = (int *) calloc(numTraining * numOutputNodes, sizeof(int));
	testSamples = (int *) calloc(numTest * numInputNodes, sizeof(int));

	return;
}


/* sigmoid function */
float Sigmoid(float num)
{
	return (float)(1/(1+exp(-num)));
}

/* propagate the input through the neural network */
void ForwardPropagation(int* input, float* output)
{
	int layer, node, inNode;
	float tempO;

	int threadCnt = 1;
	int myStartNode, myEndNode;
	int threadRank = 0;

#ifdef _OPENMP
	threadCnt = atoi((getenv("OMP_NUM_THREADS")));
#endif

#pragma omp parallel num_threads(threadCnt)\
		default(none), private(threadRank, myStartNode, myEndNode, layer, node, inNode, tempO)\
		shared(threadCnt, numHiddenLayers,  hiddenWeights, hiddenNodeOutput, hiddenNodeBias,\
			   input, neuronsPerLayer, numOutputNodes, outputWeights, output, outputBias,\
			   inputWeights, numInputNodes)
	{
#ifdef _OPENMP
		threadRank = omp_get_thread_num();
#endif

		/* calculate start and end for nodes */
		if(threadRank < neuronsPerLayer%threadCnt)
		{
			myStartNode  = threadRank * (neuronsPerLayer/threadCnt + 1);
			myEndNode = (threadRank + 1) * (neuronsPerLayer/threadCnt + 1);
		}
		else
		{
			myStartNode = (threadRank * (neuronsPerLayer/threadCnt)) + neuronsPerLayer%threadCnt;
			myEndNode = ((threadRank + 1) * (neuronsPerLayer/threadCnt)) + neuronsPerLayer%threadCnt;
		}

		/* propagate through first hidden layer */
		for(node = myStartNode; node < myEndNode; node++)
		{
			tempO = 0;

			for(inNode = 0; inNode < numInputNodes; inNode++)
			{
				tempO += input[inNode]*inputWeights[getIndex2d(node, inNode, numInputNodes)];
			}

			hiddenNodeOutput[getIndex2d(0,node, neuronsPerLayer)] = Sigmoid(tempO +
					hiddenNodeBias[getIndex2d(0, node, neuronsPerLayer)]);

		}

		/* propagate through each next hidden layer */
		for(layer = 1; layer < numHiddenLayers; layer++)
		{
			/* sync threads each time for this loop as it cannot be done
			 * in parallel */
#pragma omp barrier
			for(node = myStartNode; node < myEndNode; node++)
			{
				tempO = 0;

				for(inNode = 0; inNode < neuronsPerLayer; inNode++)
				{
					tempO += hiddenNodeOutput[getIndex2d(layer-1, inNode, neuronsPerLayer)]*
							hiddenWeights[getIndex3d(layer-1, node, inNode, neuronsPerLayer, neuronsPerLayer)];
				}


				hiddenNodeOutput[getIndex2d(layer, node, neuronsPerLayer)] = Sigmoid(tempO +
						hiddenNodeBias[getIndex2d(layer, node, neuronsPerLayer)]);

			}

		}

		layer = numHiddenLayers-1;

		/* sync threads as they all must be done with calc last layer
		 * outputs */
#pragma omp barrier

		/* calculate start and end for nodes */
		if(threadRank < numOutputNodes%threadCnt)
		{
			myStartNode  = threadRank * (numOutputNodes/threadCnt + 1);
			myEndNode = (threadRank + 1) * (numOutputNodes/threadCnt + 1);
		}
		else
		{
			myStartNode = (threadRank * (numOutputNodes/threadCnt)) + numOutputNodes%threadCnt;
			myEndNode = ((threadRank + 1) * (numOutputNodes/threadCnt)) + numOutputNodes%threadCnt;
		}

		/* propagate through output weights */
		for(node = myStartNode; node < myEndNode; node++)
		{
			tempO = 0;

			for(inNode = 0; inNode < neuronsPerLayer; inNode++)
			{

				tempO += hiddenNodeOutput[getIndex2d(layer, inNode, neuronsPerLayer)] *
						outputWeights[getIndex2d(node, inNode, neuronsPerLayer)];
			}

			output[node] = Sigmoid(tempO + outputBias[node]);

		}

	}
	return;
}

/* dout/dnet = out(1-out) */
/* dError/dout = -(target - out) = d MSE = 2 *0.5(target - out)^2 = */
/* dnet/ dw = out*/
void CalcDelta (int *targetOutput,
			float *hiddenDelta,
			float *outputDelta)
{
	int layer, node, inNode;

	int threadCnt = 1;
	int myStartNode, myEndNode;
	int threadRank = 0;

#ifdef _OPENMP
	threadCnt = atoi((getenv("OMP_NUM_THREADS")));
#endif

#pragma omp parallel num_threads(threadCnt)\
		default(none), private(threadRank, myStartNode, myEndNode, layer, node, inNode)\
		shared(threadCnt, numHiddenLayers,  hiddenWeights, hiddenNodeOutput, hiddenNodeBias,\
				targetOutput, neuronsPerLayer, numOutputNodes, outputWeights, outputBias,\
				hiddenDelta, outputDelta, outputOutputs)
	{
#ifdef _OPENMP
		threadRank = omp_get_thread_num();
#endif

		/* calculate start and end for nodes */
		if(threadRank < numOutputNodes%threadCnt)
		{
			myStartNode  = threadRank * (numOutputNodes/threadCnt + 1);
			myEndNode = (threadRank + 1) * (numOutputNodes/threadCnt + 1);
		}
		else
		{
			myStartNode = (threadRank * (numOutputNodes/threadCnt)) + numOutputNodes%threadCnt;
			myEndNode = ((threadRank + 1) * (numOutputNodes/threadCnt)) + numOutputNodes%threadCnt;
		}

		/* calculate output deltas */
		for(node = myStartNode; node < myEndNode; node++)
		{
			/* delta = dError/douti * douti/dneti */
			outputDelta[node] =  outputOutputs[node] * (1 - outputOutputs[node]) *
					(targetOutput[node] - outputOutputs[node]);

		}

		/* calculate start and end for nodes */
		if(threadRank < neuronsPerLayer%threadCnt)
		{
			myStartNode  = threadRank * (neuronsPerLayer/threadCnt + 1);
			myEndNode = (threadRank + 1) * (neuronsPerLayer/threadCnt + 1);
		}
		else
		{
			myStartNode = (threadRank * (neuronsPerLayer/threadCnt)) + neuronsPerLayer%threadCnt;
			myEndNode = ((threadRank + 1) * (neuronsPerLayer/threadCnt)) + neuronsPerLayer%threadCnt;
		}

		/* calculate hidden node deltas */
		for(layer = numHiddenLayers - 1; layer >= 0 ; layer--)
		{
/* threads need to sync for each layer */
#pragma omp barrier

			if (layer == numHiddenLayers - 1)
			{
				/* last layer in the neural net */
				/* for each node */
				for(node = myStartNode; node < myEndNode; node++)
				{

					/* for each output */
					for(inNode = 0; inNode < numOutputNodes; inNode++)
					{

						hiddenDelta[getIndex2d(layer, node, neuronsPerLayer)] +=
								outputDelta[inNode] * outputWeights[getIndex2d(inNode, node, neuronsPerLayer)];
					}

					hiddenDelta[getIndex2d(layer, node, neuronsPerLayer)] *=
							hiddenNodeOutput[getIndex2d(layer, node, neuronsPerLayer)] *
							(1 - hiddenNodeOutput[getIndex2d(layer, node, neuronsPerLayer)]);

				}
			}
			else
			{
				/* hidden layer propagation */
				/* for each node */
				for(node = myStartNode; node < myEndNode; node++)
				{

					/* for each output */
					for(inNode = 0; inNode < neuronsPerLayer; inNode++)
					{											/* i k j*/
						hiddenDelta[getIndex2d(layer, node, neuronsPerLayer)] +=
								hiddenDelta[getIndex2d(layer+1, inNode, neuronsPerLayer)] *
								hiddenWeights[getIndex3d(layer, inNode, node, neuronsPerLayer, neuronsPerLayer)];
					}

					hiddenDelta[getIndex2d(layer, node, neuronsPerLayer)] *=
							hiddenNodeOutput[getIndex2d(layer, node, neuronsPerLayer)] *
							(1 - hiddenNodeOutput[getIndex2d(layer, node, neuronsPerLayer)]);

				}

			}

		}

	}
	return;
}

void UpdateNetwork (float *hiddenDelta,
					float *outputDelta,
					int *input)
{
	int layer, node, inNode;

	int threadCnt = 1;
	int myStartNode, myEndNode;
	int threadRank = 0;

#ifdef _OPENMP
	threadCnt = atoi((getenv("OMP_NUM_THREADS")));
#endif

#pragma omp parallel num_threads(threadCnt)\
		default(none), private(threadRank, myStartNode, myEndNode, layer, node, inNode)\
		shared(threadCnt, numHiddenLayers,  hiddenWeights, hiddenNodeOutput, hiddenNodeBias,\
				input, learningRate, neuronsPerLayer, numOutputNodes, outputWeights, outputBias,\
				hiddenDelta, outputDelta, outputOutputs, inputWeights, numInputNodes)
	{
#ifdef _OPENMP
		threadRank = omp_get_thread_num();
#endif

		/* calculate start and end for nodes */
		if(threadRank < neuronsPerLayer%threadCnt)
		{
			myStartNode  = threadRank * (neuronsPerLayer/threadCnt + 1);
			myEndNode = (threadRank + 1) * (neuronsPerLayer/threadCnt + 1);
		}
		else
		{
			myStartNode = (threadRank * (neuronsPerLayer/threadCnt)) + neuronsPerLayer%threadCnt;
			myEndNode = ((threadRank + 1) * (neuronsPerLayer/threadCnt)) + neuronsPerLayer%threadCnt;
		}

		for(layer = 0; layer < numHiddenLayers; layer++)
		{
/* threads need to sync for each layer */
#pragma omp barrier

			for(node = myStartNode; node < myEndNode; node++)
			{
				/* update first layer */
				if(layer == 0)
				{
					for(inNode = 0; inNode < neuronsPerLayer; inNode++)
					{
						inputWeights[getIndex2d(node, inNode, numInputNodes)] +=
													learningRate * hiddenDelta[getIndex2d(layer, node, neuronsPerLayer)] * input[inNode];

						/*					hiddenWeights[i][j][k] -= learningRate * hiddenDelta[i][k] * input[k]; */

					}

					/* update biases */
					hiddenNodeBias[getIndex2d(layer, node, neuronsPerLayer)] +=
							learningRate * hiddenDelta[getIndex2d(layer, node, neuronsPerLayer)];
				}
				else
				{
					for(inNode = 0; inNode < neuronsPerLayer; inNode++)
					{
						hiddenWeights[getIndex3d(layer-1, node, inNode, neuronsPerLayer, neuronsPerLayer)] +=
								learningRate * hiddenDelta[getIndex2d(layer, node, neuronsPerLayer)] *
								hiddenNodeOutput[getIndex2d(layer-1, inNode, neuronsPerLayer)];
						/* hiddenWeights[i][j][k] -= learningRate * hiddenDelta[i][k] * hiddenNodeOutput[i-1][k]; */

					}
				}

			}
		}


		layer = numHiddenLayers - 1;

		/* calculate start and end for nodes */
		if(threadRank < numOutputNodes%threadCnt)
		{
			myStartNode  = threadRank * (numOutputNodes/threadCnt + 1);
			myEndNode = (threadRank + 1) * (numOutputNodes/threadCnt + 1);
		}
		else
		{
			myStartNode = (threadRank * (numOutputNodes/threadCnt)) + numOutputNodes%threadCnt;
			myEndNode = ((threadRank + 1) * (numOutputNodes/threadCnt)) + numOutputNodes%threadCnt;
		}

/* threads need to finish updating last layer before calc output updates */
#pragma omp barrier

		/* output layer */
		for(node = myStartNode; node < myEndNode; node++)
		{

			for(inNode = 0; inNode < neuronsPerLayer; inNode++)
			{
				outputWeights[getIndex2d(node, inNode, neuronsPerLayer)] += learningRate * outputDelta[node] *
						hiddenNodeOutput[getIndex2d(layer, inNode, neuronsPerLayer)];

				/* outputWeights[i][j] += learningRate * outputDelta[j] * hiddenNodeOutput[k][j]; */

			}

			outputBias[node] += learningRate * outputDelta[node];

		}
	}
	return;
}

/* backpropagtation algorithm */
void Backpropagation (int* input, int* targetOutput)
{
	int i = 0;
	float *hiddenDelta, *outputDelta;

	/* allocate output, hiddendelta and outputdelta */
	hiddenDelta = (float *)calloc(numHiddenLayers * neuronsPerLayer, sizeof(float));
	outputDelta = (float *)calloc(numOutputNodes, sizeof(float));

	ForwardPropagation (input, outputOutputs);

	CalcDelta(targetOutput,
			hiddenDelta,
			outputDelta);

	UpdateNetwork(hiddenDelta, outputDelta, input);


	free(hiddenDelta);
	free(outputDelta);

	return;
}

void SyncLearning(void)
{
	int  i, p, size;

	MPI_Comm_size(MPI_COMM_WORLD, &p);

	/* send weight from everyone to everyone else */
	MPI_Barrier(MPI_COMM_WORLD);


	/* send weights and biases to everyone */
	/* send/sum inputWeights to all p */
	MPI_Allreduce(MPI_IN_PLACE,
				&inputWeights[0],
				(neuronsPerLayer * numInputNodes),
				MPI_FLOAT,
				MPI_SUM,
				MPI_COMM_WORLD);

	/* send/sum hiddenWeights to all p */
	MPI_Allreduce(MPI_IN_PLACE,
			&hiddenWeights[0],
			((numHiddenLayers-1) * neuronsPerLayer * neuronsPerLayer),
			MPI_FLOAT,
			MPI_SUM,
			MPI_COMM_WORLD);

	/* send/sum hiddenNodeBias to all p */
	MPI_Allreduce(MPI_IN_PLACE,
			&hiddenNodeBias[0],
			(numHiddenLayers * neuronsPerLayer),
			MPI_FLOAT,
			MPI_SUM,
			MPI_COMM_WORLD);

	/* send/sum outputWeights to all p */
	MPI_Allreduce(MPI_IN_PLACE,
			&outputWeights[0],
			(numOutputNodes * neuronsPerLayer),
			MPI_FLOAT,
			MPI_SUM,
			MPI_COMM_WORLD);

	/* send/sum outputBias to all p */
	MPI_Allreduce(MPI_IN_PLACE,
			&outputBias[0],
			(numOutputNodes),
			MPI_FLOAT,
			MPI_SUM,
			MPI_COMM_WORLD);

	/* average all weights and biases */
	/* instead of doing individual loops for each assume hiddenNodeWeights
	 * is the largest length, and check bounds */
	for(i  = 0; i < ((numHiddenLayers-1) * neuronsPerLayer * neuronsPerLayer); i++)
	{
		if(i < (neuronsPerLayer * numInputNodes))
		{
			inputWeights[i] /= p;
		}

		if(i < (numOutputNodes))
		{
			outputBias[i] /= p;
		}

		if(i < (numOutputNodes * neuronsPerLayer))
		{
			outputWeights[i] /= p;
		}

		if(i < (numHiddenLayers * neuronsPerLayer))
		{
			hiddenNodeBias[i] /= p;
		}

		hiddenWeights[i] /= p;
	}


	return;
}

/* train the neural network using backpropagation */
void Train(void)
{
	int  i, j, count;
	float *output, val = 0, error = 0, MSE = 0;

	/* allocate output */
	output = (float *)calloc(numOutputNodes, sizeof(float));

	/* randomly train the input patterns until MSE < 0.01 */
	count = 0;
	i = rand()%numTrainingSamples;

	while(count < 5000)
	{
		Backpropagation(&trainingSamples[getIndex2d(i, 0 , neuronsPerLayer)],
				        &trainingTargets[getIndex2d(i, 0 , numOutputNodes)]);

		ForwardPropagation(&trainingSamples[getIndex2d(i, 0 , neuronsPerLayer)], output);

		/* MSE */
		error = 0;

		for(j = 0; j < numOutputNodes; j++)
		{
			error += (trainingTargets[getIndex2d(i, j , neuronsPerLayer)] - output[j]) *
					 (trainingTargets[getIndex2d(i, j , neuronsPerLayer)] - output[j]);
		}


		error = error / numOutputNodes;
	/*	printf("error %f\n", error);*/
		if(error <= 0.01)
		{
			if(count % 1000 == 0)
			{
				SyncLearning();
			}
			count++;
		}

		/* update training sample used */
		i = rand()%numTrainingSamples;
	}

	free(output);
}


void printOutput(int *value, int length, int cols)
{	/* cols - number of columns to print */
	int i;

	for(i = 0; i < length; i++)
	{
		if((i%cols) == 0)
		{
			printf("\n");
		}
		if(value[i] == 1)
		{
			printf("X");
		}
		else
		{
			printf(" ");
		}
	}

	printf("\n\n");

}

void Test(void)
{
	int i, j, * intOutput;
	float *output;

	/* allocate output */
	output = (float *)calloc(numOutputNodes, sizeof(float));
	intOutput = (int *)calloc(numOutputNodes, sizeof(int));

	for(i = 0; i < numTestSamples; i++)
	{
		ForwardPropagation(&testSamples[i*neuronsPerLayer], output);

		/* convert float output to integer, threshold = 0.5 since using round */
		for(j = 0; j < numOutputNodes; j++)
		{
			if(roundf(output[j]) < 0.1)
			{
				intOutput[j] = 0;
			}
			else
			{
				intOutput[j] = 1;
			}
		}

		printOutput(intOutput, numOutputNodes, 5);
	}

	free(intOutput);
	free(output);

	return;
}

void SendInputs(int *input, int trainingInputsCnt, int sendCnt, int worldSize, int tag)
{
	int i, numToSend = 0, dest = 0;
	MPI_Request request;

	/* since p 0 has read in all training data skip those samples */
	if(0 < trainingInputsCnt%worldSize)
	{
		i = sendCnt * (trainingInputsCnt/worldSize + 1);
	}
	else
	{
		i = sendCnt * (trainingInputsCnt/worldSize);
	}

	dest = 1;

	/* send each sample to the respective processor */
	while(dest < worldSize)
	{
		if(dest < trainingInputsCnt%worldSize)
		{
			numToSend = trainingInputsCnt/worldSize + 1;
		}
		else
		{
			numToSend = trainingInputsCnt/worldSize;
		}


		/* send training sources */
		MPI_Isend(&trainingSamples[i],
				sendCnt*numToSend,
				MPI_INT,
				dest,
				tag,
				MPI_COMM_WORLD,
				&request);


		i += sendCnt*numToSend;
		dest++;

	}

	return;
}
void cleanUp(void)
{
	free(hiddenWeights);
	free(hiddenNodeBias);
	free(hiddenNodeOutput);

	free(outputWeights);
	free(outputBias);
	free(outputOutputs);

	free(trainingSamples);
	free(trainingTargets);
	free(testSamples);

	return;
}
// Main function
int main(int argc, char** argv){
	int k= 0;
	int i, numToSend, numToRecv, numTrainingInputs, dest = 0;
	int p;// Number of processes
	int tag = TAG;// Tag for message
	MPI_Status status;// Return status for receive
	MPI_Request request;
	char* trainingFile, * trainingTargetFile, * testingFile;
	double start, end;


	/* read num inputs/outputs nodes */
	numInputNodes = atoi(argv[1]);

	numOutputNodes = atoi(argv[2]);

	numTrainingInputs = atoi(argv[3]);

	numTestSamples = atoi(argv[4]);

	/* read the number of Hidden layers in net */
	numHiddenLayers = atoi(argv[5]);

	neuronsPerLayer = atoi(argv[6]);

	/* read learning rate */
	learningRate = atof(argv[7]);

	/* read testing data file */
	testingFile = argv[8];

	/* read training data file */
	trainingFile = argv[9];

	/* read training target data  */
	trainingTargetFile = argv[10];


	// initializing MPI structures and checking p is odd
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	/* initialize the neural network */
	InitNeuralNet();
	InitSamples(numTrainingInputs, numTestSamples);


	/* set to number of samples to train */
	if(myRank < numTrainingInputs%p)
	{
		numTrainingSamples = numTrainingInputs/p + 1;
	}
	else
	{
		numTrainingSamples = numTrainingInputs/p;
	}

	if(myRank == 0)
	{
		ReadFile(trainingFile, numInputNodes, numTrainingInputs, trainingSamples);
		ReadFile(trainingTargetFile, numOutputNodes, numTrainingInputs, trainingTargets);
		ReadFile(testingFile, numInputNodes, numTestSamples, testSamples);


		/* send training sample */
		SendInputs(trainingSamples, numTrainingInputs, numInputNodes, p, TRAINING_SAMPLES);

		/* send training target outputs */
		SendInputs(trainingTargets, numTrainingInputs, numOutputNodes, p, TRAINING_TARGETS);

	}
	else
	{
		/* receive training samples */
		MPI_Recv(&trainingSamples[0],
				numTrainingSamples * numInputNodes,
				MPI_INT,
				0,
				TRAINING_SAMPLES,
				MPI_COMM_WORLD,
				&status);

		/* receive training targets */
		MPI_Recv(&trainingTargets[0],
				numTrainingSamples * numOutputNodes,
				MPI_INT,
				0,
				TRAINING_TARGETS,
				MPI_COMM_WORLD,
				&status);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if(myRank == 0)
	{
		start = MPI_Wtime();
	}

	/* train the neural network */
	Train();
	/* done learning */

	if(myRank == 0)
	{
		end = MPI_Wtime();
		printf("Duration: %f seconds\n", (end-start));

		Test();
	}

	cleanUp();

	// finalizing MPI structures
	MPI_Finalize();
	return 0;
}

