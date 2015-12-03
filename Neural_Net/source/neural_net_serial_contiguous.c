/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS2045: Introduction to HPC Systems
 * Student: Mike Fehr
 * Instructor Bryan Mills, University of Pittsburgh
 * Serial neural net code. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "timer.h"
#include <string.h>
#include "alphabet.h"

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
int numInputNodes;

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
int verbose;


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
		fscanf(ifp, "%d", &val);

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

	inputWeights = (float *) malloc(numInputNodes * neuronsPerLayer * sizeof(float));

	/* allocate and initialize inputWeights*/
	for(index = 0; index < size; index++)
	{
		inputWeights[index] = ((-2)*((float)rand()/RAND_MAX)) + 1;
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

	/* propagate through first hidden layer */
	for(node = 0; node < neuronsPerLayer; node++)
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
		for(node = 0; node < neuronsPerLayer; node++)
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

	/* propagate through output weights */
	for(node = 0; node < numOutputNodes; node++)
	{
		tempO = 0;

		for(inNode = 0; inNode < neuronsPerLayer; inNode++)
		{

			tempO += hiddenNodeOutput[getIndex2d(layer, inNode, neuronsPerLayer)] *
					 outputWeights[getIndex2d(node, inNode, neuronsPerLayer)];
		}

		output[node] = Sigmoid(tempO + outputBias[node]);

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

	/* calculate output deltas */
	for(node = 0; node < numOutputNodes; node++)
	{
		/* delta = dError/douti * douti/dneti */
		outputDelta[node] =  outputOutputs[node] * (1 - outputOutputs[node]) *
							(targetOutput[node] - outputOutputs[node]);

	}

	/* calculate hidden node deltas */
	for(layer = numHiddenLayers - 1; layer >= 0 ; layer--)
	{
		if (layer == numHiddenLayers - 1)
		{
			/* last layer in the neural net */
			/* for each node */
			for(node = 0; node < neuronsPerLayer; node++)
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
			for(node = 0; node < neuronsPerLayer; node++)
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


	return;
}

void UpdateNetwork (float *hiddenDelta,
					float *outputDelta,
					int *input)
{
	int layer, node, inNode;

	for(layer = 0; layer < numHiddenLayers; layer++)
	{
		for(node = 0; node < neuronsPerLayer; node++)
		{
			/* update first layer */
			if(layer == 0)
			{
				for(inNode = 0; inNode < numInputNodes; inNode++)
				{
					inputWeights[getIndex2d(node, inNode, numInputNodes)] +=
							learningRate * hiddenDelta[getIndex2d(layer, node, neuronsPerLayer)] * input[inNode];

/*					hiddenWeights[i][j][k] -= learningRate * hiddenDelta[i][k] * input[k]; */

				}

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

			/* update biases */
							hiddenNodeBias[getIndex2d(layer, node, neuronsPerLayer)] +=
									learningRate * hiddenDelta[getIndex2d(layer, node, neuronsPerLayer)];

		}
	}


	layer = numHiddenLayers - 1;

	/* output layer */
	for(node = 0; node < numOutputNodes; node++)
	{

		for(inNode = 0; inNode < neuronsPerLayer; inNode++)
		{
			outputWeights[getIndex2d(node, inNode, neuronsPerLayer)] += learningRate * outputDelta[node] *
																	   hiddenNodeOutput[getIndex2d(layer, inNode, neuronsPerLayer)];

			/* outputWeights[i][j] += learningRate * outputDelta[j] * hiddenNodeOutput[k][j]; */

		}

		outputBias[node] += learningRate * outputDelta[node];

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


/* train the neural network using backpropagation */
void Train(void)
{
	int  i, j, count, syncRate = 100;;
	float *output,  error = 0, MSE = 0;
	float targetMSE = 0.01;
	int timeout = 5000;

	if(getenv("TARGET_MSE") != 0)
	{
		targetMSE = atof(getenv("TARGET_MSE"));
	}

	if(getenv("SYNC_RATE") != 0)
	{
		syncRate = atoi(getenv("SYNC_RATE"));
	}

	/* allocate output */
	output = (float *)calloc(numOutputNodes, sizeof(float));


	/* randomly train the input patterns until MSE < 0.01 */
	count = 0;
	i = rand()%numTrainingSamples;
	while(count < 5000)
	{

		Backpropagation(&trainingSamples[getIndex2d(i, 0 , numInputNodes)],
				        &trainingTargets[getIndex2d(i, 0 , numOutputNodes)]);

		ForwardPropagation(&trainingSamples[getIndex2d(i, 0 , numInputNodes)], output);

		/* MSE */
		error = 0;
		for(j = 0; j < numOutputNodes; j++)
		{
			error += (trainingTargets[getIndex2d(i, j , numOutputNodes)] - output[j]) *
					 (trainingTargets[getIndex2d(i, j , numOutputNodes)] - output[j]);
		}
		error = error / numOutputNodes;


		if(error <= targetMSE)
		{
			timeout = 5000;

			if(count % 200 == 0)
			{
				learningRate /= 1.5;
			}
			count++;
		}

		timeout--;

		if(timeout == 0)
		{
			break;
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
void calcProbabilities(float * probs, int  * out)
{
	int i = 0, j = 0, numCorrect = 0;

	for(i = 0; i < NUM_CHARS; i++)
	{
		numCorrect = 0;
		for(j = 0; j < numOutputNodes; j++)
		{
			if(out[j] == ALPHABET_MAPPING[i][j])
			{
				numCorrect++;
			}
		}
		probs[i] = (float)numCorrect/numOutputNodes;
	}

	return;
}
char predict(float * in)
{
	int i = 0, maxIdx = 0;
	float max = 0;

	for(i = 0; i < NUM_CHARS; i++ )
	{
		if(in[i] > max)
		{
			max = in[i];
			maxIdx = i;
		}
	}

	return (maxIdx);
}
void Test(void)
{
	int i, j, * intOutput;
	float *output;
	float probabilities[NUM_CHARS];
	float accuracy = 0;
	char pred = 0;

	/* allocate output */
	output = (float *)calloc(neuronsPerLayer, sizeof(float));
	intOutput = (int *)calloc(neuronsPerLayer, sizeof(int));

	for(i = 0; i < numTestSamples; i++)
	{

		ForwardPropagation(&testSamples[i*numInputNodes], output);

		/* convert float output to integer, threshold = 0.5 since using round */
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


		calcProbabilities(probabilities, intOutput);
		pred = predict(probabilities);

		printf("Predicted Character: %c\n", pred+65);

		/* assumes inputs are in abc order for easy calculation */
		if(pred == i)
		{
			accuracy++;
		}

		if(verbose)
		{
			printOutput(intOutput, numOutputNodes, 5);
		}
	}

	accuracy /= numTestSamples;
	printf("Prediction Accuracy: %f\n", accuracy);

	free(intOutput);
	free(output);

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

/* Main function */
int main(int argc, char** argv){
	char* trainingFile, * trainingTargetFile, * testingFile;
	double duration = 0;

	verbose = 0;

	if(getenv("VERBOSE") != 0)
	{
		verbose = atoi(getenv("VERBOSE"));
	}

	/* read num inputs/outputs nodes */
	numInputNodes = atoi(argv[1]);

	numOutputNodes = atoi(argv[2]);

	numTrainingSamples = atoi(argv[3]);

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

	/* initialize the neural network */
	InitNeuralNet();
	InitSamples(numTrainingSamples, numTestSamples);

	ReadFile(trainingFile, numInputNodes, numTrainingSamples, trainingSamples);
	ReadFile(trainingTargetFile, numOutputNodes, numTrainingSamples, trainingTargets);
	ReadFile(testingFile, numInputNodes, numTestSamples, testSamples);


	/* train the neural network */
	timerStart();

	Train();

	duration = timerStop();
	printf("Duration: %f seconds\n", (duration));

	Test();

	cleanUp();

	return 0;
}

