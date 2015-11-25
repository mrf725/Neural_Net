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

/* weights between neurons for each layer
   # Hidden Layers, # Neurons, #Neurons inputs from 
   previous layer
 i = index for each layer
 j = index of each neuron for that layer
 k = index of weight from neuron k to i */
float*** hiddenWeights;

/* weight of the bias term per each hidden layer */
float**  hiddenNodeBias;

/* output at each hidden node
   # Hidden Layer, # Neurons*/
float** hiddenNodeOutput;


/* weight between last hidden layer and output
   #Neuron, #outputs */
/* i - ith output node
   j - jth input from last hidden layer */
float**  outputWeights;
float*   outputBias;

/* the output value at each output neuron */
float* outputOutputs;
int numOutputNodes;
int neuronsPerLayer;
int numHiddenLayers;
float learningRate;


int** trainingSamples;
int** trainingTargets;
int** testSamples;
int numTrainingSamples, numTestSamples;


// Reads training or test data from a text file
int ReadFile(char *file_name, int valuesPerLine, int numLines, int** arr){
	FILE *ifp;
	int i, j, val;
	char *mode = "r";
	ifp = fopen(file_name, mode);

	if (ifp == NULL) {
		return 1;
	}

	// reading values
	for(i = 0; i < numLines; i++)
	{
		for(j = 0; j < valuesPerLine; j++)
		{
			if(!feof(ifp))
			{
				fscanf(ifp, "%d", &val);

				arr[i][j] = val;
			}
		}
	}

	// closing file
	fclose(ifp);

	return 0;
}


void InitNeuralNet(void){
	int layer, node, inputNode;
	time_t t;

	/* seed rand with time */
	srand((unsigned) time(&t));

	/* allocate and initialize hiddenWeights, hiddenNodeOutput,
     hiddenNodeBias */
	hiddenWeights = (float ***) calloc(numHiddenLayers, sizeof(float**));
	hiddenNodeOutput = (float **) calloc(numHiddenLayers, sizeof(float*));
	hiddenNodeBias = (float **) calloc(numHiddenLayers, sizeof(float*));

	for(layer = 0; layer < numHiddenLayers; layer++)
	{
		hiddenWeights[layer] = (float**) calloc(neuronsPerLayer, sizeof(float*));
		hiddenNodeOutput[layer] = (float *) calloc(neuronsPerLayer, sizeof(float));
		hiddenNodeBias[layer] = (float *) calloc(neuronsPerLayer, sizeof(float));

		for(node = 0; node < neuronsPerLayer; node++)
		{
			hiddenWeights[layer][node] = (float*) calloc(neuronsPerLayer, sizeof(float));

			hiddenNodeOutput[layer][node] = ((-2)*((float)rand()/RAND_MAX)) + 1;

			hiddenNodeBias[layer][node] = ((-2)*((float)rand()/RAND_MAX)) + 1;


			for(inputNode = 0; inputNode < neuronsPerLayer; inputNode++)
			{
				/* initialize weight between [-1, 1] */
				hiddenWeights[layer][node][inputNode] = ((-2)*((float)rand()/RAND_MAX)) + 1;

			}


		}
	}

	/* allocate and initialize outputWeights, outputOutputs,
     outputNodeBias */
	outputWeights = (float **) calloc(numOutputNodes, sizeof(float *));
	outputOutputs = (float *) calloc(numOutputNodes, sizeof(float));
	outputBias = (float *) calloc(numOutputNodes, sizeof(float));

	for(node = 0; node < numOutputNodes; node++)
	{
		outputWeights[node] = (float *) calloc(neuronsPerLayer, sizeof(float));

		outputOutputs[node] = ((-2)*((float)rand()/RAND_MAX)) + 1;

		outputBias[node] = ((-2)*((float)rand()/RAND_MAX)) + 1;

		for(inputNode = 0; inputNode < neuronsPerLayer; inputNode++)
		{
			outputWeights[node][inputNode] = ((-2)*((float)rand()/RAND_MAX)) + 1;

		}
	}


	return;
}

void InitSamples(numTraining, numTest)
{
	int i, j;

	trainingSamples = (int **) calloc(numTraining, sizeof(int*));
	trainingTargets = (int **) calloc(numTraining, sizeof(int*));
	testSamples = (int **) calloc(numTest, sizeof(int*));

	for(i = 0; i < numTraining; i++)
	{
		trainingSamples[i] = (int *) calloc(neuronsPerLayer, sizeof(int));
		trainingTargets[i] = (int *) calloc(numOutputNodes, sizeof(int));
	}

	for(i = 0; i < numTest; i++)
	{
		testSamples[i] = (int *) calloc(neuronsPerLayer, sizeof(int));
	}


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

		for(inNode = 0; inNode < neuronsPerLayer; inNode++)
		{
			tempO += input[inNode]*hiddenWeights[0][node][inNode];
		}

		hiddenNodeOutput[0][node] = Sigmoid(tempO + hiddenNodeBias[0][node]);

	}

	/* propagate through each next hidden layer */
	for(layer = 1; layer < numHiddenLayers; layer++)
	{
		for(node = 0; node < neuronsPerLayer; node++)
		{
			tempO = 0;

			for(inNode = 0; inNode < neuronsPerLayer; inNode++)
			{
				tempO += hiddenNodeOutput[layer-1][inNode]*hiddenWeights[layer][node][inNode];
			}

			hiddenNodeOutput[layer][node] = Sigmoid(tempO + hiddenNodeBias[layer][node]);

		}

	}

	layer = numHiddenLayers-1;
	/* propagate through output weights */
	for(node = 0; node < numOutputNodes; node++)
	{
		tempO = 0;

		for(inNode = 0; inNode < neuronsPerLayer; inNode++)
		{
			tempO += hiddenNodeOutput[layer][inNode]*outputWeights[node][inNode];
		}

		output[node] = Sigmoid(tempO + outputBias[node]);

	}

	return;
}

/* dout/dnet = out(1-out) */
/* dError/dout = -(target - out) = d MSE = 2 *0.5(target - out)^2 = */
/* dnet/ dw = out*/
void CalcDelta (int *targetOutput, 
			float **hiddenDelta,
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
					hiddenDelta[layer][node] += outputDelta[inNode] * outputWeights[inNode][node];
				}

				hiddenDelta[layer][node] *= hiddenNodeOutput[layer][node] * (1 - hiddenNodeOutput[layer][node]);

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
					hiddenDelta[layer][node] += hiddenDelta[layer+1][inNode] * hiddenWeights[layer][inNode][node];
				}

				hiddenDelta[layer][node] *= hiddenNodeOutput[layer][node] * (1 - hiddenNodeOutput[layer][node]);

			}

		}

	}


	return;
}

void UpdateNetwork (float **hiddenDelta, 
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
				for(inNode = 0; inNode < neuronsPerLayer; inNode++)
				{
					hiddenWeights[layer][node][inNode] += learningRate * hiddenDelta[layer][node] * input[inNode];

/*					hiddenWeights[i][j][k] -= learningRate * hiddenDelta[i][k] * input[k]; */

				}

				/* update biases */
				hiddenNodeBias[layer][node] += learningRate * hiddenDelta[layer][node];
			}
			else
			{
				for(inNode = 0; inNode < neuronsPerLayer; inNode++)
				{
					 hiddenWeights[layer][node][inNode] += learningRate * hiddenDelta[layer][node] * hiddenNodeOutput[layer-1][inNode];
					/* hiddenWeights[i][j][k] -= learningRate * hiddenDelta[i][k] * hiddenNodeOutput[i-1][k]; */

				}
			}

		}
	}


	layer = numHiddenLayers - 1;

	/* output layer */
	for(node = 0; node < numOutputNodes; node++)
	{

		for(inNode = 0; inNode < neuronsPerLayer; inNode++)
		{
			outputWeights[node][inNode] += learningRate * outputDelta[node] * hiddenNodeOutput[layer][inNode];

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
	float **hiddenDelta, *outputDelta;

	/* allocate output, hiddendelta and outputdelta */
	hiddenDelta = (float **)calloc(numHiddenLayers, sizeof(float *));
	outputDelta = (float *)calloc(numOutputNodes, sizeof(float));

	for(i = 0; i < numHiddenLayers; i++)
	{
		hiddenDelta[i] = (float *)calloc(neuronsPerLayer,  sizeof(float));
	}

	ForwardPropagation (input, outputOutputs);

	CalcDelta(targetOutput,
			hiddenDelta,
			outputDelta);

	UpdateNetwork(hiddenDelta, outputDelta, input);


	for(i = 0; i < numHiddenLayers; i++)
	{
		free(hiddenDelta[i]);
	}

	free(hiddenDelta);
	free(outputDelta);

	return;
}


/* train the neural network using backpropagation */
void Train(void)
{
	int  i, j, count;
	float *output,  error = 0, MSE = 0;

	/* allocate output */
	output = (float *)calloc(numOutputNodes, sizeof(float));


	/* randomly train the input patterns until MSE < 0.01 */
	count = 0;
	i = rand()%numTrainingSamples;
	while(count < 5000)
	{

		Backpropagation(trainingSamples[i], trainingTargets[i]);

		ForwardPropagation(trainingSamples[i], output);

		/* MSE */
		error = 0;
		for(j = 0; j < numOutputNodes; j++)
		{
			error += (trainingTargets[i][j] - output[j])*(trainingTargets[i][j] - output[j]);
		}
		error = error / numOutputNodes;


		if(error <= 0.001)
		{
			count++;
		}

		/* update training sample used */
		i = rand()%numTrainingSamples;
	}
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
	output = (float *)calloc(neuronsPerLayer, sizeof(float));
	intOutput = (int *)calloc(neuronsPerLayer, sizeof(int));

	for(i = 0; i < numTestSamples; i++)
	{
		ForwardPropagation(testSamples[i], output);

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

	return;
}

/* Main function */
int main(int argc, char** argv){
	int** trainingData,**  targetValues;
	char* trainingFile, * trainingTargetFile, * testingFile;

	/* read num inputs/outputs nodes */
	neuronsPerLayer = atoi(argv[1]);

	numOutputNodes = atoi(argv[2]);

	numTrainingSamples = atoi(argv[3]);

	numTestSamples = atoi(argv[4]);

	/* read the number of Hidden layers in net */
	numHiddenLayers = atoi(argv[5]);

	/* read learning rate */
	learningRate = atof(argv[6]);

	/* read testing data file */
	testingFile = argv[7];

	/* read training data file */
	trainingFile = argv[8];

	/* read training target data  */
	trainingTargetFile = argv[9];

	/* initialize the neural network */
	InitNeuralNet();
	InitSamples(numTrainingSamples, numTestSamples);

	ReadFile(trainingFile, neuronsPerLayer, numTrainingSamples, trainingSamples);
	ReadFile(trainingTargetFile, numOutputNodes, numTrainingSamples, trainingTargets);
	ReadFile(testingFile, neuronsPerLayer, numTestSamples, testSamples);

	/* train the neural network */
	Train();

	Test();

	return 0;
}

