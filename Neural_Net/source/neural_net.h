/*
 * neural_net.h
 *
 *  Created on: Nov 29, 2015
 *      Author: Michael
 */

#ifndef SOURCE_NEURAL_NET_
#define SOURCE_NEURAL_NET_

typedef struct{
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


	/* the output value at each output neuron */
	float* outputOutputs;

}Neural_Net;


#endif /* SOURCE_NEURAL_NET_ */
