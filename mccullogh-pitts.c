#include <stdio.h>
#include <stdlib.h>

//#define TOT_INPUTS 8 //DEPRICATED
#define LAYERS 1 //depth of network
#define NUM_INPUTS 2 //number of inputs per neuron and width of network
#define TOT_NEU (NUM_INPUTS * LAYERS) //total number of neurons
#define TOT_IN (NUM_INPUTS * LAYERS) //total number of if inputs
#define TOT_WEIGHTS (NUM_INPUTS*TOT_NEU)
#define THRESHHOLD_HIGH -50
#define THRESHHOLD_LOW 0
//#define THRESHHOLD_MODE 0 //0 = below, 1 = above, 2 = between, 3 = beyond DEPRICATED
#define BELOW 0
#define ABOVE 1
#define BETWEEN 2
#define BEYOND 3
#define THRESHHOLD_BIAS 0 //DEPRICATED
#define WEIGHT_MAX 10 //DEPRICATED
#define WEIGHT_MIN -10 //DEPRICATED
#define WEIGHT_DEPTH_MAX 5 //how deeply to search weight space
#define WEIGHT_RANGE (WEIGHT_MAX - WEIGHT_MIN)
#define NUM_TESTCASES 4 //number of elements in KNOWN_INS and KNOWN_OUTS
#define KNOWN_INS {0,1,2,3} //must be the same length as KNOWN_OUTS
#define KNOWN_OUTS {3,2,1,0} //must be the same length as KNOWN_INS
#define TEST_WEIGHTS {0,1,2,3}
//#define INIT_WEIGHTS {1,2,3,4,5,6,7,8,9,10,11,12,13,14} //initialize weights vector depricated
//#define TRIES_LIMIT 0 unused
//#define WEIGHT_INIT_MODE 0 //unused
#define ZEROIZE 0
#define PARALLEL_LOAD 1
#define ADD 2 //depricated
#define GET_NEURONS(mDP) ((neuron*)mDP[0]) //get neuron array from master data ponter
#define GET_INPUTS(mDP) ((int*)mDP[1]) //get input vector from master data pointer
#define GET_WEIGHTS(mDP) ((int*)mDP[2]) //get weight vector from master data pointer
#define GET_OUTPUTS(mDP) (&((int*)mDP[1])[TOT_IN]) //get output vector from master data pointer
#define VEC_MULT(vec,len,scal) for(i=0;i<len;i++) vec[i]*=(scal)
#define VEC_ADD(vec,len,vec2) for(i=0;i<len;i++) vec[i]+=vec2[i]
#define VEC_ADD2(vOut,len,v1,v2) for(i=0;i<len;i++) vOut[i] = v1[i]+v2[i]
#define INT_TO_VEC(vec,num,len) for(i=0;i<len;i++) vec[i]=(num & 1<<i)
#define VEC_TO_INT(num,vec,len) for(i=0;i<len;i++) num+=(vec[i]<<i) //num MUST be initialized to zero
#define VEC_SUM(num,vec,len) for(i=0;i<len;i++) num+=(vec[i]) //num must be initialized to zero

/* run this program using the console pauser or add your own getch, system("pause") or input loop */

typedef struct
{
 int* inputs;
 int* weights;
 int* out;
} neuron;

//pipe up a single neuron
int neuronInit(neuron* this, int* input, int* weight, int* output)
{
 this->inputs = input;
 this->weights = weight;
 this->out = output;

 return 0;
}

//resolve neuron outputs
int neuronResolve(neuron* this, int threshholdMode)
{
 int sum = THRESHHOLD_BIAS;
 int i;
 
 for(i=0;i<NUM_INPUTS;i++)
  sum += this->inputs[i] * this->weights[i];

 if(threshholdMode == BELOW) //below
  *(this->out) = sum < THRESHHOLD_LOW ? 1 : 0;
 else if(threshholdMode == ABOVE) //above
  *(this->out) = sum > THRESHHOLD_HIGH ? 1 : 0;
 else if(threshholdMode == BETWEEN) //between
   *(this->out) = sum < THRESHHOLD_HIGH && sum > THRESHHOLD_LOW ? 1 : 0;
 else if(threshholdMode == BEYOND) //beyond
   *(this->out) = sum > THRESHHOLD_HIGH || sum < THRESHHOLD_LOW ? 1 : 0;
 else //invalid threshholdMode
  return 1;
 
 return 0;
}

//allocate input-output vectors, neurons, and pipe everything up
int initAllNeurons(void** mDP) //masterDataPointer [0] = nuerons, [1] = input vectors, 2 = weight vectors
{
 mDP[0] = malloc(TOT_NEU*sizeof(neuron)); //neurons
 mDP[1] = malloc((TOT_IN+NUM_INPUTS)*sizeof(int)); //inputs and outputs
 mDP[2] = malloc(TOT_WEIGHTS*sizeof(int)); //weights
 
 int i;
 neuron* currentNeuron;
 int* currentInput;
 int* currentWeight;
 int* currentOutput;
 
 for(i=0;i<TOT_NEU;i++)
 {
  currentNeuron = &((neuron*)mDP[0])[i];
  currentInput = &((int*)mDP[1])[(i/NUM_INPUTS)*NUM_INPUTS];
  currentWeight = &((int*)mDP[2])[i*NUM_INPUTS];
  currentOutput = &((int*)mDP[1])[(i/NUM_INPUTS)*NUM_INPUTS+NUM_INPUTS]; //whoopsiedoodle, this line may be bad :(
  neuronInit(currentNeuron, currentInput, currentWeight, currentOutput);
 }
return 0;
}

//print a vector
int printVec(int* vec, int len)
{
 printf("<");
 int i;
 for(i=0;i<len;i++)
  printf("%d,",vec[i]);
 printf(">\n");
}

//zeroize weight vector, or parallel load preset values
int setVec(int* vec, int len, int mode, int* load) //mode 0 = zeroize, mode 1 = parallel load, mode 2 = add
{
 int i;
 
 if(mode==ZEROIZE)
  for(i=0;i<len;i++)
   vec[i] = load[0];
 else if(mode==PARALLEL_LOAD)
  for(i=0;i<len;i++)
   vec[i] = load[i];
 else if(mode==ADD)
  for(i=0;i<len;i++)
   vec[i] += load[i];
 else //invalid mode
  return 1;

 return 0;
}

//hopefully this is self-explanitory
int resolveAllNeurons(neuron* neurons)
{

 int i;
 for(i = 0; i<TOT_NEU; i++)
  neuronResolve(&neurons[i], BELOW);
  
  return 0;
}

//test known inputs vs. known outputs
int testCase(int* gonogo, int input, int output, neuron* net, int* netIn, int* netOut)
{
 int toCompare = 0;

 int i;
 INT_TO_VEC(netIn, input, NUM_INPUTS);
 resolveAllNeurons(net);
 VEC_TO_INT(toCompare, netOut, NUM_INPUTS);
 *gonogo = (toCompare == output);
 
 return 0;
}

int testCases(int* result, int* ins, int* outs, int numOf, neuron* net, int* netIn, int* netOut)
{
 *result = 1;
 int results[numOf];

 int i;
 for(i=0; i<numOf; i++)
 {
  testCase(&results[i], ins[i], outs[i], net, netIn, netOut);
  printf("%d,", results[i]);
  *result = *result && results[i];
 }
 printf("\n");
 
 return 0;
}

int descFromWeights(int* desc, int* weights, int depth)
{
 int i,s;
 int k=0;
 for(i=0;i<TOT_WEIGHTS;i++)
 {
  for(s=0;s<weights[i];s++)
  {
   desc[k]=i;
   //printf("%d %d %d\n",weights[i],k,i);
   k++;
  }
 }

 return 0;
}

int weightsFromDesc(int* weights, int* desc, int depth)
{
 int zero = 0;
 setVec(weights,TOT_WEIGHTS,ZEROIZE,&zero);
 int i;
 for(i=0;i<depth;i++)
  weights[desc[i]]++;
}

//adjust weight vector
int nextWeight(int* weights, int* desc, int depth)
{ 
 //printVec(desc,depth);
 desc[0]++;
 //printVec(desc,depth);

 int i;
 for(i=0;i<depth-1;i++)
 {
  //printf("%d %d\n",desc[i],TOT_WEIGHTS-1);
  if(desc[i]>TOT_WEIGHTS-1)
  {
   desc[i]=0;
   desc[i+1]++;
  }
 }
 //printVec(desc,depth);
    
 if(desc[depth-1]>TOT_WEIGHTS-1)
  return 1; //Done at this depth
 else
  weightsFromDesc(weights, desc, depth);

 return 0;
}

int singleAdjustW(int* gonogo, void** mDP, int depth, int ptr)//still not quite right :(
{
 int i;
 int ins[NUM_TESTCASES] = KNOWN_INS;
 int outs[NUM_TESTCASES] = KNOWN_OUTS;
 int* inVec = GET_INPUTS(mDP);
 int* outVec = GET_INPUTS(mDP);
 int* weightVec = GET_WEIGHTS(mDP);
 neuron* neuVec = GET_NEURONS(mDP);
 
 if(depth <= WEIGHT_DEPTH_MAX)
 {
  weightVec[ptr]++;
  printVec(weightVec,TOT_WEIGHTS);
  testCases(gonogo, ins, outs, NUM_TESTCASES, neuVec, inVec, outVec);
  if(!*gonogo)
  {
   weightVec[ptr]*=-1;
   printVec(weightVec,TOT_WEIGHTS);
   testCases(gonogo, ins, outs, NUM_TESTCASES, neuVec, inVec, outVec);
   weightVec[ptr]*=-1;
   
   if(depth < WEIGHT_DEPTH_MAX)
    for(i=0;i<TOT_WEIGHTS && !*gonogo;i++)
     singleAdjustW(gonogo, mDP, depth+1, i);
   if(!*gonogo)
    GET_WEIGHTS(mDP)[ptr]--;
  }
 }

 return 0;
}

int testWeightsFromDesc()
{
 int desc[] = {3,3,2};
 int depth = 3;
 int weights[TOT_WEIGHTS] = {0,0,0,0};
 
 weightsFromDesc(weights,desc,depth);
 printVec(weights, TOT_WEIGHTS);
}

int testDescFromWeights()
{
 int weights[] = {0,1,2,0};
 int depth;
 
 int i;
 VEC_SUM(depth,weights,TOT_WEIGHTS);
 printf("%d\n",depth);

 int desc[depth];
 printVec(desc,depth);
 
 descFromWeights(desc, weights, depth);
 
 printVec(desc,depth);
 
 return 0;
}

int testWeightAdjustment()
{
 void* mDP[3]; //masterDataPointer [0] = nuerons, [1] = input output vectors, 2 = weight vectors
 initAllNeurons(mDP);
 
 int weightStart[TOT_WEIGHTS] = {0,1,2,0};
 
 setVec(GET_WEIGHTS(mDP), TOT_WEIGHTS, PARALLEL_LOAD, weightStart);
 
 printVec(GET_WEIGHTS(mDP),TOT_WEIGHTS);
 
 int i;
 int depth=3;
 //VEC_SUM(depth,weightStart,TOT_WEIGHTS);
 
 int desc[] = {1,2,2};
 
 for(i=0;i<30;i++)
 {
  nextWeight(GET_WEIGHTS(mDP),desc,depth);
  printVec(GET_WEIGHTS(mDP),TOT_WEIGHTS);
 }
 
 return 0;
}

int theActualProgram()
{
 void* mDP[3]; //masterDataPointer [0] = nuerons, [1] = input output vectors, 2 = weight vectors
 initAllNeurons(mDP);
 
 int weightStart = ZEROIZE;
 setVec((int*)mDP[2], TOT_WEIGHTS, ZEROIZE, &weightStart);
 
 int done;
 int i;
 int gonogo;
 int ins[NUM_TESTCASES] = KNOWN_INS;
 int outs[NUM_TESTCASES] = KNOWN_OUTS;
 
 testCases(&gonogo, ins, outs, NUM_TESTCASES, GET_NEURONS(mDP), GET_INPUTS(mDP), GET_OUTPUTS(mDP));
 for(i=0;i<TOT_WEIGHTS && !gonogo;i++)
  singleAdjustW(&gonogo, mDP, 1, i);
   
 
 if(!gonogo)
  printf("No vector found that matches test cases\n");
 else
  printVec(GET_WEIGHTS(mDP),TOT_WEIGHTS);
 
 return 0;
}

int main(int argc, char *argv[])
{
 //testWeightsFromDesc();
 //testDescFromWeights();
 testWeightAdjustment();
 //theActualProgram();
 
 return 0;
}

/* DEPRICATED
//take an integer and split it into a vector
int intToVec(int* vector, int number, int vectorWidth)
{
 int i;
 for(i=0; i<vectorWidth; i++)
  vector[i] = (number & 1<<i) ? 1 : 0;
 
 return 0;
}
*/

/* DEPRICATED
//take a vector and compress it into an integer
int vecToInt(int* number, int* vector, int vectorWidth)
{
 *number = 0;
 
 int i;
 for(i=0; i<vectorWidth; i++)
  number += (vector[i] ? 1 : 0) << i;

 return 0;
}
*/

/* DEPRICATED
//make neurons and count them
int makeNeurons(neuron* neurons, int* tot_neurons)
{

 neurons = malloc(sizeof(neuron) * *tot_neurons);
 
 return 0;
}
*/

/* DEPRICATED
int linkNeurons(neuron* neurons, int tot_neurons, int* outs, int* weights)
{
 int i
 for(i = 0; i < tot_nuerons; i++)
 {
  neurons[i]->inputs  = &outs[i/NUM_INPUTS];
  neurons[i]->out     = &outs[TOT_INPUTS+i];
  neurons[i]->weights = &weights[i/NUM_INPUTS];
 }
 
 return 0;
}
*/

/* DEPRICATED
int sanatizeInput(int* ins, int input, int tot_inputs)
{
 int i;
 int mask = 1;
 
 for(i = 0; i < tot_inputs; i++)
 {
  ins[i] = input & mask;
  mask <<= 1;
 }

 return 0;
}
*/

/* DEPRICATED
int unsanatizeInput(int* ins, int* input)
{
 *input = 0
 int mask = 1;
 int i;
 
 for(i = 0; i < TOT_INPUTS; i++)
 {
  input += mask * ins[i];
  mask <<= 1;
 }
 
 return 0;
}
*/

/* DEPRICATED
int defWinLoss(int** winloss, int* length) //set defined cases
{
 int win[] = WINS;
 int loss[] = LOSSES; //may behave strangely if elements are reapeated or out of order
 //winloss should be {5,1,7,0,15,1,16,0,17,1,254,0,255,1}

 int wincount;
 for(wincount = 0; win[wincount] != -1; wincount++);
 
 int losscount;
 for(losscount = 0; loss[losscount] != -1; losscount++);
 
 *length = wincount + losscount;
 *winloss = malloc(sizeof(int) * length * 2);
 
 int i;
 int winp = 0;
 int lossp = 0;
 
 for(i = 0; i < *length; i++)
 {
  if(win[winp] < loss[lossp])
  {
   *winloss[i*2] = win[winp];
   *winloss[i*2 + 1] = 1;
   winp++
  }
  else
  {
   *winloss[i*2] = loss[lossp];
   *winloss[i*2 + 1] = 0;
   lossp++;
  }
 }
	
 return 0;
}
*/

/* SUPERCEDED
int main(int argc, char *argv[])
{
 int tot_neurons = 1;
 int i;
 for(i = 0; i < LAYERS + 1; i++)
  tot_inputs *= NUM_INPUTS;

 int tot_neurons = (tot_inputs - 1)/(NUM_INPUTS - 1); //this function changes based on network topography
 
 neuron* neurons = malloc(sizeof(neuron) * tot_neurons);
 
 int num_outs = tot_neurons + tot_inputs;
 int num_weights = num_outs - 1; //the final output has no associated weight
 int* outs    = malloc(sizeof(int) * (num_outs)); //outs includes the inputs
 int* weights = malloc(sizeof(int) * (num_weights)); 
 int* savedweights = malloc(sizeof(int) * (num_weights));
 for(i = 0; i < num_outs - 1; i++)
  weights[i] = INIT_WEIGHTS[i];

 linkNeurons(neurons, tot_neurons, outs, weights);
  
 int* winloss;
 int wllength; 
 defWinLoss(&winloss, &wllength);
 
 int s,k;
 int matchcount;
 int lastmatch = 0;
 int weightp = 0;
 int weights = 0;
 int weightadj = 1;
 int exitcon = 0;
 
 
 
 while(!exitcon)
 {
  matchcount = 0;
  for(i = 0; i < wllength * 2; i += 2) //count matches
  {
   sanatizeInput(outs, winloss[i]);
   resolveNeurons(neurons);
   if(outs[numouts - 1] == winloss[i + 1])
    matchcount++;
  }
  
  for(i = 0; i < num_weights; i++) //save the weight vector
   savedweights[k] = weights[k];
  lastmatch = matchcount; //save the matchcount
  
  if(matchcount < wllength) //adjust one weight
  {
   weights[weightp] += weightadj;
   if(weights[weightp] > MAX_WEIGHT)
    weights[weightp] -= (MAX_WEIGHT - MIN_WEIGHT);
   else if(weights[weightp] < MIN_WEIGHT)
    weights[weightp] += (MAX_WEIGHT - MIN_WEIGHT);

   if(weightp == num_weights && weightadj > 0)
   {
   	weightadj *= -1;
   	weightp = weights;
   }
   else if(weightp == num_weights && weightadj < 0)
   {
   	weightadj *= -1;
	weightadj++;
	weightp = weights;
   }
  }
  else if(matchcount == wllength)
   exitcon = 1;
  else if(weightadj >= MAX_WEIGHT - MINWEIGHT)
   exitcon = 1;
 }
 
 
 return 0;
}
*/

