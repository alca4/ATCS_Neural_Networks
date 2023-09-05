/*
* Author: Andrew Liang
* Date of Creation: 29 August 2023
* Description: A simple A-B-1 fully connected neural network
* The network has:
* an input layer with A nodes, 
* a hidden layer with B nodes, 
* and an output layer with a single node
*/

/*
Setup parameters (Constructor deals with):
A and B
train/test
number of tests
max_iterations, learning cutoff
range for initial weights (when randomizing)
load weights?
*/

#include <iostream>
#include <cmath>
using namespace std;

// Constants:
const double EPS = 1e-9;

/*
* A neural network. Honestly, I don't know how this will work right now
*/
class NeuralNetwork 
{
   // configuration parameters
   int NUM_LAYERS;
   // LAYER_SIZE[i1] stores the size of layer i1
   int* LAYER_SIZE;
   // 1 if training, 0 if testing
   bool IS_TRAIN;
   // number of test cases
   int NUM_TESTS;
   // lower and upper bounds for weights
   double WLB, WUB;
   int RNG_SEED; 
   // 1 if loading weights, 0 if randomly generated
   bool IS_LOADING;
   // learning rate
   double LAMBDA;
   // 0 if sigmoid, nothing else is valid yet
   int THRESHOLD_FUNCTION_TYPE;

   // values of the activations, activations[i1][i2] stores activation i2 at layer i1 (0-indexed)
   double** activations;
   // values of the weights, weights[i1][i2][i3] stores the weight connecting activation i2 of layer i1 to activation i3 to layer i1 + 1
   double*** weights;
   // values of the changes of weights, dweights[i1][i2][i3] stores the change to w[i1][i2][i3] the model learns
   double*** dweights;

   public:
   NeuralNetwork(int numLayers, int* layerSzs, bool isTraining, 
                 int numTests, double wlb, double wub, int rngSeed,
                 bool isLoading, double lambda, bool thresFuncType) 
   {
      cout << "The model has the following configuration errors:\n";
      if (setConfigParameters(numLayers, layerSzs, isTraining,
                               numTests, wlb, wub, isLoading, 
                               lambda, thresFuncType))
      {
         cout << "Model configured successfully!\n";
         // printConfigParameters();
         // allocateMemory();
         // initializeArrays();
      }
      else 
      {
         cout << "Model was not configured\n";
      }

      return;
   }
   /*
   NeuralNetwork(int numLayers, int* layerSzs, bool isTraining, 
                 int numTests, double wlb, double wub, int rngSeed,
                 bool isLoading, double lambda, bool thresFuncType) 
   */
   
   // private:
   bool setConfigParameters(int numLayers, int* layerSzs, bool isTraining, 
                            int numTests, double wlb, double wub, bool isLoading,
                            double lambda, bool thresFuncType) 
   {
      bool modelValid = 1;

      if (numLayers != 3) 
      {
         cout << "This neural network does not support more than 3 layers\n";
         modelValid = 0;
      }
      else NUM_LAYERS = numLayers;

      bool hasInvalidLayerSz = 0;
      for (int i = 0; i < numLayers; i++) 
      {
         if (layerSzs[i] <= 0)
         {
            hasInvalidLayerSz = 1;
         }
      }

      if (layerSzs[numLayers - 1] != 1)
      {
         cout << "There may only be one output activation\n";
         modelValid = 0;
      }
      else if (hasInvalidLayerSz)
      {
         cout << "At least one layer has an invalid size (<= 0)\n";
         modelValid = 0;
      }
      else LAYER_SIZE = layerSzs;

      IS_TRAIN = isTraining;

      if (numTests <= 0)
      {
         cout << "Invalid number of tests (<= 0)\n";
         modelValid = 0;
      }
      else NUM_TESTS = numTests;

      if (wlb - wub > EPS)
      {
         cout << "lower bound for weights must be <= upper bound for weights\n";
         modelValid = 0;
      }
      else
      {
         WLB = wlb;
         WUB = wub;
      }

      IS_LOADING = isLoading;

      LAMBDA = lambda;

      if (thresFuncType != 0)
      {
         cout << "Threshold function type is invalid, can only be 0 (sigmoid)\n";
         modelValid = 0;
      }
      else THRESHOLD_FUNCTION_TYPE = thresFuncType;
      
      return modelValid;
   } 
   /*
   bool setConfigParameters(int numLayers, int* layerSzs, bool isTraining, 
                            int numTests, double wlb, double wub, bool isLoading,
                            double lambda, bool thresFuncType) 
   */

   void printConfigParameters() 
   {
      cout << "Number of layers: " << NUM_LAYERS << endl;
      cout << "Size of each layer: " << endl;

      cout << "Layer sizes: ";
      for (int i1 = 0; i1 < NUM_LAYERS; i1++) 
         cout << LAYER_SIZE[i1] << "-" [i1 == NUM_LAYERS - 1]; // 
      cout << endl;

      return;
   }

   /*
   * Initializes arrays based on the following dimensions:
   * a: NUM_LAYERS, LAYER_SIZE
   */
   void allocateMemory() 
   {
      activations = new double*[NUM_LAYERS];
      for (int i1 = 0; i1 < NUM_LAYERS; i1++) 
      {
         activations[i1] = new double[LAYER_SIZE[i1]];
      }

      weights = new double**[NUM_LAYERS - 1];
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) 
      {
         weights[i1] = new double*[LAYER_SIZE[i1]];
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) 
         {
            weights[i1][i2] = new double[LAYER_SIZE[i1 + 1]];
         }
      }

      dweights = new double**[NUM_LAYERS - 1];
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) 
      {
         dweights[i1] = new double*[LAYER_SIZE[i1]];
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) 
         {
            dweights[i1][i2] = new double[LAYER_SIZE[i1 + 1]];
         }
      }

      return;
   } // void allocateMemory()

   /*
   * generates a pseudorandom floating point value in [lb, ub]
   */
   double genRand(double lb, double ub) 
   {
      double ret = (1.0 * rand()) / RAND_MAX;
      ret *= (ub - lb);
      ret += lb;
      return ret;
   }

   /*
   * Initializes all weights in the range [WLB, WUB]
   */
   void initializeArrays() {
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) 
      {
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) 
         {
            for (int i3 = 0; i3 < LAYER_SIZE[i1 + 1]; i3++) 
            {
               weights[i1][i2][i3] = genRand(WLB, WUB);
            }
         }
      }

      return;
   } // void initializeArrays()

   void printWeights() {
      cout << "Weights: " << endl;
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) 
      {
         cout << "From layer " << i1 << ": " << endl;
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) 
         {
            cout << "From activation " << i2 << ": ";
            for (int i3 = 0; i3 < LAYER_SIZE[i1 + 1]; i3++) 
            {
               cout << weights[i1][i2][i3] << " ";
            }
            cout << endl;
         }
      }

      return;
   } // void printWeights()

   // Testing
   double thresholdFunction(double val) 
   {
      if (THRESHOLD_FUNCTION_TYPE == 0) 
      {
         return 1.0 / (1.0 + exp(-val));
      }
      else
      {
         cout << "Threshold function invalid, somehow didn't catch this during configuration\n";
         return -1;
      }
   }

   double error(double f, double t) 
   {
      return 0.5 * (t - f) * (t - f);
   }

   /*
   * Passes though the values in the input through the model
   * As per design, 
   * This impelentation pushes updates from layer i1 to i1 + 1, instead of pulling from i1 - 1
   */
   double evaluate(double trueValue) 
   {
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) 
      {
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) 
         {
            for (int i3 = 0; i3 < LAYER_SIZE[i1 + 1]; i3++) 
            {
               activations[i1 + 1][i3] += activations[i1][i2] * weights[i1][i2][i3];
            }
         }

         for (int i2 = 0; i2 < LAYER_SIZE[i1 + 1]; i2++) 
         {
            activations[i1 + 1][i2] = thresholdFunction(activations[i1 + 1][i2]);
         }
      }

      return error(activations[NUM_LAYERS - 1][0], trueValue);
   } // double evaluate()

   void loadTest(const string& FILENAME) 
   {
      // ifstream fin(FILENAME);
      
      return;
   }
};

int main(int argc, char* argv[]) 
{
   int* layerSzs = new int[3];
   layerSzs[0] = 2;
   layerSzs[1] = 3;
   layerSzs[2] = 1;

   NeuralNetwork network(2, layerSzs, 0,
                         1, -1.5, 1.5, 42, 0, 0.3, 0);
   // if (*argv[1] == '0') network.train();
   // else network.test();
   // network.printReport();
   // network.evaluate();

   // cout << network.genRand(-1.5, 1.5) << endl;
   // cout << network.genRand(-1.5, 1.5) << endl;
   // cout << network.genRand(-1.5, 1.5) << endl;
   // cout << network.genRand(-1.5, 1.5) << endl;
   // cout << network.genRand(-1.5, 1.5) << endl;

   return 0;
}