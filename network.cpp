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
#include <fstream>
#include <cmath>
#include <vector>
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
   bool IS_TRAINING;
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
   int MAX_ITERATIONS;
   double LOSS_THRESHOLD;

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
      cout << "########################################" << endl;
      cout << "Attempting to configure model\n\n";
      if (setConfigParameters(numLayers, layerSzs, isTraining,
                               numTests, wlb, wub, isLoading, 
                               lambda, thresFuncType))
      {
         cout << "Model configured successfully!\n";
         cout << "########################################" << endl;
         if (printConfigParameters())
         {
            cout << "########################################" << endl;
            allocateMemory();
            cout << "Memory allocated\n";
            initializeArrays();
            cout << "Weights intialized\n";
            cout << "########################################" << endl;
         }
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
   
   ~NeuralNetwork() {
      delete LAYER_SIZE;
      delete activations;
      delete weights;
      delete dweights;
   }

   // private:
   bool setConfigParameters(int numLayers, int* layerSzs, bool isTraining, 
                            int numTests, double wlb, double wub, bool isLoading,
                            double lambda, bool thresFuncType) 
   {
      bool modelValid = 1;

      if (numLayers != 3) 
      {
         cout << "ERROR: This neural network does not support more or less than 3 layers\n";
         modelValid = 0;
      }
      else NUM_LAYERS = numLayers;

      vector<int> invalidLayers;
      for (int i1 = 0; i1 < numLayers; i1++) 
      {
         if (layerSzs[i1] <= 0)
         {
            invalidLayers.push_back(i1);
         }
      }

      if (invalidLayers.size() > 0 || layerSzs[numLayers - 1] != 1)
      {
         if (invalidLayers.size() > 0)
         {
            cout << "ERROR: The following layers have invalid sizes (<= 0): ";
            for (int badLayer : invalidLayers) cout << badLayer << " ";
            cout << "\n";
            modelValid = 0;
         }
         if (layerSzs[numLayers - 1] != 1)
         {
            cout << "ERROR: There may only be one output activation\n";
            modelValid = 0;
         }
      }
      else LAYER_SIZE = layerSzs;

      IS_TRAINING = isTraining;

      if (numTests <= 0)
      {
         cout << "ERROR: Invalid number of tests (<= 0)\n";
         modelValid = 0;
      }
      else NUM_TESTS = numTests;

      if (wlb - wub > EPS)
      {
         cout << "ERROR: Lower bound for weights must be <= upper bound for weights\n";
         modelValid = 0;
      }
      else
      {
         WLB = wlb;
         WUB = wub;
      }

      IS_LOADING = isLoading;

      if (lambda < EPS)
      {
         cout << "ERROR: Lambda must be positive\n";
         modelValid = 0;
      }
      else LAMBDA = lambda;

      if (thresFuncType != 0)
      {
         cout << "ERROR: Threshold function type is invalid, can only be 0 (sigmoid)\n";
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

   bool printConfigParameters() 
   {
      cout << "Printing configuration paramenters:\n\n";

      cout << "Number of layers: " << NUM_LAYERS << "\n";

      cout << "Size of each layer: ";
      for (int i1 = 0; i1 < NUM_LAYERS; i1++)
      {
         cout << LAYER_SIZE[i1] << " ";
      }
      cout << endl;

      cout << "Current mode: ";
      if (IS_TRAINING) cout << "training\n";
      else cout << "testing\n";

      cout << "Number of tests: " << NUM_TESTS << "\n";

      cout << "Weights will be initialied between " << WLB << " and " << WUB << "\n";

      cout << "Lambda is " << LAMBDA << "\n";

      cout << "The threshold function used will be: ";
      if (THRESHOLD_FUNCTION_TYPE == 0) cout << "sigmoid\n";

      cout << endl;
      cout << "Is this the correct model? [y/n]" << endl;
      string input;
      cin >> input;
      if (input == "n") return 0;
      else return 1;
   } // bool printConfigParameters()

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

   void printWeights() 
   {
      cout << "Weights:\n";
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) 
      {
         cout << "From layer " << i1 << ":\n";
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) 
         {
            cout << "From activation " << i2 << ": ";
            for (int i3 = 0; i3 < LAYER_SIZE[i1 + 1]; i3++) 
            {
               cout << weights[i1][i2][i3] << " ";
            }
            cout << "\n";
         }
      }

      return;
   } // void printWeights()

   /*
   * Passes though the values in the input through the model
   * As per design, 
   * This impelentation pushes updates from layer i1 to i1 + 1, instead of pulling from i1 - 1
   */
   double evaluate(int testCase, bool printResults) 
   {
      ifstream fin("Tests/tc0" + to_string(testCase) + ".txt");
      for (int i1 = 0; i1 < LAYER_SIZE[0]; i1++) fin >> activations[0][i1];

      double trueValue;
      fin >> trueValue;

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

      double e = error(activations[NUM_LAYERS - 1][0], trueValue);

      if (printResults) {
         cout << "Test case " << testCase << ":\t";
         cout << "expected " << trueValue << ", received " << activations[NUM_LAYERS - 1][0] << "\t";
         cout << "error is: " << e << endl;
      }

      return e;
   } // double evaluate()

   // Training
   void calculateDWForHidden(int layer)
   {
      for (int j = 0; j < LAYER_SIZE[layer]; j++)
      {

      }
   }

   void calculateDWForInput()
   {
      for (int j = 0; j < LAYER_SIZE[layer]; j++)
      {

      }
   }

   void train()
   {
      double loss;
      int epochs = 0;
      do
      {
         double curLoss = 0;
         for (int i1 = 1; i1 <= NUM_TESTS; i1++)
         {
            curLoss += evaluate(i1, 0);
            calculateDWForHidden(NUM_LAYERS - 2);
            calculateDWForInput(0);
         }

         curLoss /= NUM_TESTS;
         loss = curLoss;
         epochs++;
      } 
      while (epochs <= MAX_ITERATIONS && loss > LOSS_THRESHOLD);

      cout << "Model terminated after " << epochs << " epochs with a loss of " << loss << endl;
      if (epochs > MAX_ITERATIONS)
         cout << "Model terminated due to reaching maximum number of epochs (" << MAX_ITERATIONS << ")." << endl;
      else 
         cout << "Model terminated due to reaching low enough error <=(" << LOSS_THRESHOLD << ")." << endl;
      
      return;
   }

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

   void test() 
   {
      double totError = 0;
      for (int i1 = 1; i1 <= NUM_TESTS; i1++) totError += evaluate(i1, 1);

      cout << "Average error is: " << totError / NUM_TESTS << endl;

      return;
   }
};

int main(int argc, char* argv[]) 
{
   // Desyncs 
   ios_base::sync_with_stdio(0);
   cin.tie(NULL);
   cout.tie(NULL);

   int* layerSzs = new int[3];
   layerSzs[0] = 2;
   layerSzs[1] = 3;
   layerSzs[2] = 1;

   NeuralNetwork network(3, layerSzs, 0,
                         4, -1.5, 1.5, 42, 0, 0.3, 0);
   // network.train();
   network.test();
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