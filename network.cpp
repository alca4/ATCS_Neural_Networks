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
#include <random>
#include <iomanip>
using namespace std;

// Constants:
const double EPS = 1e-9;

/*
* A neural network. Honestly, I don't know how this will work right now
*/
struct NeuralNetwork 
{
   int NUM_LAYERS;                  // configuration parameters
   int* LAYER_SIZE;                 // LAYER_SIZE[i1] stores the size of layer i1
   bool IS_TRAINING;                // 1 if training, 0 if testing
   int NUM_TESTS;                   // number of test cases
   double WLB, WUB;                 // lower and upper bounds for weights
   bool IS_LOADING;                 // 1 if loading weights, 0 if randomly generated
   double LAMBDA;                   // learning rate
   int THRESHOLD_FUNCTION_TYPE;     // 0 if sigmoid
   int MAX_ITERATIONS;              // max iterations before stopping
   double ERROR_THRESHOLD;          // min error to continue running

   // values of the activations, activations[i1][i2] stores activation i2 at layer i1 (0-indexed)
   double** activations;
   // values of the weights, weights[i1][i2][i3] stores the weight connecting activation i2 of layer i1 to activation i3 to layer i1 + 1
   double*** weights;
   // values of the changes of weights, dweights[i1][i2][i3] stores the change to w[i1][i2][i3] the model learns
   double*** dweights;
   // values of the inputs in each test, inputValues[i1][i2] stores the i2th input of the i1th test case
   double** inputValues;
   // true values of each test
   double* trueValues;

   NeuralNetwork(int numLayers, int* layerSzs, bool isTraining, 
                 int numTests, double wlb, double wub, 
                 bool isLoading, double lambda, bool thresFuncType,
                 int maxIter, double errorThreshold) 
   {
      cout << "########################################" << endl;
      cout << "Attempting to configure model" << endl << endl;
      if (setConfigParameters(numLayers, layerSzs, isTraining,
                               numTests, wlb, wub, isLoading, 
                               lambda, thresFuncType, maxIter, errorThreshold))
      {
         cout << "Model configured successfully!" << endl;
         cout << "########################################" << endl;
         if (printConfigParameters())
         {
            cout << "########################################" << endl;
            allocateMemory();
            cout << "Memory allocated" << endl;
            initializeArrays();
            cout << "Weights intialized" << endl;
            loadTests();
            cout << "Tests loaded" << endl;
            cout << "########################################" << endl;
         }
      }
      else 
      {
         cout << "Model was not configured" << endl;
      }
      
      cout << "finished constructing" << endl;

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
      delete inputValues;
      delete trueValues;
   }

   // private:
   bool setConfigParameters(int numLayers, int* layerSzs, bool isTraining, 
                            int numTests, double wlb, double wub, bool isLoading,
                            double lambda, bool thresFuncType, 
                            int maxIter, double errorThreshold) 
   {
      bool modelValid = 1;

      if (numLayers != 3) 
      {
         cout << "ERROR: This neural network does not support more or less than 3 layers" << endl;
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
            cout << endl;
            modelValid = 0;
         }
         if (layerSzs[numLayers - 1] != 1)
         {
            cout << "ERROR: There may only be one output activation" << endl;
            modelValid = 0;
         }
      }
      else LAYER_SIZE = layerSzs;

      IS_TRAINING = isTraining;

      if (numTests <= 0)
      {
         cout << "ERROR: Invalid number of tests (<= 0)" << endl;
         modelValid = 0;
      }
      else NUM_TESTS = numTests;

      if (wlb - wub > EPS)
      {
         cout << "ERROR: Lower bound for weights must be <= upper bound for weights" << endl;
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
         cout << "ERROR: Lambda must be positive" << endl;
         modelValid = 0;
      }
      else LAMBDA = lambda;

      if (thresFuncType != 0)
      {
         cout << "ERROR: Threshold function type is invalid, can only be 0 (sigmoid)" << endl;
         modelValid = 0;
      }
      else THRESHOLD_FUNCTION_TYPE = thresFuncType;

      if (maxIter <= 0)
      {
         cout << "ERROR: Model must run for positive number of epochs." << endl;
         modelValid = 0;
      }
      else MAX_ITERATIONS = maxIter;

      if (errorThreshold < 0)
      {
         cout << "ERROR: Must have positive error threshold." << endl;
         modelValid = 0;
      }
      else ERROR_THRESHOLD = errorThreshold;
      
      return modelValid;
   } 
   /*
   bool setConfigParameters(int numLayers, int* layerSzs, bool isTraining, 
                            int numTests, double wlb, double wub, bool isLoading,
                            double lambda, bool thresFuncType) 
   */

   bool printConfigParameters() 
   {
      cout << "Printing configuration paramenters:" << endl << endl;

      cout << "Number of layers: " << NUM_LAYERS << endl;

      cout << "Size of each layer: ";
      for (int i1 = 0; i1 < NUM_LAYERS; i1++)
      {
         cout << LAYER_SIZE[i1] << " ";
      }
      cout << endl;

      cout << "Current mode: ";
      if (IS_TRAINING) cout << "training" << endl;
      else cout << "testing" << endl;

      cout << "Number of tests: " << NUM_TESTS << endl;

      cout << "Weights will be initialied between " << WLB << " and " << WUB << endl;

      cout << "Lambda is " << LAMBDA << endl;

      cout << "The threshold function used will be: ";
      if (THRESHOLD_FUNCTION_TYPE == 0) cout << "sigmoid" << endl;

      cout << "The model will stop when it reached " << MAX_ITERATIONS
           << " epochs or reaches a error lower than " << ERROR_THRESHOLD << endl;

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

      inputValues = new double*[NUM_TESTS];
      for (int i1 = 0; i1 < NUM_TESTS; i1++) 
      {
         inputValues[i1] = new double[LAYER_SIZE[0]];
      }

      trueValues = new double[NUM_TESTS];

      return;
   } // void allocateMemory()

   /*
   * generates a pseudorandom floating point value in [lb, ub]
   */
   double genRand() 
   {
      double ret = (1.0 * rand()) / RAND_MAX;
      ret *= (WUB - WLB);
      ret += WLB;
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
               weights[i1][i2][i3] = genRand();
            }
         }
      }

      return;
   } // void initializeArrays()

   void printWeights() 
   {
      cout << "Weights:" << endl;
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) 
      {
         cout << "From layer " << i1 << ":" << endl;
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) 
         {
            cout << "From activation " << i2 << ": ";
            for (int i3 = 0; i3 < LAYER_SIZE[i1 + 1]; i3++) 
            {
               cout << fixed << setprecision(3) << weights[i1][i2][i3] << " ";
            }
            cout << endl;
         }
      }

      return;
   } // void printWeights()

   void loadTests()
   {
      for (int i1 = 0; i1 < NUM_TESTS; i1++)
      {
         string fname;
         if (i1 < 10) fname = "tc0" + to_string(i1) + ".txt";
         else fname = "tc" + to_string(i1) + ".txt";

         ifstream fin("Tests/" + fname);
         for (int i2 = 0; i2 < LAYER_SIZE[0]; i2++)
         {
            fin >> inputValues[i1][i2];
            cout << fixed << setprecision(3) << inputValues[i1][i2] << " ";
         }
         fin >> trueValues[i1];
         cout << fixed << setprecision(3) << trueValues[i1] << endl;
      }
   }

   /*
   * Passes though the values in the input through the model
   * As per design, 
   * This impelentation pushes updates from layer i1 to i1 + 1, instead of pulling from i1 - 1
   */
   double evaluate(int testCase, bool printResults) 
   {
      for (int i1 = 0; i1 < LAYER_SIZE[0]; i1++) 
         activations[0][i1] = inputValues[testCase][i1];

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

      double e = error(activations[NUM_LAYERS - 1][0], trueValues[testCase]);

      if (printResults) 
      {
         cout << "Test case " << testCase << ":\t";
         cout << fixed << setprecision(3) << "expected " << trueValues[testCase] << ", received " << activations[NUM_LAYERS - 1][0] << "\t";
         cout << fixed << setprecision(3) << "error is: " << e << endl;
      }

      return e;
   } // double evaluate()

   // Training
   void calculateDWForHidden(int layer, int testCase)
   {
      double fOfThetaj = 0;
      for (int j = 0; j < LAYER_SIZE[layer]; j++)
      {
         fOfThetaj += activations[layer][j] * weights[layer][j][0];
      }
      fOfThetaj = thresholdFunction(fOfThetaj);

      for (int j = 0; j < LAYER_SIZE[layer]; j++)
      {
         dweights[layer][j][0] = (trueValues[testCase] - fOfThetaj) * 
                                 (fOfThetaj * (1 - fOfThetaj)) * 
                                 activations[layer][j] *
                                 LAMBDA;
      }
   }

   void calculateDWForInput(int layer, int testCase)
   {
      for (int i = 0; i < LAYER_SIZE[layer + 2]; i++) 
      {
         double fOfThetaj = 0;
         for (int j = 0; j < LAYER_SIZE[layer + 1]; j++)
         {
            fOfThetaj += activations[layer + 1][j] * weights[layer + 1][j][0];
         }
         fOfThetaj = thresholdFunction(fOfThetaj);

         for (int j = 0; j < LAYER_SIZE[layer + 1]; j++)
         {
            double fOfThetak = 0;
            for (int k = 0; k < LAYER_SIZE[layer]; k++)
            {
               fOfThetak += activations[layer][k] * weights[layer][k][j];
            }
            fOfThetak = thresholdFunction(fOfThetak);
            
            for (int k = 0; k < LAYER_SIZE[layer]; k++)
            {
               dweights[layer][k][j] = (trueValues[testCase] - fOfThetaj) * 
                                       (fOfThetaj * (1 - fOfThetaj)) * 
                                       weights[layer + 1][j][i] *
                                       (fOfThetak * (1 - fOfThetak)) *
                                       activations[layer][k] *
                                       LAMBDA;
            }
         }
      }
   }

   void applyDW()
   {
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++)
      {
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++)
         {
            for (int i3 = 0; i3 < LAYER_SIZE[i1 + 1]; i3++)
            {
               weights[i1][i2][i3] += dweights[i1][i2][i3];
               dweights[i1][i2][i3] = 0;
            }
         }
      }
   }

   void train()
   {
      double error;
      int epochs = 0;
      do
      {
         double curError = 0;
         for (int i1 = 0; i1 < NUM_TESTS; i1++)
         {
            curError += evaluate(i1, 0);
            calculateDWForHidden(NUM_LAYERS - 2, i1);
            calculateDWForInput(0, i1);
            applyDW();
         }

         error = curError;
         epochs++;

         if (epochs % 5000 == 0) 
         {
            cout << "Model has run " << epochs << " epochs ";
            cout << fixed << setprecision(3) << "and achieved an error of " << error << endl;
         }
      } 
      while (epochs < MAX_ITERATIONS && error > ERROR_THRESHOLD);

      cout << fixed << setprecision(3) << "Model terminated after " << epochs << " epochs with an error of " << error << endl;
      if (epochs == MAX_ITERATIONS) 
         cout << "Model terminated due to reaching maximum number of epochs (" << MAX_ITERATIONS << ")." << endl;
      else 
         cout << "Model terminated due to reaching low enough error <=(" << ERROR_THRESHOLD << ")." << endl;
      
      for (int i1 = 0; i1 < NUM_TESTS; i1++) evaluate(i1, 1);
      
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
         cout << "Threshold function invalid, somehow didn't catch this during configuration" << endl;
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

      cout << fixed << setprecision(3) << "Average error is: " << totError / NUM_TESTS << endl;

      return;
   }
};

int main(int argc, char* argv[]) 
{
   // sets the seed to the current time, ensuring different results for every run
   srand(time(NULL));

   int* layerSzs = new int[3];
   layerSzs[0] = 2;
   layerSzs[1] = 4;
   layerSzs[2] = 1;

   NeuralNetwork network(3, layerSzs, 0,
                         4, -1.5, 1.5, 0, 0.3, 0, 200000, 0.002);
   network.train();

   return 0;
}