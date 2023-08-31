/*
* Author: Andrew Liang
* Date of Creation: 29 August 2023
* Description: A simple 2-3-1 neural network
* The network has:
* an input layer with 2 nodes, 
* a hidden layer with 3 nodes, 
* and an output layer with a single node
*/

#include <iostream>
#include <cmath>
#include <cassert>
using namespace std;

// Constants 
const int SEED = 42;                      // i could know a little more about the universe

/*
* A neural network. Honestly, I don't know how this will work right now
*/
class NeuralNetwork {
   // configuration parameters
   int NUM_LAYERS;
   // LAYER_SIZE[i1] stores the size of layer i1
   int* LAYER_SIZE;
   // lower and upper bounds for weights
   int WLB = -1;
   int WUB = 1;
   // values of the activations, activations[i1][i2] stores activation i2 at layer i1 (0-indexed)
   double** activations;
   // values of the weights, weights[i1][i2][i3] stores the weight connecting activation i2 of layer i1 to activation i3 to layer i1 + 1
   double*** weights;
   // values of the changes of weights, dweights[i1][i2][i3] stores the change to w[i1][i2][i3] the model learns
   double*** dweights;
   /*
   * type of threshold function
   * currently:
   *  0 = sigmoid, currently only choice
   */
   int THRESHOLD_FUNCTION_TYPE;

   public:
   void setupModel(int numLayers, int* layerSzs, int thresFuncType) {
      setConfigParameters(numLayers, layerSzs, thresFuncType);
      printConfigParameters();
      allocateMemory();
      initializeArrays();
      // printWeights();

      return;
   }
   
   // private:
   void setConfigParameters(int numLayers, int* layerSzs, int thresFuncType) {
      NUM_LAYERS = numLayers;
      LAYER_SIZE = layerSzs;
      THRESHOLD_FUNCTION_TYPE = thresFuncType;
      
      return;
   }

   void printConfigParameters() {
      cout << "Number of layers: " << NUM_LAYERS << endl;

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
   void allocateMemory() {
      activations = new double*[NUM_LAYERS];
      for (int i1 = 0; i1 < NUM_LAYERS; i1++) {
         activations[i1] = new double[LAYER_SIZE[i1]];
      }

      weights = new double**[NUM_LAYERS - 1];
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) {
         weights[i1] = new double*[LAYER_SIZE[i1]];
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) {
            weights[i1][i2] = new double[LAYER_SIZE[i1 + 1]];
         }
      }

      dweights = new double**[NUM_LAYERS - 1];
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) {
         dweights[i1] = new double*[LAYER_SIZE[i1]];
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) {
            dweights[i1][i2] = new double[LAYER_SIZE[i1 + 1]];
         }
      }

      return;
   }

   /*
   * generates a pseudorandom floating point value in [lb, ub]
   */
   double genRand(double lb, double ub) {
      double ret = (1.0 * rand()) / RAND_MAX;
      ret *= (ub - lb);
      ret += lb;
      return ret;
   }

   /*
   * Initializes all weights in the range [WLB, WUB]
   */
   void initializeArrays() {
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) {
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) {
            for (int i3 = 0; i3 < LAYER_SIZE[i1 + 1]; i3++) {
               weights[i1][i2][i3] = genRand(WLB, WUB);
            }
         }
      }

      return;
   }

   void printWeights() {
      cout << "Weights: " << endl;
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) {
         cout << "From layer " << i1 << ": " << endl;
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) {
            cout << "From activation " << i2 << ": ";
            for (int i3 = 0; i3 < LAYER_SIZE[i1 + 1]; i3++) {
               cout << weights[i1][i2][i3] << " ";
            }
            cout << endl;
         }
      }

      return;
   }

   // Time to test
   double thresholdFunction(double val) {
      assert(THRESHOLD_FUNCTION_TYPE == 0);
      if (THRESHOLD_FUNCTION_TYPE == 0) return 1.0 / (1.0 + exp(-val));
      return -1234;
   }

   double error(double f, double t) {
      return 0.5 * (t - f) * (t - f);
   }

   /*
   * Passes though the values in the input through the model
   * As per design, 
   * This impelentation pushes updates from layer i1 to i1 + 1, instead of pulling from i1 - 1
   */
   double evaluate(double trueValue) {
      for (int i1 = 0; i1 < NUM_LAYERS - 1; i1++) {
         for (int i2 = 0; i2 < LAYER_SIZE[i1]; i2++) {
            for (int i3 = 0; i3 < LAYER_SIZE[i1 + 1]; i3++) {
               activations[i1 + 1][i3] += activations[i1][i2] * weights[i1][i2][i3];
            }
         }

         for (int i2 = 0; i2 < LAYER_SIZE[i1 + 1]; i2++) {
            activations[i1 + 1][i2] = thresholdFunction(activations[i1 + 1][i2]);
         }
      }

      return error(activations[NUM_LAYERS - 1][0], trueValue);
   }

   void loadTest() {

   }
};

int main(int argc, char* argv[]) {
   srand(SEED);                                    // Seeds the random number generator

   NeuralNetwork network;
   int* layerSzs = new int[3];
   layerSzs[0] = 2;
   layerSzs[1] = 3;
   layerSzs[2] = 1;
   network.setupModel(3, layerSzs, 0);
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