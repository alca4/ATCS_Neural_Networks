/*
* Author: Andrew Liang
* Date of Creation: 29 August 2023
* Description: A simple A-B-1 fully connected neural network
* The network has:
* an input layer with A nodes, 
* a hidden layer with B nodes, 
* and an output layer with a single node
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <iomanip>
using namespace std;

/*
* Constants
*/ 
const double EPSILON = 1e-9;
const int EXPECTED_NUM_LAYERS = 3;
const int PRECISION = 4;
const int ITERATION_PRINTING_FREQUENCY = 5000;
const string CONFIG_FILE = "config.txt";

/*
* A neural network. Honestly, I don't know how this will work right now
*/
struct NeuralNetwork 
{
   // not actually constants :|
   const string TARGET = "PARAM: "; // parameter indetifier
   const int INPUT_LAYER = 0;             // input layer is the 0st layer
   const int HIDDEN_LAYER = 1;            // hidden layer is the 1st layer
   const int OUTPUT_LAYER = 2;            // output layer is the 2nd layer

   int numLayers;                  // configuration parameters
   int* layerSize;                 // layerSize[i1] stores the size of layer i1
   bool isTraining;                // 1 if training, 0 if testing
   int numTests;                   // number of test cases
   double wlb, wub;                 // lower and upper bounds for weights
   double lambda;                   // learning rate
   int thresholdFunctionType;     // 0 if sigmoid
   int maxIterations;              // max iterations before stopping
   double errorThreshold;          // min error to continue running

   /*
   * values of the activations, activations[i1][i2] stores activation i2 at layer i1 (0-indexed)
   */
   double** activations;
   /*
   * values of the weights, 
   * weights[i1][i2][i3] stores the weight connecting activation i2 of layer i1 to activation i3 to layer i1 + 1
   */
   double*** weights;
   /*
   * values of the changes of weights, dweights[i1][i2][i3] stores the change to w[i1][i2][i3] the model learns
   */
   double*** dweights;
   /* 
   * values of the inputs in each test, inputValues[i1][i2] stores the i2th input of the i1th test case
   */
   double** inputValues;
   /*
   * true values of each test
   */
   double* trueValues;

   NeuralNetwork() 
   {
      cout << "########################################" << endl;
      cout << "Attempting to configure model" << endl << endl;
      if (inputConfigParameters())
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
         } // if (printConfigParameters())
      } // if (inputConfigParameters())
      else cout << "Model was not configured" << endl;

      return;
   } // NeuralNetwork()
   
   ~NeuralNetwork() {
      delete layerSize;
      delete activations;
      delete weights;
      delete trueValues;

      if (isTraining)
      {
         delete dweights;
         delete inputValues;
      }
      
      return;
   }

   /*
   * ##################################################
   *        Methods used to configure the model
   * ##################################################
   */

   /*
   * Inputs the integer contained in the first characters in the file
   * although it is ok to have more characters after the number
   */
   int getNextInteger(ifstream& fin)
   {
      string line;
      getline(fin, line);

      string num;
      int idx = 0;
      while (idx < (int) line.length() && line[idx] != ' ')
      {
         num += line[idx];
         idx++;
      }

      return stoi(num);
   } // int getNextInteger(ifstream& fin)

   /*
   * Inputs the floating point number contained in the first characters in the file
   * although it is ok to have more characters after the number
   */
   double getNextDouble(ifstream& fin)
   {
      string line;
      getline(fin, line);

      string num;
      int idx = 0;
      while (idx < (int) line.length() && line[idx] != ' ') 
      {
         num += line[idx];
         idx++;
      }

      return stod(num);
   } // double getNextDouble(ifstream& fin)

   /*
   * gets the next line that is not a comment (does not begin with '/' or '*')
   */
   string getNextParameter(ifstream& fin)
   {
      string line;
      while (!fin.eof() && (line.length() <= TARGET.length() || line.substr(0, TARGET.length()) != TARGET))
         getline(fin, line);
      
      string parameter;
      int idx = TARGET.size();
      while (idx < (int) line.length() && line[idx] != ' ') 
      {
         parameter += line[idx];
         idx++;
      }

      return parameter;
   }

   bool inputConfigParameters() 
   {
      bool modelValid = 1;
      ifstream fin(CONFIG_FILE);

      while (!fin.eof())
      {
         string parameter = getNextParameter(fin);
         
         if (parameter == "numLayers")
         {
            numLayers = getNextInteger(fin);
            if (numLayers != EXPECTED_NUM_LAYERS) 
            {
               cout << "ERROR: This neural network does not support more or less than " 
                    << EXPECTED_NUM_LAYERS << " layers" << endl;
               modelValid = 0;
            }
         } // if (parameter == "numLayers")

         if (parameter == "layerSizes")
         {
            layerSize = new int[numLayers];
            int numInvalidLayers = 0;

            for (int i1 = 0; i1 < numLayers; i1++) 
            {
               layerSize[i1] = getNextInteger(fin);
               if (layerSize[i1] <= 0)
               {
                  numInvalidLayers++;

                  if (numInvalidLayers == 1) // only print the following when the first invalid layer is discovered
                     cout << "ERROR: The following layers have invalid sizes (<= 0): ";
                  cout << layerSize[i1] << " ";

                  modelValid = 0;
               }
            } // for (int i1 = 0; i1 < layerSizes.length(); i1++) 
            if (numInvalidLayers > 0) cout << endl;
         } // if (parameter == "layerSizes")

         if (parameter == "isTraining")
         {
            if (getNextInteger(fin) == 1) isTraining = true;
            else isTraining = false;
         } // if (parameter == "isTraining")

         if (parameter == "numTests")
         {
            numTests = getNextInteger(fin);
            if (numTests <= 0)
            {
               cout << "ERROR: Invalid number of tests (<= 0)" << endl;
               modelValid = 0;
            }
         } // if (parameter == "numTests")

         if (parameter == "wb")
         {
            wlb = getNextDouble(fin);
            wub = getNextDouble(fin);
            if (wlb - wub > EPSILON)
            {
               cout << "ERROR: Lower bound for weights must be <= upper bound for weights" << endl;
               modelValid = 0;
            }
         } // if (parameter == "wb")

         if (parameter == "lambda")
         {
            lambda = getNextDouble(fin);
            if (lambda < EPSILON)
            {
               cout << "ERROR: lambda must be positive" << endl;
               modelValid = 0;
            }
         } // if (parameter == "lambda")

         if (parameter == "thresholdFunctionType")
         {
            thresholdFunctionType = getNextInteger(fin);
            if (thresholdFunctionType)
            {
               cout << "ERROR: Threshold function type is invalid, can only be 0 (sigmoid)" << endl;
               modelValid = 0;
            }
         } // if (parameter == "thresholdFunctionType")

         if (parameter == "maxIterations")
         {
            maxIterations = getNextInteger(fin);
            if (maxIterations <= 0)
            {
               cout << "ERROR: Model must run for positive number of iterations." << endl;
               modelValid = 0;
            }
         } // if (parameter == "maxIterations")

         if (parameter == "errorThreshold")
         {
            errorThreshold = getNextDouble(fin);
            if (errorThreshold < 0)
            {
               cout << "ERROR: Must have positive error threshold." << endl;
               modelValid = 0;
            }
         } // if (parameter == "errorThreshold")
      } // while (!fin.eof())   

      return modelValid;
   } // bool setConfigParameters() 

   /*
   * Prints the configuration of the model
   * Asks the user for confirmation
   * Only coninues with training/testing if user confirms the configuration is correct
   */
   bool printConfigParameters() 
   {
      cout << "Printing configuration paramenters:" << endl << endl;

      cout << "Number of layers: " << numLayers << endl;

      cout << "Size of each layer: ";
      for (int i1 = 0; i1 < numLayers; i1++) cout << layerSize[i1] << " ";
      cout << endl;

      cout << "The threshold function used will be: ";
      if (thresholdFunctionType == 0) cout << "sigmoid" << endl;

      cout << "Current mode: ";
      if (isTraining) cout << "training" << endl;
      else cout << "testing" << endl;

      if (isTraining)
      {
         cout << "Number of tests: " << numTests << endl;

         cout << "Weights will be initialized between " << wlb << " and " << wub << endl;

         cout << "lambda is " << lambda << endl;

         cout << "The model will stop when it reached " << maxIterations
              << " iterations or reaches a error lower than " << errorThreshold << endl;
      } // if (isTraining)

      cout << endl;
      cout << "Is this the correct model? [y/n]" << endl;
      string input;
      cin >> input;
      return (input != "n");
   } // bool printConfigParameters()

   /*
   * Initializes arrays based on the following dimensions:
   * a: numLayers, layerSize of current layer
   * weights and dweights: numLayers - 1, layerSize of current layer, layerSize of next layer
   */
   void allocateMemory() 
   {
      activations = new double*[numLayers];
      for (int i1 = 0; i1 < numLayers; i1++) 
         activations[i1] = new double[layerSize[i1]];

      weights = new double**[numLayers - 1];
      if (isTraining) dweights = new double**[numLayers - 1];
      for (int i1 = 0; i1 < numLayers - 1; i1++) 
      {
         weights[i1] = new double*[layerSize[i1]];
         if (isTraining) dweights[i1] = new double*[layerSize[i1]];
         for (int i2 = 0; i2 < layerSize[i1]; i2++) 
         {
            weights[i1][i2] = new double[layerSize[i1 + 1]];
            if (isTraining) dweights[i1][i2] = new double[layerSize[i1 + 1]];
         }
      } // for (int i1 = 0; i1 < numLayers - 1; i1++) 

      inputValues = new double*[numTests];
      for (int i1 = 0; i1 < numTests; i1++) 
         inputValues[i1] = new double[layerSize[0]];

      trueValues = new double[numTests];

      return;
   } // void allocateMemory()

   /*
   * generates a pseudorandom floating point value in [wlb, wub]
   */
   double genRand() 
   {
      double ret = (1.0 * rand()) / RAND_MAX;
      ret *= (wub - wlb);
      ret += wlb;
      return ret;
   }

   /*
   * Initializes all weights in the range [wlb, wub]
   */
   void initializeArrays() {
      for (int i1 = 0; i1 < numLayers - 1; i1++) 
         for (int i2 = 0; i2 < layerSize[i1]; i2++) 
            for (int i3 = 0; i3 < layerSize[i1 + 1]; i3++) 
               weights[i1][i2][i3] = genRand();

      return;
   } // void initializeArrays()

   void printWeights() 
   {
      cout << "Weights:" << endl;
      for (int i1 = 0; i1 < numLayers - 1; i1++) 
      {
         cout << "From layer " << i1 << ":" << endl;
         for (int i2 = 0; i2 < layerSize[i1]; i2++) 
         {
            cout << "From activation " << i2 << ": ";
            for (int i3 = 0; i3 < layerSize[i1 + 1]; i3++) 
               cout << fixed << setprecision(PRECISION) << weights[i1][i2][i3] << " ";
            cout << endl;
         }
      } // for (int i1 = 0; i1 < numLayers - 1; i1++) 

      return;
   } // void printWeights()

   /*
   * ##################################################
   *        Methods used in training and testing
   * ##################################################
   */

   /*
   * stores values in each test case into inputValues and trueValues
   */
   void loadTests()
   {
      for (int i1 = 0; i1 < numTests; i1++)
      {
         /*
         * tests will be stored under Tests/
         * with the format described below
         */
         string fname = "Tests/tc" + to_string(i1) + ".txt";

         ifstream fin(fname);
         for (int i2 = 0; i2 < layerSize[0]; i2++)
         {
            fin >> inputValues[i1][i2];
            cout << fixed << setprecision(PRECISION) << inputValues[i1][i2] << " ";
         }
         fin >> trueValues[i1];
         cout << fixed << setprecision(PRECISION) << trueValues[i1] << endl;
      } // for (int i1 = 0; i1 < numTests; i1++)

      return;
   } // void loadTests()

   /*
   * computes the threshold function depending on value of thresholdFunctionType
   * type is 0: sigmoid
   * if type is not any of the above values, it should be captured when configuration is initially inputted
   */
   double thresholdFunction(double val) 
   {
      if (thresholdFunctionType == 0) 
      {
         return 1.0 / (1.0 + exp(-val));
      }
      else
      {
         cout << "Threshold function invalid, somehow didn't catch this during configuration" << endl;
         return -1;
      }
   } // double thresholdFunction(double val)

   double error(double f, double t) 
   {
      return 0.5 * (t - f) * (t - f);
   }

   /*
   * Passes though the values in the input through the model
   * As per design, 
   * This impelentation pushes updates from layer i1 to i1 + 1, instead of pulling from i1 - 1
   */
   double forwardPass(int testCase, bool printResults) 
   {
      for (int i1 = 0; i1 < layerSize[0]; i1++) 
         activations[0][i1] = inputValues[testCase][i1];

      for (int i1 = 0; i1 < numLayers - 1; i1++) 
      {
         for (int i2 = 0; i2 < layerSize[i1]; i2++) 
            for (int i3 = 0; i3 < layerSize[i1 + 1]; i3++) 
               activations[i1 + 1][i3] += activations[i1][i2] * weights[i1][i2][i3];

         for (int i2 = 0; i2 < layerSize[i1 + 1]; i2++) 
            activations[i1 + 1][i2] = thresholdFunction(activations[i1 + 1][i2]);
      }

      double e = error(activations[2][0], trueValues[testCase]);

      if (printResults) 
      {
         cout << "Test case " << testCase << ":\t";
         cout << fixed << setprecision(PRECISION) << "expected " << trueValues[testCase] << ", received " << activations[2][0] << "\t";
         cout << fixed << setprecision(PRECISION) << "error is: " << e << endl;
      }

      return e;
   } // double forwardPass()

   /*
   * ##################################################
   *        Methods used entirely for training
   * ##################################################
   */

   /*
   * Calculates the changes in weights for 
   */
   void calculateDWForHidden(int testCase)
   {
      double fOfThetaj = 0.0;
      for (int j = 0; j < layerSize[HIDDEN_LAYER]; j++)
         fOfThetaj += activations[HIDDEN_LAYER][j] * weights[HIDDEN_LAYER][j][0];
      fOfThetaj = thresholdFunction(fOfThetaj);

      for (int j = 0; j < layerSize[HIDDEN_LAYER]; j++)
      {
         dweights[HIDDEN_LAYER][j][0] = -1.0 * (trueValues[testCase] - fOfThetaj) * 
                                        (fOfThetaj * (1.0 - fOfThetaj)) * 
                                        activations[HIDDEN_LAYER][j];
      }
   }

   /*
   * layer must be 0
   * testCase must be between 0 and numTests - 1
   */
   void calculateDWForInput(int testCase)
   {
      double fOfThetazero = 0.0;
      for (int j = 0; j < layerSize[HIDDEN_LAYER]; j++)
         fOfThetazero += activations[HIDDEN_LAYER][j] * weights[HIDDEN_LAYER][j][0];
      fOfThetazero = thresholdFunction(fOfThetazero);

      for (int j = 0; j < layerSize[HIDDEN_LAYER]; j++)
      {
         double fOfThetaj = 0.0;
         for (int k = 0; k < layerSize[INPUT_LAYER]; k++)
            fOfThetaj += activations[INPUT_LAYER][k] * weights[INPUT_LAYER][k][j];
         fOfThetaj = thresholdFunction(fOfThetaj);
         
         for (int k = 0; k < layerSize[INPUT_LAYER]; k++)
         {
            dweights[INPUT_LAYER][k][j] = -1.0 * activations[INPUT_LAYER][k] *
                                          (fOfThetaj * (1.0 - fOfThetaj)) *
                                          (trueValues[testCase] - fOfThetazero) * 
                                          (fOfThetazero * (1.0 - fOfThetazero)) * 
                                          weights[HIDDEN_LAYER][j][0];
         }
      } // for (int j = 0; j < layerSize[HIDDEN_LAYER]; j++)

      return;
   }

   void applyDW()
   {
      for (int i1 = 0; i1 < numLayers - 1; i1++)
         for (int i2 = 0; i2 < layerSize[i1]; i2++)
            for (int i3 = 0; i3 < layerSize[i1 + 1]; i3++)
            {
               weights[i1][i2][i3] += -1.0 * dweights[i1][i2][i3] * lambda;
               dweights[i1][i2][i3] = 0.0;
            }

      return;
   }

   void train()
   {
      double curError = 0.0;
      int iterations = 0;
      do
      {
         curError = 0.0;
         for (int i1 = 0; i1 < numTests; i1++)
         {
            curError += forwardPass(i1, 0);
            calculateDWForHidden(i1);
            calculateDWForInput(i1);
            applyDW();
         }

         iterations++;

         if (iterations % ITERATION_PRINTING_FREQUENCY == 0) 
         {
            cout << "Model has run " << iterations << " iterations ";
            cout << fixed << setprecision(PRECISION) << "and achieved an error of " << curError << endl;
         }
      } 
      while (iterations < maxIterations && curError > errorThreshold);

      cout << fixed << setprecision(PRECISION) << "Model terminated after " << iterations << " iterations with an error of " << curError << endl;
      if (iterations == maxIterations) 
         cout << "Model terminated due to reaching maximum number of iterations (" << maxIterations << ")." << endl;
      else 
         cout << "Model terminated due to reaching low enough error <=(" << errorThreshold << ")." << endl;
      
      for (int i1 = 0; i1 < numTests; i1++) forwardPass(i1, 1);
      
      return;
   } // void train()

   /*
   * ##################################################
   *        Methods used in testing
   * ##################################################
   */

   void test() 
   {
      double totError = 0.0;
      for (int i1 = 1; i1 <= numTests; i1++) totError += forwardPass(i1, 1);

      cout << fixed << setprecision(PRECISION) << "Average error is: " << totError / numTests << endl;

      return;
   }
}; // struct NeuralNetwork

int main(int argc, char* argv[]) 
{
   srand(time(NULL));         // sets the random seed to current time, so no two runs are the same
   NeuralNetwork network;
   network.train();

   return 0;
}