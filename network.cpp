/*
* Author: Andrew Liang
* Date of Creation: 27 September 2023
* Description: A simple A-B-C fully connected neural network
* The network has:
* an input layer with A nodes, 
* a hidden layer with B nodes, 
* and an output layer with C nodes
* Allows for the network to be trained and tested
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <iomanip>
#include <time.h>
using namespace std;

/*
* Constants
*/ 
const double EPSILON = 1e-9;                          // used to compare floating point numbers
const int PRECISION = 9;                              // number of digits after the decimal of printed out float values

/*
* sigmoid function, a type of activation function. formula: 1/(1+e^-val)
*/
double sigmoid(double val) 
{
   return 1.0 / (1.0 + exp(-val));
} // double sigmoid(double val)
/*
* derivative of the sigmoid function. σ(val) = σ(val) * (1 - σ(val))
*/
double sigmoidDerivative(double val) 
{
   double sigmoidVal = sigmoid(val);
   return sigmoidVal * (1.0 - sigmoidVal);
} // double sigmoidDerivative(double val)

/*
* A neural network.
* Currently, it is of the form A-B-1
* Can train and test
*/
struct NeuralNetwork 
{
   /*
   * Model constants
   */
   const string TARGET = "PARAM: ";                   // parameter indetifier
   const string CONFIG_FILE = "config.txt";           // location of configuration file
   const string WEIGHTS_FILE = "weights.txt";         // location of weights file

   const string TEST_FILE_PREFIX = "Tests/tc";        // prefix and suffix for test case file addresses (from home directory)
   const string TEST_FILE_SUFFIX = ".txt";            // tests are of the form "Tests/tc1.txt", where the test case number changes
   const string SAVED_MODELS_DIRECTORY = "Models/";   // directory of saved models

   const int INPUT_LAYER = 0;                         // input layer is the 0st layer
   const int HIDDEN_LAYER = 1;                        // hidden layer is the 1st layer
   const int OUTPUT_LAYER = 2;                        // output layer is the 2nd layer

   const int EXPECTED_NUM_LAYERS = 3;

   /*
   * Configuration parameters
   */
   int numActivationLayers;                           // number of layers, must be 3
   int* activationLayerSize;                          // activationLayerSize[i1] stores the size of layer i1 (output layer must be 1)
   bool isTraining;                                   // 1 if training, 0 if testing
   int numTests;                                      // number of test cases
   double wlb, wub;                                   // lower and upper bounds for weights
   double lambda;                                     // learning rate
   int thresholdFunctionType;                         // 0 if sigmoid (only option)
   double (*thresholdFunction) (double);              // actual threshold function
   double (*thresholdFunctionDerivative) (double);    // actual derivative of the threshold function
   int maxIterations;                                 // max iterations before stopping
   double errorThreshold;                             // min error to continue running
   int iterationPrintingFrequency;                    // frequency at which training information is printed (e.x. once per 5000 iterations)
   bool loadWeights;                                  // 1 if load, 0 if randomly generate
   bool saveModel;                                    // 1 if save model to file, 0 if no 
   bool saveModel;                                    // 1 if save model to file, 0 if no 

   bool initializedModelBody = false;                 // if activations/weights/inputs/answers have been initialized

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
   * values of the partial derivative with respect to each weight, dErrordWeight[i1][i2][i3] stores the partial derivative of w[i1][i2][i3]
   */
   double*** dErrordWeight;
   /* 
   * values of the inputs in each test, inputValues[i1][i2] stores the i2th input of the i1th test case
   */
   double** inputValues;
   /*
   * true values of each test
   */
   double** trueValues;

   /*
   * Intermediate training values
   * theta[i1][i2] = ∑ over i3 (activations[i1 - 1][i3] * weights[i1 - 1][i3][i2])
   * smallOmega[i] = (trueValues[i] - activations[i])
   * smallPsi[i] = smallOmega[i] * thresholdFunctionDerivative(theta[OUTPUT_LAYER][i])
   * bigOmega[j] = ∑ over I (smallOmega[I] * weight[HIDDEN_LAYER][j][I])
   */
   double** theta;
   double* smallOmega;
   double* smallPsi;
   double* bigOmega;

   /*
   * Configures model parameters and sets up model for training/running
   */
   NeuralNetwork() 
   {
      cout << "########################################" << endl;
      cout << "Attempting to configure model" << endl << endl;
      if (inputConfigParameters())
      {
         cout << "Model configured successfully!" << endl;
         cout << "########################################" << endl;
         printConfigParameters();
         cout << "########################################" << endl;

         allocateMemory();
         initializedModelBody = true;
         cout << "Memory allocated" << endl;

         if (loadWeights)
         {
            loadWeightsFromFile();
            cout << "Weights loaded" << endl;
         }
         else 
         {
            initializeArrays();
            cout << "Weights intialized" << endl;
         }

         loadTests();
         cout << "Tests loaded" << endl;

         cout << "########################################" << endl;

         if (isTraining) 
         {
            train();

            if (saveModel) 
            {
               saveWeightsToFile();
               cout << "Model weights saved!" << endl;
            }
         }
         else test();
      } // if (inputConfigParameters())
      else cout << "Model was not configured" << endl;

      return;
   } // NeuralNetwork()
   
   /*
   * destroys arrays created in the network
   */
   ~NeuralNetwork() 
   {
      delete activationLayerSize;

      if (initializedModelBody)
      {
         delete activations;
         delete weights;
         delete trueValues;
         delete inputValues;

         delete theta;
         delete smallOmega;

         if (isTraining)
         {
            delete dErrordWeight;
            delete smallPsi;
            delete bigOmega;
         }
      } // if (initializedModelBody)
      
      return;
   } // ~NeuralNetwork()

   /*
   * ##################################################
   *        Methods used to configure the model
   * ##################################################
   */

   /*
   * Inputs the integer contained in the first characters in the file before a space
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
   * Inputs the floating point number contained in the first characters in the file before a space
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
   * gets the next nonempty line that is not a comment (does not begin with '/' or '*')
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
   } // string getNextParameter(ifstream& fin)

   /*
   * reads in input configuration parameters from CONFIG_FILE
   * only parameters listed as member variables of the model will be considered
   */
   bool inputConfigParameters() 
   {
      bool modelValid = 1;
      ifstream fin(CONFIG_FILE);

      while (!fin.eof())
      {
         /*
         * Case matching
         * inputs the parameters and ensures that they are valid values
         */
         string parameter = getNextParameter(fin);
         
         if (parameter == "numActivationLayers")
         {
            numActivationLayers = getNextInteger(fin);
            if (numActivationLayers != EXPECTED_NUM_LAYERS) 
            {
               cout << "ERROR: This neural network does not support more or less than " 
                    << EXPECTED_NUM_LAYERS << " activation layers" << endl;
               modelValid = 0;
            }
         } // if (parameter == "numActivationLayers")

         if (parameter == "activationLayerSizes")
         {
            activationLayerSize = new int[numActivationLayers];
            int numInvalidLayers = 0;

            for (int i1 = 0; i1 < numActivationLayers; i1++) 
            {
               activationLayerSize[i1] = getNextInteger(fin);
               if (activationLayerSize[i1] <= 0)
               {
                  numInvalidLayers++;

                  if (numInvalidLayers == 1) // only print the following when the first invalid layer is discovered
                     cout << "ERROR: The following activation layers have invalid sizes (<= 0): ";
                  cout << activationLayerSize[i1] << " ";

                  modelValid = 0;
               }
            } // for (int i1 = 0; i1 < numActivationLayers; i1++) 
            if (numInvalidLayers > 0) cout << endl;
         } // if (parameter == "activationLayerSizes")

         if (parameter == "isTraining")
         {
            int num = getNextInteger(fin);
            if (num == 1) isTraining = true;
            else if (num == 0) isTraining = false;
            else 
            {
               cout << "ERROR: isTraining should be 0 (testing) or 1 (training)" << endl;
               modelValid = 0;
            }
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

         if (parameter == "thresholdFunctionType")
         {
            int thresholdFunctionType = getNextInteger(fin);
            if (thresholdFunctionType == 0)
            {
               thresholdFunction = sigmoid;
               thresholdFunctionDerivative = sigmoidDerivative;
            }
            else 
            {
               cout << "ERROR: Threshold function type is invalid, can only be 0 (sigmoid)" << endl;
               modelValid = 0;
            }
         } // if (parameter == "thresholdFunctionType")

         if (parameter == "loadWeights")
         {
            loadWeights = getNextInteger(fin);
            if (loadWeights != 0 && loadWeights != 1)
            {
               cout << "ERROR: loadWeights should be 0 (randomize) or 1 (load)" << endl;
               modelValid = 0;
            }
         } // if (parameter == "loadWeights")

         if (parameter == "saveModel")
         {
            saveModel = getNextInteger(fin);
            if (saveModel != 0 && saveModel != 1)
            {
               cout << "ERROR: saveModel should be 0 (don't save) or 1 (save)" << endl;
               modelValid = 0;
            }
         } // if (parameter == "saveModel")

         if (isTraining)
         {
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

            if (parameter == "iterationPrintingFrequency")
            {
               iterationPrintingFrequency = getNextInteger(fin);
               if (iterationPrintingFrequency <= 0)
               {
                  cout << "ERROR: iteration printing frequency must be > 0." << endl;
                  modelValid = 0;
               }
            } // if (parameter == "iterationPrintingFrequency")
         } // if (isTraining)
      } // while (!fin.eof())   

      return modelValid;
   } // bool setConfigParameters() 

   /*
   * Prints the configuration of the model
   */
   void printConfigParameters() 
   {
      cout << "Printing configuration paramenters:" << endl << endl;

      cout << "Number of activation layers: " << numActivationLayers << endl;

      cout << "Size of each activation layer: ";
      for (int i1 = 0; i1 < numActivationLayers; i1++) cout << activationLayerSize[i1] << " ";
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
      
      return;
   } // void printConfigParameters()

   /*
   * Initializes arrays based on the following dimensions:
   * a: numActivationLayers, activationLayerSize of current layer
   * weights and dErrordWeight: numActivationLayers - 1, activationLayerSize of current layer, activationLayerSize of next layer
   * inputvalues: numTests, activationLayerSize of input layer
   * truevalues: numTests
   */
   void allocateMemory() 
   {
      activations = new double*[numActivationLayers];
      for (int i1 = 0; i1 < numActivationLayers; i1++) 
         activations[i1] = new double[activationLayerSize[i1]];

      weights = new double**[numActivationLayers - 1];
      for (int i1 = 0; i1 < numActivationLayers - 1; i1++) 
      {
         weights[i1] = new double*[activationLayerSize[i1]];
         for (int i2 = 0; i2 < activationLayerSize[i1]; i2++) 
            weights[i1][i2] = new double[activationLayerSize[i1 + 1]];
      } // for (int i1 = 0; i1 < numActivationLayers - 1; i1++) 

      inputValues = new double*[numTests];
      for (int i1 = 0; i1 < numTests; i1++) 
         inputValues[i1] = new double[activationLayerSize[INPUT_LAYER]];

      trueValues = new double*[numTests];
      for (int i1 = 0; i1 < numTests; i1++) 
         trueValues[i1] = new double[activationLayerSize[OUTPUT_LAYER]];
      
      theta = new double*[numActivationLayers];
      for (int i1 = 0; i1 < numActivationLayers; i1++) 
         theta[i1] = new double[activationLayerSize[i1]];
      smallOmega = new double[activationLayerSize[OUTPUT_LAYER]];

      if (isTraining)
      {
         
         dErrordWeight = new double**[numActivationLayers - 1];
         for (int i1 = 0; i1 < numActivationLayers - 1; i1++) 
         {
            dErrordWeight[i1] = new double*[activationLayerSize[i1]];
            for (int i2 = 0; i2 < activationLayerSize[i1]; i2++) 
               dErrordWeight[i1][i2] = new double[activationLayerSize[i1 + 1]];
         } // for (int i1 = 0; i1 < numActivationLayers - 1; i1++) 

         smallPsi = new double[activationLayerSize[OUTPUT_LAYER]];
         bigOmega = new double[activationLayerSize[HIDDEN_LAYER]];
      } // if (isTraining)

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
   } // double genRand()

   /*
   * Initializes all weights in the range [wlb, wub]
   */
   void initializeArrays() 
   {
      for (int i1 = 0; i1 < numActivationLayers - 1; i1++) 
         for (int i2 = 0; i2 < activationLayerSize[i1]; i2++) 
            for (int i3 = 0; i3 < activationLayerSize[i1 + 1]; i3++) 
               weights[i1][i2][i3] = genRand();

      return;
   } // void initializeArrays()

   void loadWeightsFromFile()
   {
      ifstream fin(WEIGHTS_FILE);
      for (int i1 = 0; (i1 < numActivationLayers - 1); i1++)
         for (int i2 = 0; i2 < activationLayerSize[i1]; i2++)
            for (int i3 = 0; i3 < activationLayerSize[i1 + 1]; i3++)
               fin >> weights[i1][i2][i3];
      
      return;
   } // void loadWeightsFromFile();

   void saveWeightsToFile()
   {
      time_t rawtime;
      time(&rawtime);
      string fname = SAVED_MODELS_DIRECTORY;
      for (int i1 = 0; i1 < numActivationLayers; i1++)
         fname += to_string(activationLayerSize[i1]) + "-";
      fname += to_string(rawtime);

      ofstream fout(fname);
      for (int i1 = 0; i1 < numActivationLayers - 1; i1++)
         for (int i2 = 0; i2 < activationLayerSize[i1]; i2++)
            for (int i3 = 0; i3 < activationLayerSize[i1 + 1]; i3++)
               fout << weights[i1][i2][i3] << " ";
      
      return;
   } // void saveWeightsToFile()

   void printWeights() 
   {
      cout << "Weights:" << endl;
      for (int i1 = 0; i1 < numActivationLayers - 1; i1++) 
      {
         cout << "From layer " << i1 << ":" << endl;
         for (int i2 = 0; i2 < activationLayerSize[i1]; i2++) 
         {
            cout << "From activation " << i2 << ": ";
            for (int i3 = 0; i3 < activationLayerSize[i1 + 1]; i3++) 
               cout << fixed << setprecision(PRECISION) << weights[i1][i2][i3] << " ";
            cout << endl;
         }
      } // for (int i1 = 0; i1 < numActivationLayers - 1; i1++) 

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
      cout << "Truth table:" << endl;
      for (int i1 = 0; i1 < numTests; i1++)
      {
         ifstream fin(TEST_FILE_PREFIX + to_string(i1) + TEST_FILE_SUFFIX);

         for (int i2 = 0; i2 < activationLayerSize[INPUT_LAYER]; i2++)
         {
            fin >> inputValues[i1][i2];
            cout << inputValues[i1][i2] << " ";
         }
         cout << ": ";

         for (int i2 = 0; i2 < activationLayerSize[OUTPUT_LAYER]; i2++)
         {
            fin >> trueValues[i1][i2];
            cout << trueValues[i1][i2] << " ";
         }
         cout << endl;
      } // for (int i1 = 0; i1 < numTests; i1++)

      return;
   } // void loadTests()

   /*
   * calculates the error according to the function error = 1/2∑(t-f)^2
   */
   double error(int testCase) 
   {
      double err = 0.0;
      for (int i = 0; i < activationLayerSize[OUTPUT_LAYER]; i++)
         err += 0.5 * smallOmega[i] * smallOmega[i];

      return err;
   } // double error(int testCase)

   /*
   * Passes though the values in the input through the model
   * if printResults is true, the predicted value, true value, and error are given for the test
   * calculates theta and smallOmega values
   */
   void forwardPass(int testCase) 
   {
      for (int i1 = 0; i1 < activationLayerSize[INPUT_LAYER]; i1++) 
         activations[INPUT_LAYER][i1] = inputValues[testCase][i1];

      for (int i1 = 0; i1 < numActivationLayers - 1; i1++) 
         for (int i3 = 0; i3 < activationLayerSize[i1 + 1]; i3++) 
         {
            for (int i2 = 0; i2 < activationLayerSize[i1]; i2++) 
               theta[i1 + 1][i3] += activations[i1][i2] * weights[i1][i2][i3];
            activations[i1 + 1][i3] = thresholdFunction(theta[i1 + 1][i3]);
         }
      
      for (int i = 0; i < activationLayerSize[OUTPUT_LAYER]; i++)
         smallOmega[i] = trueValues[testCase][i] - activations[OUTPUT_LAYER][i];

      return;
   } // void forwardPass()

   void printResults(int testCase)
   {
      cout << "Test case " << testCase << ":" << endl;
      for (int i = 0; i < activationLayerSize[OUTPUT_LAYER]; i++) 
      {
         cout << "Output " << i << ": ";
         cout << fixed << setprecision(PRECISION) << "expected " << trueValues[testCase][i]
                        << ", received " << activations[OUTPUT_LAYER][i] << endl;
      }
      
      cout << fixed << setprecision(PRECISION) << "error is: " << error(testCase) << endl;
      
      return;
   } // void printResults(int testCase)

   double calculateAverageError(bool wantResults)
   {
      double avgError = 0.0;
      for (int i1 = 0; i1 < numTests; i1++)
      {
         forwardPass(i1);
         avgError += error(i1);
         if (wantResults) printResults(i1);
         resetThetas();
      }
      avgError /= (double) numTests;

      if (wantResults) cout << "Average error is " << avgError << endl;

      return avgError;
   } // void calculateAverageError(int testCase)
   /*
   * ##################################################
   *        Methods used entirely for training
   * ##################################################
   */

   void calculateSmallPsiAndBigOmega()
   {
      for (int i = 0; i < activationLayerSize[OUTPUT_LAYER]; i++)
         smallPsi[i] = smallOmega[i] * thresholdFunctionDerivative(theta[OUTPUT_LAYER][i]);
      
      for (int j = 0; j < activationLayerSize[HIDDEN_LAYER]; j++)
         for (int i = 0; i < activationLayerSize[OUTPUT_LAYER]; i++)
            bigOmega[j] += smallPsi[i] * weights[HIDDEN_LAYER][j][i];
      
      return;
   } // void calculateSmallPsiAndBigOmega()

   void resetBigOmega()
   {
      for (int j = 0; j < activationLayerSize[HIDDEN_LAYER]; j++)
         bigOmega[j] = 0.0;

      return;
   } // void resetBigOmega()

   void resetThetas()
   {  
      for (int i1 = 0; i1 < numActivationLayers; i1++)
         for (int i2 = 0; i2 < activationLayerSize[i1]; i2++)
            theta[i1][i2] = 0.0;

      return;
   } // void resetThetas()

   /*
   * Calculates the partial derivative of each weight from the hidden layer to the output layer
   * testCase must be between 0 and numTests - 1
   */
   void calculatePartialDerivativeForHidden(int testCase)
   {
      for (int i = 0; i < activationLayerSize[OUTPUT_LAYER]; i++)
         for (int j = 0; j < activationLayerSize[HIDDEN_LAYER]; j++)
            dErrordWeight[HIDDEN_LAYER][j][i] = -smallPsi[i] * activations[HIDDEN_LAYER][j];

      return;
   } // void calculatePartialDerivativeForHidden(int testCase)

   /*
   * Calculates the partial derivative of each weight from the input layer to the hidden layer
   * testCase must be between 0 and numTests - 1
   */
   void calculatePartialDerivativeForInput(int testCase)
   {
      for (int i = 0; i < activationLayerSize[OUTPUT_LAYER]; i++)
         for (int j = 0; j < activationLayerSize[HIDDEN_LAYER]; j++)
            for (int k = 0; k < activationLayerSize[INPUT_LAYER]; k++)
               dErrordWeight[INPUT_LAYER][k][j] = -activations[INPUT_LAYER][k] *
                                                  thresholdFunctionDerivative(theta[HIDDEN_LAYER][j]) *
                                                  bigOmega[j];

      return;
   } // void calculatePartialDerivativeForInput(int testCase)

   /*
   * applies the changes to each weight
   * ∆w = -lambda * dErrordWeight[w]
   */
   void applyDW()
   {
      for (int i1 = 0; i1 < numActivationLayers - 1; i1++)
         for (int i2 = 0; i2 < activationLayerSize[i1]; i2++)
            for (int i3 = 0; i3 < activationLayerSize[i1 + 1]; i3++)
            {
               weights[i1][i2][i3] += -dErrordWeight[i1][i2][i3] * lambda;
               dErrordWeight[i1][i2][i3] = 0.0;
            }

      return;
   } // void applyDW()

   /*
   * trains the model by running gradient descent until one of two conditions
   * 1. number of iterations exceeds maxIterations
   * 2. average error over all testcases is below errorThreshold
   */
   void train()
   {
      double avgError = 0.0;
      int iterations = 0;
      do 
      {
         for (int i1 = 0; i1 < numTests; i1++)
         {
            forwardPass(i1);
            calculateSmallPsiAndBigOmega();

            calculatePartialDerivativeForHidden(i1);
            calculatePartialDerivativeForInput(i1);

            applyDW();
            resetBigOmega();
            resetThetas();
         } // for (int i1 = 0; i1 < numTests; i1++)

         iterations++;

         avgError = calculateAverageError(0);

         if (iterations % iterationPrintingFrequency == 0) 
         {
            cout << "Model has run " << iterations << " iterations ";
            cout << fixed << setprecision(PRECISION) << "and achieved an error of " << avgError << endl;
         }
      } // do
      while (iterations < maxIterations && avgError - errorThreshold > EPSILON);

      cout << fixed << setprecision(PRECISION) << "Model terminated after " << iterations 
           << " iterations with an error of " << avgError << endl;
      if (iterations == maxIterations) 
         cout << "Model terminated due to reaching maximum number of iterations (" << maxIterations << ")." << endl;
      else 
         cout << fixed << setprecision(PRECISION) 
              << "Model terminated due to reaching low enough error (<=" << errorThreshold << ")." << endl;
      
      calculateAverageError(1);
      
      return;
   } // void train()

   /*
   * ##################################################
   *        Methods used in testing
   * ##################################################
   */

   /*
   * tests the model by reporting error after one forward pass of the model
   */
   void test() 
   {
      calculateAverageError(true);

      return;
   } // void test()
}; // struct NeuralNetwork

int main(int argc, char* argv[]) 
{
   srand(time(NULL));         // sets the random seed to current time, so no two runs are the same
   NeuralNetwork network;

   return 0;
}