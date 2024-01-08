#pragma GCC optimize("O3,unroll-loops")
/*
* Author: Andrew Liang
* Date of Creation: 17 November 2023
* Description: A N-layer fully connected neural network
* Allows for the network to be trained and tested
* Utilizes backpropagation

*   Table of contents
*       Outside of Neural Network
*           double sigmoid()
*           double sigmoidDerivative()
*           double tanhDerivative()
*   Within Neural Network
*       NeuralNetwork(const& string configuration)
*       ~NeuralNetwork()
*       int getNextInteger(ifstream& fin)
*       double getNextDouble(ifstream& fin)
*       string getNextParameter(ifstream& fin)
*       bool inputConfigParameters() 
*       void printConfigParameters() 
*       void allocateMemory() 
*       double genRand() 
*       void initializeArrays() 
*       bool loadWeightsFromFile()
*       void saveWeightsToFile()
*       void printModelWeights() 
*       void printTruthTable()
*       void loadTests()
*       double error(int testCase) 
*       void forwardPassEvaluate(int testCase) 
*       void forwardPassTrain(int testCase) 
*       void printResults(int testCase)
*       double calculateAverageError(bool wantResults)
*       void learn()
*       void train()
*       void test() 
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <unordered_map>
#include <cassert>
using namespace std;

/*
* Constants
*/ 
const double EPSILON = 1e-24;                          // used to compare floating point numbers
const double INF = 1e9;
const int PRECISION = 9;                              // number of digits after the decimal for high precision
const string DEFAULT_CONFIG_FILE = "config.txt";       // default configuration file

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
* derivative of the tanh function. tanh'(val) = 1 - 2 * tanh(val)
*/
double tanhDerivative(double val) 
{
    double tanhVal = tanh(val);
    return 1.0 - tanhVal * tanhVal;
} // double sigmoidDerivative(double val)
double relu(double val) 
{
    if (val > 0.0) return val;
    else return 0.1 * val;
}
double reluDerivative(double val) 
{
    return (val > 0.0 ? 1.0 : 0.1);
} // double sigmoidDerivative(double val)

/*
* A N-layer neural network.
* Can train and test
*/
struct NeuralNetwork 
{
    /*
    * Model constants
    */
    const string TARGET = "PARAM: ";                   // parameter indentifier
    const string TEST_FILE_PREFIX = "tc";              // prefix and suffix for test case file addresses (from home directory)
    const string TEST_FILE_SUFFIX = ".txt";            // tests are of the form "Tests/tc1.txt", where the test case number changes
    unordered_map<string, int> PARAMETERS = {   {"numActivationLayers", 0},
                                                {"activationLayerSizes", 1},
                                                {"isTraining", 2},
                                                {"numClasses", 3},
                                                {"thresholdFunctionType", 4},
                                                {"loadWeights", 5},
                                                {"saveModel", 6},
                                                {"wb", 7},
                                                {"lambda", 8},
                                                {"maxIterations", 9},
                                                {"errorThreshold", 10},
                                                {"iterationPrintingFrequency", 11},
                                                {"printWeights", 12},
                                                {"problem", 13},
                                                {"weightsFile", 14},
                                                {"willPrintTruthTable", 15},
                                                {"numInEachClass", 20},};

    const int INPUT_LAYER = 0;                         // input layer is the 0th layer
    const int FIRST_HIDDEN_LAYER = 1;                  // first hidden layer is the 1st layer
    int lastHiddenLayer;
    int outputLayer;

    /*
    * Configuration parameters
    */
    int numActivationLayers;                            // number of layers
    int* activationLayerSize;                           // activationLayerSize[i1] stores the size of layer i1 (output layer must be 1)
    bool isTraining;                                    // 1 if training, 0 if testing
    int numTests;                                       // number of test cases
    int numClasses;
    int* numInEachClass;
    double wlb, wub;                                    // lower and upper bounds for weights
    double (*thresholdFunction) (double);               // actual threshold function
    double (*thresholdFunctionDerivative) (double);     // actual derivative of the threshold function
    int maxIterations;                                  // max iterations before stopping
    double errorThreshold;                              // min error to continue running
    int iterationPrintingFrequency;                     // frequency at which training information is printed (e.x. once per 5000 iterations)
    bool loadWeights;                                   // 1 if load, 0 if randomly generate
    bool saveModel;                                     // 1 if save model to file, 0 if no
    string configFile;                                  // location of configuration file
    string weightsFile;                                 // location of weights file
    bool printWeights;                                  // whether weights are to be printed
    string problem;                                     // the directory where tests are
    bool willPrintTruthTable;                           // whether to print the truth table

    int thresholdFunctionType;                          // 0 if sigmoid
                                                        // 1 if tanh
    double lambda;                                      // learning rate

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
    * values of the inputs in each test, inputValues[i1][i2] stores the i2th input of the i1th test case
    */
    double** inputValues;
    /*
    * true values of each test
    */
    double** trueValues;

    /*
    * Intermediate training values
    */
    double** theta;
    double** psi;
    double** omega;

    /*
    * Configures model parameters and sets up model for training/running
    */
    NeuralNetwork(const string& configuration) 
    {
        configFile = configuration;
        cout << "########################################" << endl;
        cout << "Attempting to configure model from file '" << configFile << "'" << endl << endl;
        if (inputConfigParameters())
        {
            cout << "Model configured successfully!" << endl;
            allocateMemory();
            cout << "Memory allocated" << endl;
            initializedModelBody = true;

            bool canRun = true;
            if (loadWeights)
            {
                if (loadWeightsFromFile()) cout << "Weights loaded successfully!" << endl;
                else 
                {
                    cout << "Model configuration failed, architectures do not match" << endl;
                    canRun = false;
                }
            }
            else 
            {
                initializeArrays();
                cout << "Weights intialized!" << endl;
            }

            if (canRun)
            {
                printConfigParameters();
                loadTests();
                cout << "Tests loaded" << endl;

                cout << fixed << setprecision(PRECISION);

                if (isTraining) 
                {
                    auto start = chrono::system_clock::now();
                    train();
                    auto end = chrono::system_clock::now();
                    chrono::duration<double> elapsed_time = end - start;
                    cout << "Took " << elapsed_time.count() << "s to train" << endl;

                    if (saveModel) 
                    {
                        saveWeightsToFile();
                        cout << "Model weights saved to file " << weightsFile << "!" << endl;
                    }
                } // if (isTraining)
                else 
                    {
                    auto start = chrono::system_clock::now();
                    test();
                    auto end = chrono::system_clock::now();
                    chrono::duration<double> elapsed_time = end - start;
                    cout << "Took " << elapsed_time.count() << "s to test" << endl;
                }
            } // if (canRun)
        } // if (inputConfigParameters())
        else cout << "Model was not configured" << endl;

        return;
    } // NeuralNetwork()
    
    /*
    * destroys arrays created in the network
    */
    ~NeuralNetwork() 
    {
        delete[] activationLayerSize;

        if (initializedModelBody)
        {
            delete[] activations;
            delete[] weights;
            delete[] trueValues;
            delete[] inputValues;

            if (isTraining) 
            {
                delete[] psi;
                delete[] omega;
                delete[] theta;
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
    * gets the next parameter (line will start with "PARAM: ")
    */
    string getNextParameter(ifstream& fin)
    {
        string line;
        do getline(fin, line);
        while (!fin.eof() && (line.length() <= TARGET.length() || line.substr(0, TARGET.length()) != TARGET));
        
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
    * reads in input configuration parameters from configFile
    * only parameters listed as member variables of the model will be considered
    */
    bool inputConfigParameters() 
    {
        bool modelValid = 1;
        ifstream fin(configFile);

        while (!fin.eof())
        {
            string param = getNextParameter(fin);
            int val = -1;
            if (PARAMETERS.count(param)) val = PARAMETERS[param];

            switch (val)
            {
                case 0:     // numActivationLayers
                {
                    numActivationLayers = getNextInteger(fin);
                    if (numActivationLayers <= 0) 
                    {
                        cout << "ERROR: This neural network must have a positive number of layers" << endl;
                        modelValid = 0;
                    }
                    outputLayer = numActivationLayers - 1;
                    lastHiddenLayer = outputLayer - 1;
                    break;
                }

                case 1:     // activationLayerSizes  
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
                            {
                                cout << "ERROR: The following activation layers have invalid sizes (<= 0): ";
                                cout << activationLayerSize[i1] << " ";
                            }

                            modelValid = 0;
                        }
                    } // for (int i1 = 0; i1 < numActivationLayers; i1++) 
                    if (numInvalidLayers > 0) cout << endl;  
                    break;
                }

                case 2:     // isTraining  
                {
                    isTraining = getNextInteger(fin);
                    if (isTraining != 0 && isTraining != 1)
                    {
                        cout << "ERROR: isTraining should be 0 (testing) or 1 (training)" << endl;
                        modelValid = 0;
                    }
                    break;
                }

                case 3:     // numClasses
                {
                    numClasses = getNextInteger(fin);
                    if (numClasses <= 0)
                    {
                        cout << "ERROR: Invalid number of classes (<= 0)" << endl;
                        modelValid = 0;
                    }

                    numInEachClass = new int[numClasses];
                    break;
                }
                
                case 4:     // thresholdFunctionType
                {
                    thresholdFunctionType = getNextInteger(fin);
                    if (thresholdFunctionType == 0)
                    {
                        thresholdFunction = sigmoid;
                        thresholdFunctionDerivative = sigmoidDerivative;
                    }
                    else if (thresholdFunctionType == 1)
                    {
                        thresholdFunction = tanh;
                        thresholdFunctionDerivative = tanhDerivative;
                    }
                    else if (thresholdFunctionType == 2)
                    {
                        thresholdFunction = relu;
                        thresholdFunctionDerivative = reluDerivative;
                    }
                    else 
                    {
                        cout << "ERROR: Threshold function type is invalid, can only be 0 (sigmoid) or 1 (tanh) opr 2 (relu)" << endl;
                        modelValid = 0;
                    }
                    break;
                }

                case 5:     // loadWeights
                {
                    loadWeights = getNextInteger(fin);
                    if (loadWeights != 0 && loadWeights != 1)
                    {
                        cout << "ERROR: loadWeights should be 0 (randomize) or 1 (load)" << endl;
                        modelValid = 0;
                    }
                    break;
                }

                case 6:     // saveModel
                {
                    saveModel = getNextInteger(fin);
                    if (saveModel != 0 && saveModel != 1)
                    {
                        cout << "ERROR: saveModel should be 0 (don't save) or 1 (save)" << endl;
                        modelValid = 0;
                    }
                    break;
                }

                case 7:     // wb
                {
                    wlb = getNextDouble(fin);
                    wub = getNextDouble(fin);
                    if (wlb - wub > EPSILON)
                    {
                        cout << "ERROR: Lower bound for weights must be <= upper bound for weights" << endl;
                        modelValid = 0;
                    }
                    break;
                }

                case 8:     // lambda
                {
                    lambda = getNextDouble(fin);
                    if (lambda < EPSILON)
                    {
                        cout << "ERROR: lambda must be positive" << endl;
                        modelValid = 0;
                    }
                    break;
                }

                case 9:     // maxIterations
                {
                    maxIterations = getNextInteger(fin);
                    if (maxIterations <= 0)
                    {
                        cout << "ERROR: Model must run for positive number of iterations." << endl;
                        modelValid = 0;
                    }
                    break;
                }

                case 10:    // errorThreshold
                {
                    errorThreshold = getNextDouble(fin);
                    if (errorThreshold < 0)
                    {
                        cout << "ERROR: Must have positive error threshold." << endl;
                        modelValid = 0;
                    }
                    break;
                }

                case 11:    // iterationPrintingFrequency
                {
                    iterationPrintingFrequency = getNextInteger(fin);
                    if (iterationPrintingFrequency <= 0)
                    {
                        cout << "ERROR: iteration printing frequency must be > 0." << endl;
                        modelValid = 0;
                    }
                    break;
                }

                case 12:    // printWeights
                {
                    printWeights = getNextInteger(fin);
                    if (printWeights != 0 && printWeights != 1)
                    {
                        cout << "ERROR: printWeights should be 0 (don't print) or 1 (print)" << endl;
                        printWeights = 0;
                    }
                    break;
                }

                case 13:    // problem
                {
                    getline(fin, problem);
                    break;
                }
                
                case 14:    // weights file
                {
                    getline(fin, weightsFile);
                    break;
                }
                
                case 15:    // printTruthTable
                {
                    willPrintTruthTable = getNextInteger(fin);
                    if (willPrintTruthTable != 0 && willPrintTruthTable != 1)
                    {
                        cout << "ERROR: willPrintTruthTable should be 0 (dont print) or 1 (print)" << endl;
                        modelValid = 0;
                    }
                    break;
                }
                
                case 20:    // numInEachClass
                {
                    numTests = 0;
                    for (int i1 = 0; i1 < numClasses; i1++)
                    {
                        numInEachClass[i1] = getNextInteger(fin);
                        if (numInEachClass[i1] <= 0)
                        {
                            cout << "ERROR: Class " << i1 << " has an invalid number of classes (<= 0)" << endl;
                            modelValid = 0;
                        }

                        numTests += numInEachClass[i1];
                    }
                    break;
                }

                default:
                    cout << "ERROR: " << param << " is not a parameter" << endl;
            } // switch (val)
        } // while (!fin.eof())

        return modelValid;
    } // bool inputConfigParameters() 

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

        if (printWeights) printModelWeights();

        cout << "The threshold function used will be: ";
        if (thresholdFunctionType == 0) cout << "sigmoid" << endl;
        if (thresholdFunctionType == 1) cout << "tanh" << endl;
        if (thresholdFunctionType == 2) cout << "relu" << endl;

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
        else cout << "Weights will be read from " << weightsFile << endl;
        cout << endl;
        
        return;
    } // void printConfigParameters()

    /*
    * Initializes arrays based on the following dimensions:
    * a: numActivationLayers, activationLayerSize of current layer
    * weights: numActivationLayers - 1, activationLayerSize of current layer, activationLayerSize of next layer
    * inputvalues: numTests, activationLayerSize of input layer
    * truevalues: numTests
    * 
    * theta, psi, and omega: numActivationLayers, activationLayerSize of current layer
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
        }

        inputValues = new double*[numTests];
        for (int testCase = 0; testCase < numTests; testCase++) 
            inputValues[testCase] = new double[activationLayerSize[INPUT_LAYER]];

        trueValues = new double*[numTests];
        for (int testCase = 0; testCase < numTests; testCase++) 
            trueValues[testCase] = new double[activationLayerSize[outputLayer]];
                
        if (isTraining) 
        {
            theta = new double*[numActivationLayers];
            for (int i1 = 0; i1 < numActivationLayers; i1++) 
                theta[i1] = new double[activationLayerSize[i1]];
            
            psi = new double*[numActivationLayers];
            for (int i1 = 0; i1 < numActivationLayers; i1++) 
                psi[i1] = new double[activationLayerSize[i1]];
            
            omega = new double*[numActivationLayers];
            for (int i1 = 0; i1 < numActivationLayers; i1++) 
                omega[i1] = new double[activationLayerSize[i1]];
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
        for (int i1 = INPUT_LAYER; i1 < outputLayer; i1++) 
            for (int i2 = 0; i2 < activationLayerSize[i1]; i2++) 
                for (int i3 = 0; i3 < activationLayerSize[i1 + 1]; i3++) 
                    weights[i1][i2][i3] = genRand();

        return;
    } // void initializeArrays()

    bool loadWeightsFromFile()
    {
        ifstream fin(weightsFile);

        bool isCorrectNetwork = true;
        for (int i1 = 0; i1 < numActivationLayers; i1++)
        {
            int layerSz;
            fin >> layerSz;
            if (layerSz != activationLayerSize[i1]) isCorrectNetwork = false;
        }
        
        if (isCorrectNetwork)
        {
            for (int i1 = 0; i1 < numActivationLayers - 1; i1++)
                for (int i2 = 0; i2 < activationLayerSize[i1]; i2++)
                    for (int i3 = 0; i3 < activationLayerSize[i1 + 1]; i3++)
                        fin >> weights[i1][i2][i3];
        }
        
        return isCorrectNetwork;
    } // bool loadWeightsFromFile();

    void saveWeightsToFile()
    {
        time_t rawtime;
        time(&rawtime);

        ofstream fout(weightsFile);
        fout << fixed << setprecision(PRECISION);
        for (int i1 = 0; i1 < numActivationLayers; i1++) fout << activationLayerSize[i1] << " ";
        fout << endl;

        for (int i1 = 0; i1 < numActivationLayers - 1; i1++)
            for (int i2 = 0; i2 < activationLayerSize[i1]; i2++)
                for (int i3 = 0; i3 < activationLayerSize[i1 + 1]; i3++)
                    fout << weights[i1][i2][i3] << " ";
        
        return;
    } // void saveWeightsToFile()

    void printModelWeights() 
    {
        // cout << "Weights:" << endl;
        // for (int i1 = 0; i1 < numActivationLayers - 1; i1++) 
        // {
        //     cout << "From activation layer " << i1 << " to " << i1 + 1 << endl;
        //     for (int i2 = 0; i2 < activationLayerSize[i1]; i2++) 
        //     {
        //         for (int i3 = 0; i3 < activationLayerSize[i1 + 1]; i3++) 
        //             cout << weights[i1][i2][i3] << "\t";
        //         cout << endl;
        //     }
        // } // for (int i1 = 0; i1 < numActivationLayers - 1; i1++) 

        return;
    } // void printModelWeights()

    /*
    * ##################################################
    *        Methods used in training and testing
    * ##################################################
    */

    /*
    * prints the truth table
    */
    void printTruthTable() 
    {
        cout << "Truth table:" << endl;
        for (int testCase = 0; testCase < numTests; testCase++)
        {
            for (int m = 0; m < activationLayerSize[INPUT_LAYER]; m++)
                cout << inputValues[testCase][m] << " ";
            
            cout << "| ";

            for (int i = 0; i < activationLayerSize[outputLayer]; i++)
                cout << trueValues[testCase][i] << " ";
            
            cout << endl;
        } // for (int testCase = 0; testCase < numTests; testCase++)

        return;
    } // void printTruthTable

    /*
    * stores values in each test case into inputValues and trueValues
    */
    void loadTests()
    {
        int tc = 0;
        for (int number = 0; number < numClasses; number++)
        {
            for (int testCase = 0; testCase < numInEachClass[number]; testCase++)
            {
                string infileName = problem;
                infileName += to_string(number + 1) + "/" + to_string(number + 1) + "." + to_string(testCase + 1) + ".txt";
                ifstream fin(infileName);

                for (int m = 0; m < activationLayerSize[INPUT_LAYER]; m++)
                    fin >> inputValues[tc][m];

                for (int i = 0; i < activationLayerSize[outputLayer]; i++)
                    if (i == number) trueValues[tc][i] = 1.0;
                    else trueValues[tc][i] = 0.0;
                
                tc++;
            }
        } // for (int testCase = 0; testCase < numTests; testCase++)

        return;
    } // void loadTests()

    /*
    * calculates cross-entropy loss
    */
    double error(int testCase) 
    {
        double err = 0.0;
        for (int i = 0; i < activationLayerSize[outputLayer]; i++) 
            err += 0.5 * (trueValues[testCase][i] - activations[outputLayer][i])
                       * (trueValues[testCase][i] - activations[outputLayer][i]);
        
        return err;
    } // double error(int testCase)

    /*
    * Passes though the values in the input through the model
    */
    void forwardPassEvaluate(int testCase) 
    {
        for (int b = 0; b < activationLayerSize[INPUT_LAYER]; b++)
            activations[INPUT_LAYER][b] = inputValues[testCase][b];

        double theta;
        for (int a = FIRST_HIDDEN_LAYER; a <= outputLayer; a++) 
        {
            for (int b = 0; b < activationLayerSize[a]; b++)
            {
                theta = 0.0;
                for (int c = 0; c < activationLayerSize[a - 1]; c++)
                    theta += activations[a - 1][c] * weights[a - 1][c][b] * 0.5;
                
                activations[a][b] = thresholdFunction(theta);
            }
        } // for (int a = FIRST_HIDDEN_LAYER; a <= lastHiddenLayer; a++) 

        return;
    } // void forwardPassEvaluate(int testCase)
    
    /*
    * Passes though the values in the input through the model
    * Also calculates psi values
    * Needs to be separate function as not to have conditionals within training
    */
    void forwardPassTrain(int testCase) 
    {
        for (int b = 0; b < activationLayerSize[INPUT_LAYER]; b++)
            activations[INPUT_LAYER][b] = inputValues[testCase][b];

        for (int a = FIRST_HIDDEN_LAYER; a < outputLayer; a++) 
        {
            for (int b = 0; b < activationLayerSize[a]; b++)
            {
                theta[a][b] = 0.0;
                for (int c = 0; c < activationLayerSize[a - 1]; c++) 
                    theta[a][b] += activations[a - 1][c] * weights[a - 1][c][b];
                
                activations[a][b] = thresholdFunction(theta[a][b]);
            }
        } // for (int a = FIRST_HIDDEN_LAYER; a <= lastHiddenLayer; a++) 

        int a = outputLayer;
        for (int b = 0; b < activationLayerSize[a]; b++)
        {
            theta[a][b] = 0.0;
            for (int c = 0; c < activationLayerSize[a - 1]; c++) 
                theta[a][b] += activations[a - 1][c] * weights[a - 1][c][b];
            
            activations[a][b] = thresholdFunction(theta[a][b]);

            psi[outputLayer][a] = (trueValues[testCase][b] - activations[outputLayer][b]) *
                                  thresholdFunctionDerivative(theta[outputLayer][b]);
        }

        return;
    } // void forwardPassTrain(int testCAse)

    void printResults(int testCase)
    {
        for (int i = 0; i < activationLayerSize[outputLayer]; i++) 
            cout << activations[outputLayer][i] << " ";

        cout << "| ";

        for (int i = 0; i < activationLayerSize[outputLayer]; i++) 
            cout << trueValues[testCase][i] << " ";
        cout << endl;
        
        return;
    } // void printResults(int testCase)

    double calculateAverageError(bool wantResults)
    {
        if (wantResults) cout << "Final truth table:" << endl;

        double avgError = 0.0;
        int tc = 0;
        for (int c = 0; c < numClasses; c++)
        {
            for (int n = 0; n < numInEachClass[c]; n++)
            {
                forwardPassEvaluate(tc);
                avgError += error(tc);
                if (wantResults) printResults(tc);

                tc++;
            }
            if (wantResults) cout << endl;
        }
        avgError /= (double) numTests;

        if (wantResults) cout << "Average error is " << avgError << endl;

        return avgError;
    } // double calculateAverageError(bool wantResults)

    /*
    * ##################################################
    *        Methods used entirely for training
    * ##################################################
    */

    /*
    * Modifies the weights
    */
    void learn()
    {
        for (int a = lastHiddenLayer; a > FIRST_HIDDEN_LAYER; a--) 
        {
            for (int b = 0; b < activationLayerSize[a]; b++)
            {
                omega[a][b] = 0.0;
                for (int c = 0; c < activationLayerSize[a + 1]; c++)
                {
                    omega[a][b] += psi[a + 1][c] * weights[a][b][c];
                    weights[a][b][c] -= lambda * activations[a][b] * psi[a + 1][c];
                } // for (int c = 0; c < activationLayerSize[a + 1]; c++)

                psi[a][b] = omega[a][b] * thresholdFunctionDerivative(theta[a][b]);
            } // for (int b = 0; b < activationLayerSize[a]; b++)
        } // for (int a = lastHiddenLayer; a > FIRST_HIDDEN_LAYER; a--) 
        
        int a = FIRST_HIDDEN_LAYER, d = INPUT_LAYER;
        for (int b = 0; b < activationLayerSize[a]; b++)
        {
            omega[a][b] = 0.0;
            for (int c = 0; c < activationLayerSize[a + 1]; c++)
            {
                omega[a][b] += psi[a + 1][c] * weights[a][b][c];
                weights[a][b][c] -= lambda * activations[a][b] * psi[a + 1][c];
            } // for (int c = 0; c < activationLayerSize[FIRST_HIDDEN_LAYER + 1]; c++)

            psi[a][b] = omega[a][b] * thresholdFunctionDerivative(theta[a][b]);

            for (int c = 0; c < activationLayerSize[d]; c++)
                weights[d][c][b] -= lambda * activations[d][c] * psi[a][b];
        } // for (int b = 0; b < activationLayerSize[FIRST_HIDDEN_LAYER]; b++)

        return;
    } // void learn()

    /*
    * trains the model by running gradient descent until one of two conditions
    * 1. number of iterations exceeds maxIterations
    * 2. average error over all testcases is below errorThreshold
    */
    void train()
    {
        double lastCycleError = INF;
        double avgError = 0.0;
        int iterations = 0;
        bool willPrint;

        int lastImprovementCycle = 0;
        do 
        {
            willPrint = false;
            for (int testCase = 0; testCase < numTests; testCase++)
            {
                forwardPassTrain(testCase);
                learn();
            }

            iterations++;

            if (iterations % 10 == 0) willPrint = true;
            if (iterations % 100 == 0 && saveModel) saveWeightsToFile();
            if (iterations % iterationPrintingFrequency == 0) 
            {
                avgError = calculateAverageError(willPrint);
                cout << "Iteration: " << iterations << "\t";
                cout << "Error: " << avgError << "\t";
                cout << endl;
            }
            else avgError = calculateAverageError(willPrint);
        } // do
        while (iterations < maxIterations && avgError - errorThreshold > EPSILON);

        cout << endl;

        if (avgError - errorThreshold > EPSILON) cout << "FAILURE: Model did not converge!" << endl;
        else cout << "SUCCESS: Model converged after " << iterations << " iterations!" << endl;

        cout << "Error: " << avgError << endl;
        cout << endl;

        if (iterations == maxIterations) 
            cout << "Model terminated due to reaching maximum number of iterations (" << maxIterations << ")." << endl;
        else 
            cout << "Model terminated due to reaching low enough error (<=" << errorThreshold << ")." << endl;
        
        calculateAverageError(true);
        
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

    if (argc == 1) NeuralNetwork network(DEFAULT_CONFIG_FILE);
    else if (argc == 2) NeuralNetwork network(argv[1]);
    else cout << "Provide a single commandline argument, the configuration file" << endl;

    return 0;
}