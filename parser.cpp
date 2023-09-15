/*
* Author: Andrew Liang
* Date of Creation: 15 Sept 2023
* Description: Parses configuration files for the model
*/
#include <iostream>
#include <string>
#include <map>
using namespace std;

const char* INPUT_FILE = "config.txt";

map<string, string> configureModel() 
{
   map<string, string> inputs;

   string param;
   string curLine;
   while (!cin.eof())
   {
      getline(cin, curLine, ' ');

      if (curLine[0] != '/' && curLine[0] != '*') // remove header comments
      {
         
      }

      cout << curLine << endl;
   }

   return inputs;
}

int main()
{
   freopen(INPUT_FILE, "r", stdin);

   configureModel();

   return 0;
}