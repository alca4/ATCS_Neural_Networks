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
      getline(cin, curLine);

      if (curLine.length() > 0 && curLine[0] != '/' && curLine[0] != '*') // remove header comments
      {
         // take everything until you reach a space
         string info;
         int idx = 0; 
         while (idx < (int) curLine.length() && curLine[idx] != ' ') 
         {
            info += curLine[idx];
            idx++;
         }

         if (info.length() > 0) {
            if (param == "") param = info;
            else 
            {
               inputs[param] = info;
               param = "";
            }
         }
      }
   }

   return inputs;
}

int main()
{
   freopen(INPUT_FILE, "r", stdin);

   configureModel();

   return 0;
}