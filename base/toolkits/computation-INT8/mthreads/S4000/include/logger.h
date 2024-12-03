#pragma once
#include <string>
#include <fstream>
class logger {
  std::ofstream outFile;

 public:
  logger(std::string FileName);
  ~logger();
  // Overloaded function to print on stdout/android activity
  void print(std::string str);
  void print(double val);
  void print(float val);
  void print(int val);
  void print(unsigned int val);
};
