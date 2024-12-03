#include "logger.h"
#include <iomanip>
#include <iostream>

logger::logger(std::string FileName) {
  outFile.open(FileName);
  if (!outFile.is_open()) {
    throw std::runtime_error("Failed to open log file.");
  }
  outFile.flush();
}

logger::~logger() {
  outFile.close();
}


void logger::print(std::string str) {

  std::cout << str;
  std::cout.flush();
  if (!outFile.is_open()) {
    throw std::runtime_error("Failed to open log file.");
  }
  if (!outFile.good()) {
    std::cerr << "outFile is in a bad state!" << std::endl;
    outFile.clear();
  }
  if (!(outFile << str)) {
    std::cerr << "Failed to write to outFile." << std::endl;
  }
  outFile.flush();

}

void logger::print(double val) {
  std::cout << std::setprecision(2) << std::fixed;
  std::cout << val;
  std::cout.flush();
  outFile << std::setprecision(2) << std::fixed;
  outFile << val;
  outFile.flush();
}

void logger::print(float val) {
  std::cout << std::setprecision(2) << std::fixed;
  std::cout << val;
  std::cout.flush();
  outFile << std::setprecision(2) << std::fixed;
  outFile << val;
  outFile.flush();
}

void logger::print(int val) {
  std::cout << val;
  std::cout.flush();
  outFile << val;
  outFile.flush();
}

void logger::print(unsigned int val) {
  std::cout << val;
  std::cout.flush();
  outFile << val;
  outFile.flush();
}