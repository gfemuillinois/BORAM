/*!
 * @header monitor.h
 * @brief [...].
 *  
 * @author C. Silva Ramos <caio.silva_@hotmail.com>
 * @copyright  2002-2004 C. Armando Duarte <caduarte@illinois.edu>
 * @version    0.1
 */

#ifndef MONITOR_H
#define MONITOR_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>

class Monitor {
public:
   void write(const std::string val) {
      _monitor.open(_filename, std::ios_base::app);
      _monitor << val;
      _monitor.close();        
   }

   void timeRestart() {
      _begin = std::chrono::steady_clock::now();
   }

   void printTime() {
      std::chrono::steady_clock::time_point end = 
                  std::chrono::steady_clock::now();
      std::stringstream line;
      line << "\n" << std::setw(18) << "TIME TO SOLVE: " << std::setw(12) 
                   << std::setprecision(6) << 
                   std::chrono::duration_cast<std::chrono::microseconds>(end - _begin).count()/1.0e+6 
                   << std::fixed << " sec.\n\n";
      this->write(line.str());
   }

   Monitor() : _filename("") {}

   Monitor(const std::string filename) : _filename(filename) {
        
      _monitor.open(_filename, std::ofstream::trunc);
      _monitor.close();

   }
private:
   std::ofstream _monitor;
   std::string _filename;

   std::chrono::steady_clock::time_point _begin;

};

#endif // MONITOR_H