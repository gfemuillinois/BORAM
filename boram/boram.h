/*!
 * @header boram.h
 * @brief This header contains the implementation of L-BFGS Over-Relaxed 
 * Alternate Minimization solver (BORAM).
 
 * @author C. Silva Ramos <caio.silva_@hotmail.com>
 * @copyright  2002-2004 C. Armando Duarte <caduarte@illinois.edu>
 * @version    0.1
 */

#ifndef BORAM_H
#define BORAM_H

#include "lbfgs.h"
#include "am.h"

template <typename Foo, typename realType = double>
class BORAMSolver: 
public alternMinSolver<Foo, realType>,  
public LBFGSSolver<Foo, realType> {
private:
   // Alias for a generic Eigen vector
   template<typename T>
   using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

   // Alias for a generic Eigen dense matrix
   template<typename T>
   using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

   // Alias for a generic reference to a Eigen vector
   template<typename T>
   using RefVec = Eigen::Ref<Vec<T>>;

   // Alias for a generic const reference to a Eigen vector
   template<typename T>
   using ConstRefVec = Eigen::Ref<const Vec<T>>;

public:
   // Constructor
   BORAMSolver(): alternMinSolver<Foo, realType>(1.8),
                   LBFGSSolver<Foo, realType>() {
      this->_nItMax = 100;
      this->_absResTol = 1.0e-15;
      this->_relResTol = 5.0e-3;
      this->_relSolTol = 1.0e-02;

      _relaxing = false;
      _Nw = 8;
      _convRate = -0.1;

      this->_type = minSolver<Foo, realType>::ESBORAM;
   }

   // Removing copy contructor
   BORAMSolver(const BORAMSolver &obj) = delete;

   // Destructor
   virtual ~BORAMSolver() override = default;

   inline void setTrendLineNumSamples(const int unsigned nSamples) {_Nw = nSamples;}
   inline void setConvRate(const realType convRate) {_convRate = convRate;}

   inline int solveAnalysis(RefVec<realType> X,
                              RefVec<realType> dX,
                              RefVec<realType> G) override {

      // Writting in the monitoring file a preamble
      // on convergence check
      this->monitorPreamble();             

      // Initializing class variables
      this->initialize();

      // Early exit if the initial guess solution is already a minimizer
      if (!this->checkConvergence(X, dX, G)) {

         // Saving start time, before run any
         // iteration/update
         this->monitorTimeRestart();

         // Loop over iterations
         int k=0;
         for( ; ; k++) {

            // Updating number of iterations
            this->_nIt = k+1;

            printf("\nBORAMSolver::solveAnalysis(): Solving iteration %d", this->_nIt);

            // Applying boram for a given iteration.
            this->applyBORAM(X, dX, G);

            // Convergence test 
            if (this->checkConvergence(X, dX, G)) 
               break;

            // Checking maximum number of iterations
            if (this->_nIt >= this->_nItMax) {
               std::cerr << "\nBORAMSolver::solveAnalysis(): The maximum number of "
                     "iterations was reached and the problem did not converge\n";
               throw std::exception();
               return -1;
            }
            
         }
      }

      // Writing to the monitoring file the time taken by 
      // the boram scheme to converge
      this->monitorPrintTime();

      _relaxing = false;
      _GHistNorm.resize(0);

      return this->_nIt;
   }

private:
   inline void applyBORAM(RefVec<realType> X, 
                           RefVec<realType> dX, 
                           RefVec<realType> G) {

      if(this->checkHistory()){

         if(_relaxing) {
            _relaxing = false;
            this->restart(); // Restarting L-BFGS variables
         }

         printf("\nLBFGS Iteration:  %d", this->_nIt); 
         this->applyLBFGS(X, dX, G); 
      } else {
         printf("\nOver-Relaxed Iteration:  %d", this->_nIt); 
         this->applyStaggered(X, dX, G);
         _relaxing = true;
      }

      realType GNorm = G.norm();
      this->addHistoryVal(GNorm);

   }

   inline void addHistoryVal(realType const GNorm) {
      
      if( this->_nIt > 1 ){
         Vec<realType> prevG = _GHistNorm.segment(1, _Nw-1);
         _GHistNorm.segment(0, _Nw-1) = prevG;
         _GHistNorm(_Nw-1) = GNorm;

         printf("\n");   
         for(int i=0; i<_Nw-1; i++)
            printf("%10.8e > ", _GHistNorm(i));   
         printf("%10.8e \n ", _GHistNorm(_Nw-1));   
      }
   }

   inline bool checkHistory() {
      bool isDecend = true;

      if( this->_nIt > _Nw + 1 ) {
         
         // S(b) = (y - Xb)^T (y - Xb)
         // dSb/db = -2 X^Ty + 2 X^TXb = 0
         // b = inv(X^TX)X^Ty
         // alpha = b(0)
         // beta = b(1)  *** slope ***
         // f(Iter) = 10^(alpha + beta*Iter) [Trendline equation]

         // y = log10(GNorms) 
         Vec<realType> logG = _GHistNorm.array().log10();

         // X is a matrix where the first column (X(:,0)) is filled with ones and
         // second column (X(:,1)) with the x values, i.e, the iteration number. 
         Mat<realType> ItMat = Mat<realType>::Ones(_Nw, 2);
         ItMat.col(1) = Vec<realType>::LinSpaced(_Nw, this->_nIt - _Nw + 1, this->_nIt);

         // Computing X^TX
         Mat<realType> XtX = ItMat.transpose()*ItMat;

         // X^TX is a 2x2 matrix, here the inverse of 2x2 matrix is computed
         realType detXtX = XtX(0,0)*XtX(1,1) - XtX(0,1)*XtX(1,0);
         Mat<realType> invXtX(2,2);
         invXtX(0,0) =  XtX(1,1)/detXtX; invXtX(1,1) =  XtX(0,0)/detXtX;
         invXtX(0,1) = -XtX(0,1)/detXtX; invXtX(1,0) = -XtX(1,0)/detXtX;

         // Computing coefficients vector
         Vec<realType> b = invXtX*ItMat.transpose()*logG;

         // getting slope
         realType beta = b(1);

         printf("\nIteration: %d, trendline inclination: %10.8e\n", this->_nIt, beta); 

         // Checking the user's specified convergence rate.
         if(beta > _convRate) 
            isDecend = false;
      }

      return isDecend;
   }

   inline void initialize() override {

      LBFGSSolver<Foo, realType>::initialize();

      _GHistNorm = Vec<realType>::Zero(_Nw);

   }
   
   /** Variables **/

   bool _relaxing; 
   int _Nw;
   realType _convRate;
   
   Vec<realType> _GHistNorm;
};

#endif // BORAM_H