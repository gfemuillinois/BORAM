/*!
 * @header am.h
 * @brief This header contains the implementation of Alternative Minimization (AM)
 * solver.
 
 * @author C. Silva Ramos <caio.silva_@hotmail.com>
 * @copyright  2002-2004 C. Armando Duarte <caduarte@illinois.edu>
 * @version    0.1
 */

#ifndef AM_H
#define AM_H

#include "minsolver.h"

/**
 * @class alternMinSolver
 * 
 * @brief Alternative Minimization (AM) algorithm to solve multi-physical problems.
 *  
 * @details Running back and forth between each physics, this class
 * allows to solve each physics individually, producing a weak coupling
 * that ends when the coupled problem converges (full problem).  
 *  
 * @see minSolver
 * 
 * @tparam Foo
 *    Generic object type that must meet the requirements specified in the 
 *    base class and, furthermore, allow access to the following method:
 *    - solveAnalysis();
 *       This method must be able to solve each physics individually, as a linear
 *       or no non-linear procedure.
 *    
 */
template <typename Foo, typename realType = double>
class alternMinSolver: virtual public minSolver<Foo, realType> {
private:
   // Alias for a generic Eigen vector
   template<typename T>
   using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

   // Alias for a generic reference to a Eigen vector
   template<typename T>
   using RefVec = Eigen::Ref<Vec<T>>;

public:
   /**
    * @brief Construct a new alternMinSolver object
    * 
    */
   alternMinSolver(): minSolver<Foo, realType>() {
      this->_type = minSolver<Foo, realType>::ESAM;
      this->_nItMax = 100;
      this->_absResTol = 1.0e-15;
      this->_relResTol = 5.0e-03;
      this->_relSolTol = 1.0e-02;
      _linesearch = new LineSearch<Foo, realType>();
   }

   /**
    * @brief Construct a new over/under relaxed alternate Min Solver object
    * 
    * @param relaxParam 
    */
   alternMinSolver(const realType relaxParam): minSolver<Foo, realType>() {
      this->_type = minSolver<Foo, realType>::ESRAM;
      this->_nItMax = 100;
      this->_absResTol = 1.0e-15;
      this->_relResTol = 5.0e-03;
      this->_relSolTol = 1.0e-02;
      _linesearch = new RelaxedLineSearch<Physics, realType>(relaxParam);
   }

   // Removing copy contructor
   alternMinSolver(const alternMinSolver &obj) = delete;

   // Destructor
   virtual ~alternMinSolver() override {
      delete _linesearch;
   }

   inline void setRelaxParameter(const realType relaxParam) {
      
     if(this->_type != minSolver<Foo, realType>::ESRAM && 
        this->_type != minSolver<Foo, realType>::ESBORAM) {
         std::cerr << "\nalternMinSolver::setRelaxParameter(): An over/under relaxed  "
                     " AM was not defined.\n";
         throw std::exception();
      }

      RelaxedLineSearch<Foo, realType>* rls = 
                    dynamic_cast<RelaxedLineSearch<Foo, realType>*>(_linesearch);
      
      rls->setLineSearchParameter(relaxParam);
   }

   /**
    * @brief Get the Line Search object
    * 
    * @return LineSearch<Foo, realType>* 
    */
   inline LineSearch<Foo, realType>* getLineSearch() const {return _linesearch;}

   /**
    * @brief This method computes the solution of a multi-physical
    * problem
    * 
    * @return int 
    *    Number of iterations 
    */
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
         
         // Saving start time, before run any AM iteration
         this->monitorTimeRestart();

         // Loop over iterations
         int k=0;
         for( ; ; k++) {
            
            // Updating number of iterations
            this->_nIt = k+1;

            printf("\nalternMinSolver::solveAnalysis(): Solving iteration %d\n", this->_nIt);

            // Applying staggered scheme for a given iteration.
            // All physics will be solved, one at a time, 
            // using its own solving procedure. 
            this->applyStaggered(X, dX, G);

            // Convergence test 
            if (this->checkConvergence(X, dX, G)) 
               break;

            // Checking maximum number of iterations
            if (this->_nIt >= this->_nItMax) {
               std::cerr << "\nalternMinSolver::solveAnalysis(): The maximum number of "
                     "iterations was reached and the problem did not converge\n";
               throw std::exception();
               return -1;
            }
         }
      }

      // Writing to the monitoring file the time taken by 
      // the AM scheme to converge
      this->monitorPrintTime();

      return this->_nIt;
   }

protected:
   /**
    * @brief 
    * 
    * 
    */
   inline void applyStaggered(RefVec<realType> X, 
                              RefVec<realType> dX, 
                              RefVec<realType> G) {
      
      // Global vector map based on the physics number of 
      // Degrees of Freedom (DOFs). 
      int initPos = 0;

      typename std::vector<Foo*>::iterator it = this->_mphys.begin();
      // Looping over the physics to solve each one individually. Also, scalling 
      // the incremental solution using a line search parameter.
      for(; it != this->_mphys.end(); it++) {
         
         // Storing the number of DOFs for a given physics
         int physDim = (*it)->getNumEquations();

         RefVec<realType> XoldPhys = X.segment(initPos, physDim);
         RefVec<realType> dXPhys = dX.segment(initPos, physDim);
         RefVec<realType> GoldPhys = G.segment(initPos, physDim);
         Vec<realType> XPhys = Vec<realType>::Zero(physDim);

         // Storing initial solution.
         (*it)->getSolution(XoldPhys);

         // Solving the physics using its own procedure.
         // This structure allows each problem be linear 
         // or non-linear, each physics controls its behavior.
         (*it)->solveAnalysis();
         
         // Storing the current solution, it means, the solution
         // after the solving procedure converge.
         (*it)->getSolution(XPhys);

         // Computing the correction of the solution.
         dXPhys = XPhys - XoldPhys;

         // Updating solution as X = Xold + lambda*dX.
         // The default is lambda = 1.0, however lambda is a 
         // user-defined parameter when Over-relaxed-staggered scheme 
         // is in use. 
         // Also, the best value of lambda could be computed, in
         // this case, the residual vector before solving the physics
         // (Gold) is needed. Since, so far, only default lambda and 
         // relaxed scheme is available here, Gold is not used.

         // X = Xold;
         _linesearch->applyLineSearch((*it), XoldPhys, dXPhys, GoldPhys);

         // Updating mapping of the next physics
         initPos += physDim;         
      }

   }

private:
   /** Variables **/ 

   // Line search object applied to each physics
   LineSearch<Foo, realType>* _linesearch;

};

#endif // AM_H
