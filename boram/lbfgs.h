/*!
 * @header lbfgs.h
 * @brief This header contains the implementation of L-BFGS.
 
    Limited-Memory Broyden-Fletcher-Goldfarb-Shano (L-BFGS) solver for unconstrained 
    nonlinear system of equations.
 
 * @author C. Silva Ramos <caio.silva_@hotmail.com>
 * @copyright  2002-2004 C. Armando Duarte <caduarte@illinois.edu>
 * @version    0.1
 */

#ifndef LBFGS_H
#define LBFGS_H

#include "minsolver.h"

/**
 * @class LBFGSSolver
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
 *    base class.
 *    
 */
template <typename Foo, typename realType = double>
class LBFGSSolver: virtual public minSolver<Foo, realType> {
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
   enum LBFGSType {ELBFGS, ECLBFGS, EMPLBFGS, ELBFGSNone};

   // Constructor
   LBFGSSolver(): minSolver<Foo, realType>() {
      this->_type = minSolver<Foo, realType>::ESLBFGS;
      _BFGStype = ELBFGS;
      this->_nItMax = 100;
      this->_absResTol = 1.0e-15;
      this->_relResTol = 5.0e-03;
      this->_relSolTol = 1.0e-02;
      _m = 8;
      _linesearch = new QNRLineSearch<LBFGSSolver<Physics, realType>, realType>();
   }

   // Removing copy contructor
   LBFGSSolver(const LBFGSSolver &obj) = delete;

   // Destructor
   virtual ~LBFGSSolver() override {
      delete _linesearch;
   }

   inline void setNumUpdates(const int val) {_m = val;}

   /**
    * @brief Get the Line Search object
    * 
    * @return LineSearch<Foo, realType>* 
    */
   inline LineSearch<LBFGSSolver, realType>* getLineSearch() const {return _linesearch;}

   /**
    * @brief Set the Line Search object
    * 
    * @param ls 
    */
   inline void setLineSearch(LineSearch<LBFGSSolver, realType> *ls) {
      delete _linesearch;
      _linesearch = ls;
   }

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
         // LBFGS iteration/update
         this->monitorTimeRestart();

         // Loop over iterations
         int k=0;
         for( ; ; k++) {

            // Updating number of iterations
            this->_nIt = k+1;

            printf("\nLBFGSSolver::solveAnalysis(): Solving iteration %d", this->_nIt);
            printf("\n                                 update block = %d\n", this->_nIt/this->_m);

            // Applying recursive formula to compute dX = - K * g.
            // Also computing line search parameter and checking 
            // convergence. 
            this->applyLBFGS(X, dX, G);

            // Convergence test 
            if (this->checkConvergence(X, dX, G)) 
               break;

            // Checking maximum number of iterations
            if (this->_nIt >= this->_nItMax) {
               std::cerr << "\nLBFGSSolver::solveAnalysis(): The maximum number of "
                     "iterations was reached and the problem did not converge\n";
               throw std::exception();
               return -1;
            }
         }
      }

      // Writing to the monitoring file the time taken by 
      // the lbfgs scheme to converge
      this->monitorPrintTime();

      this->_G0Norm.resize(0);

      return this->_nIt;
   }

protected:
   inline void applyLBFGS(RefVec<realType> X, 
                           RefVec<realType> dX, 
                           RefVec<realType> G) {

      // Periodic restart, computing new jacobian matrix 
      // and factorize. Also restarting class variables. 
      if(_restart) 
         this->reset(); 

      /**
       * LBFGS two-loop recursion
       * 
       * for i = m,m-1,m-2,...,0
       *    rho[i] = 1 / (y[i]^T s[i])
       *    alpha[i] = rho[i] * (s[i]^T r)
       *    r = r - (alpha[i] * y[i])
       * end
       * 
       * r = K_0^{-1} * r
       * 
       * for i = 0,1,2,...,m
       *    beta = rho[i] * (y[i]^T r)
       *    r = r + ((alpha[i] - beta) * s[i])
       * end
       * 
       * r = -r
       * 
       */
      
      Vec<realType> alpha(_mCorr);

      dX.noalias() = G;

      // Loop 1
      int j = _mCorr;
      printf("\nLBFGSSolver::applyLBFGS(): Iter = %d", this->_nIt);
      printf("\n   LOOP 1");
      for(int i=0; i<_mCorr; i++) {
         j -= 1;
         alpha[j] = _s.col(j).dot(dX)*_rho(j);
         dX.noalias() -= alpha(j) * _y.col(j);

         printf("\n   alpha_%d = %10.8e", j, alpha.coeffRef(j));
      }

      // Apply initial K_0
      this->solve(dX);

      // Loop 2
      printf("\n   LOOP 2");
      for(int i=0; i<_mCorr; i++) {
         const realType beta = dX.dot(_y.col(j))*_rho(j);
         dX.noalias() += (alpha(j)-beta)*_s.col(j);

         printf("\n   beta = %10.8e", beta);
         printf("\n   (alpha_%d - beta) = %10.8e", j, (alpha(j)-beta));

         j += 1;
      }

      dX.noalias() = -dX;

      Vec<realType> Gold = G;
      Vec<realType> Xold = X;
      _linesearch->applyLineSearch(this, X, dX, G);

      // Adding new correction.
      this->addCorrection(X, Xold, G, Gold);

   }

   inline void restart() {_restart = true;}

   inline void initialize() override {

      minSolver<Foo, realType>::initialize();

      _s = Mat<realType>::Zero(this->_dim, _m);
      _y = Mat<realType>::Zero(this->_dim, _m);
      _rho = Vec<realType>::Zero(_m);

      _restart=true;

   }

private:
   inline void reset() {

      _s = Mat<realType>::Zero(this->_dim, _m);
      _y = Mat<realType>::Zero(this->_dim, _m);
      _rho = Vec<realType>::Zero(_m);

      _mCorr = 0;
      _numSkipUpd = 0;

      _restart = false;
      
      this->computeJacobian();

      if(this->_nIt!=1) {
         printf("\nLBFGSSolver::reset(): Periodic restart!");
         printf("\n                  Number of updates = %d", this->_m);
         printf("\n               Number of iterations = %d\n", this->_nIt);
      }
   }

   inline void addCorrection(RefVec<realType> &X, 
                              const ConstRefVec<realType> &Xold, 
                              RefVec<realType> &G, 
                              const ConstRefVec<realType> &Gold) {

      if(_mCorr < _m && _mCorr+_numSkipUpd < _m) {

         Vec<realType> y = G - Gold;
         Vec<realType> s = X - Xold;

         realType yTs = s.dot(y);

         bool isAddCorr = false;

         //Standard L-BFGS
         if(_BFGStype==ELBFGS) {
            if(yTs > 1.0e-16) 
              isAddCorr = true;

            printf("\nLBFGSSolver::addCorrection(): "
                   "\n   LBFGS Check: %s"
                   "\n   it = %d, y_k.s_k = %10.8e\n", 
                   isAddCorr ? "true" : "false", this->_nIt, yTs);

         //'Caution' L-BFGS
         } else if (_BFGStype==ECLBFGS) {
            realType Gnorm = G.norm();
            realType mu;

            realType sNorm = s.squaredNorm();
            realType eta = yTs/sNorm;

            if(Gnorm>=1.0)
               mu = std::pow(Gnorm,0.01);
            else
               mu = std::pow(Gnorm, 3.0);

            if(eta>0.1*mu)
               isAddCorr = true;

            printf("\nLBFGSSolver::addCorrection(): "
                   "\n   CLBFGS Check: %s"
                   "\n   it = %d, y_k.s_k/||s_k||^2 - mu(||G||) = %10.8e\n", 
                   isAddCorr ? "true" : "false", this->_nIt, eta - 0.1*mu);

         // MP L-BFGS
         } else if (_BFGStype==EMPLBFGS) {
            realType Gnorm = G.norm();
            realType mu;

            realType sNorm = s.squaredNorm();
            realType eta = yTs/sNorm;

            if(Gnorm>=1.0)
               mu = std::pow(Gnorm,0.01);
            else
               mu = std::pow(Gnorm, 3.0);

            if(eta>0.1*mu){
               realType t = 1.0 + std::max(0.0, -eta);

               y.noalias() = y + t*Gnorm*s;
            }

            isAddCorr = true;
         }
         

         if(isAddCorr) {
            // Adding new correction.
            // Update s, y and rho
            // s_k   = x_k - x_{k-1}
            // y_k   = g_k - g_{k-1}
            // rho_k = 1/(s_k.y_k)
            _s.col(_mCorr).noalias() = s;
            _y.col(_mCorr).noalias() = y;
            _rho(_mCorr) = 1.0/(_s.col(_mCorr).dot(_y.col(_mCorr)));

            printf("\nLBFGSSolver::addCorrection(): it = %d, rho_%d = %10.8e\n", this->_nIt, _mCorr, _rho(_mCorr));

            _mCorr++;
         } else {
            _numSkipUpd++;
            printf("\nLBFGSSolver::addCorrection(): skipping update that does not generate a positive defined Hessian!"
                   "\n                              y_%dTs_%d = %10.8e\n", _mCorr+_numSkipUpd, _mCorr+_numSkipUpd, yTs);
         }
      } else {
         _restart=true;
      }
      

   }

   /** Variables **/

   LBFGSType _BFGStype;

   LineSearch<LBFGSSolver<Foo, realType>, realType>* _linesearch;

   int  _m; 
   int  _mCorr;
   int  _numSkipUpd;
   bool _restart;

   Vec<realType> _rho;
   Mat<realType> _s;
   Mat<realType> _y;
};

#endif // LBFGS_H