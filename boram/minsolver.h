/*!
 * @header minsolver.h
 * @brief This header contains the implementation of base class
 * for minimization solver algorithms.
 
 * @author C. Silva Ramos <caio.silva_@hotmail.com>
 * @copyright  2002-2004 C. Armando Duarte <caduarte@illinois.edu>
 * @version    0.1
 */

#ifndef MINSOLVER_H
#define MINSOLVER_H

#include <Eigen/Dense>
#include <exception>
#include "monitor.h"
#include "linesearch.h"

// Alias for a generic Eigen vector
template<typename T>
using EigenVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

// Alias for a generic reference to a Eigen vector
template<typename T>
using RefEigenVec = Eigen::Ref<EigenVec<T>>;

/**
 * @class minSolver
 * 
 * @brief Based class used for derived minimization solver classes.
 *  
 * @tparam Foo
 *    Generic object type that must allow access to these methods:
 *    
 *    - getNumEquations();
 *       This method must return a integer contain the number of Degrees of 
 *       Freedom (DOFs) related to the current physics.
 * 
 *    - getSolution( RefEigenVec );
 *       This method must receive a reference to a Eigen vector and overwrite 
 *       it with the current physics solution.
 * 
 *    - setSolution( const EigenVec& );
 *       This method must receive a const reference to a Eigen vector and update 
 *       the physics solution using this vector.
 * 
 *    - computeGradient( RefEigenVec ); // AQUINATHAN Why not computeResidual?
 *       This method must receive a reference to a Eigen vector and overwrite 
 *       it with the current physics gradient/residual.
 * 
 *    - computeJacobian();
 *       This method must compute the hessian/tangent/jacobian matrix and factorize 
 *       it.            
 * 
 *    - solve( RefEigenVec );
 *       This method must receive a reference to a Eigen vector. Using the factorized
 *       jacobian matrix, compute the solution using the vector given as argument and 
 *       overwrite it with the solution.
 * 
 *    AQUINATHAN Shouldn't we also implement solveAnalysis? AM uses it for each physics
 */
template <typename Foo, typename realType = double>
class minSolver {
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
   enum SolverType {ESAM, ESRAM, ESLBFGS, ESBORAM, ESNone};

   /**
    * @brief Construct a new minSolver object
    * 
    */
   minSolver():
   _type(ESNone),_dim(-1),_nItMax(-1),_nIt(-1),
   _absResTol(-1),_relResTol(-1),_relSolTol(-1),_monitor(nullptr) {}

   // Removing copy constructor
   minSolver(const minSolver &obj) = delete;

   /**
    * @brief Destroy the minSolver object
    * 
    */
   virtual ~minSolver() {

      delete _monitor;

      typename std::vector<Foo*>::iterator it;
      for(it = _mphys.begin(); it != _mphys.end(); it++)
         delete (*it);

   } 

   inline SolverType type() const {return _type;}
   inline void setMaxNumIter(int const num) {_nItMax=num;}
   inline void setRelIncrSolTolerance(const realType tol) {_relSolTol=tol;}
   inline void setRelResTolerance(const realType tol) {_relResTol=tol;}
   inline void setAbsResTolerance(const realType tol) {_absResTol=tol;}

   /**
    * @brief Adding a new physics to the minimization
    * solver algorithm
    * 
    * @param phys 
    *    Physics object that respects the requirements of the
    *    typename class
    */
   inline void addPhysics(Foo* phys) {
      _mphys.push_back(phys);
   }

   /**
    * @brief Get the list of Physics object
    * 
    * @return std::vector<Foo*> 
    */
   inline std::vector<Foo*> getPhysics() {return _mphys;}

   inline void monitorName(std::string name) {
      _monitor = new Monitor(name);
   }

   /**
    * @brief Virtual function to solve the full analysis.
    * 
    * @details It must be not called, since there is no implementation 
    * for base class. 
    * 
    * @return int 
    */
   inline virtual int solveAnalysis(RefVec<realType> X,
                              RefVec<realType> dX,
                              RefVec<realType> G) {
      std::cerr << "\nminSolver::solve(): There is no "
                "definition of solve() function in the base class\n";
      throw std::exception();
      return 0;
   }

   /**
    * @brief This method returns the full problem solution
    * 
    * @param X 
    *    Reference to a Eigen vector
    */
   inline void getSolution(RefVec<realType> X) {
      
      int initPos = 0;
      typename std::vector<Foo*>::iterator it = _mphys.begin();
      // Looping over the physics to get the solution vector of
      // each one individually.
      for(; it != _mphys.end(); it++) {

         // Number of degrees of freedom of each physics.
         int physDim = (*it)->getNumEquations();

         // Saving the solution vector.
         (*it)->getSolution(X.segment(initPos, physDim));
         
         // Updating global vector mapping for next physics.
         initPos += physDim;                
      }

   }
   
   /**
    * @brief This method defines the full problem solution
    * 
    * @param X 
    *    Const reference to a Eigen vector
    */
   inline void setSolution(const ConstRefVec<realType> &X) {
      
      int initPos = 0;
      typename std::vector<Foo*>::iterator it = _mphys.begin();
      // Looping over the physics to set the solution vector of
      // each one individually.
      for(; it != _mphys.end(); it++) {

         // Number of degrees of freedom of each physics.
         int physDim = (*it)->getNumEquations();

         // Setting the solution vector.
         (*it)->setSolution(X.segment(initPos, physDim));
         
         // Updating global vector mapping for next physics.
         initPos += physDim;                
      }

   }

   /**
    * @brief This method returns the full problem gradient/residual
    * 
    * @param G 
    *    Reference to a Eigen vector
    */
   inline void computeGradient(RefVec<realType> G) {

      int initPos = 0;
      typename std::vector<Foo*>::iterator it = _mphys.begin();
      // Looping over the physics to compute the residual vector of
      // each one individually.
      for(; it != _mphys.end(); it++) {

         // Number of degrees of freedom of each physics.
         int physDim = (*it)->getNumEquations();

         // Computing residual vector.
         (*it)->computeGradient(G.segment(initPos, physDim));
         
         // Updating global vector mapping for next physics.
         initPos += physDim;                
      }
   }

   /**
    * @brief Compute L2 norm of the full residual vector
    * 
    * @details Based on the number of DOFs of each physics,
    * this method stores, individually, the L2 norm of a piece 
    * of the global vector that is related to the physics.
    * 
    * @param G 
    *    Const reference to a Eigen vector
    */
   virtual void initializeConvCriterion(const ConstRefVec<realType> &G, const ConstRefVec<realType> &dX) {
      
      _G0Norm = Vec<realType>::Zero(this->_mphys.size());
      _dX0Norm = Vec<realType>::Zero(this->_mphys.size());

      int phys = 0;
      int initPos = 0;
      typename std::vector<Foo*>::iterator it = _mphys.begin();
      for(; it != _mphys.end(); it++, phys++) {

         // Number of degrees of freedom of each physics.
         int physDim = (*it)->getNumEquations();

         // Computing L2 norm of the residual vector 
         // for each physics.
         _G0Norm(phys) = G.segment(initPos, physDim).norm();
         _dX0Norm(phys) = dX.segment(initPos, physDim).norm();

         if(_dX0Norm(phys) < 1.0e-15)
            _dX0Norm(phys) = 1.0;

         if(_G0Norm(phys) < 1.0e-15)
            _G0Norm(phys) = 1.0;
         
         // Updating global vector mapping for next physics.
         initPos += physDim;                 
      }

   }

   /**
    * @brief Insert a user-defined information to the convergence monitor
    * 
    * @param val 
    *    std::string
    */
   inline void printOnConvMonitor(const std::string val) {_monitor->write(val);}

   /**
    * @brief Starting/Restarting the convergence monitor timer
    * 
    */
   inline void monitorTimeRestart() {_monitor->timeRestart();}

   /**
    * @brief Computing and printing the time between call monitorTimeRestart() 
    * and this method
    * 
    */
   inline void monitorPrintTime() {_monitor->printTime();}

protected:

   /**
    * @brief Compute and factorize the full jacobian matrix
    * 
    */
   inline void computeJacobian() {

     //     int initPos = 0;
      typename std::vector<Foo*>::iterator it = _mphys.begin();
      // Looping over the physics to set the solution vector of
      // each one individually.
      for(; it != _mphys.end(); it++) {

         // Number of degrees of freedom of each physics.
         // CAD: This is set but not used: clang gives warning.
         // int physDim = (*it)->getNumEquations();

         // Computing and factoring the jacobian matrix.
         (*it)->computeJacobian();
         
         // Updating global vector mapping for next physics.
         // CAD: This is set but not used: clang gives warning.
         // initPos += physDim;                
      }

   }

   /**
    * @brief Compute the incremental solution vector
    * 
    * @param dX 
    *    Reference to a Eigen vector
    */
   inline void solve(RefVec<realType> dX) {

      int initPos = 0;
      typename std::vector<Foo*>::iterator it = _mphys.begin();
      // Looping over the physics to set the solution vector of
      // each one individually.
      for(; it != _mphys.end(); it++) {

         // Number of degrees of freedom of each physics.
         int physDim = (*it)->getNumEquations();

         // Setting the solution vector.
         (*it)->solve(dX.segment(initPos, physDim));
         
         // Updating global vector mapping for next physics.
         initPos += physDim;                
      }   

   }  

   /**
    * @brief Initialize class variables
    * 
    */
   inline virtual void initialize() {

      int globDim = 0;
      typename std::vector<Foo*>::iterator it = this->_mphys.begin();
      // Looping over the physics to compute the residual vector of
      // each one individually.
      for(; it != this->_mphys.end(); it++)
         globDim += (*it)->getNumEquations();
      
      _dim = globDim;

      _nIt = 0;
     
   };

   /**
    * @brief Evaluate the convergence of each physics individually.
    * 
    * @param X 
    *    Const reference to the solution vector
    * 
    * @param dX 
    *    Const reference to the increment solution vector
    * 
    * @param G 
    *    Const reference to the gradient/residual vector
    * 
    * @return bool  
    */
   inline bool checkConvergence(const ConstRefVec<realType> &X, 
                                 const ConstRefVec<realType> &dX, 
                                 const ConstRefVec<realType> &G) {
      
      std::vector<bool> physConvCheck;
      std::stringstream line;

      physConvCheck.resize(_mphys.size());

      // Writting in the monitoring file the current iteration
      // and the physics identifier. 
      int phys = 0;
      line << std::setw(7) << std::to_string(_nIt) 
           << std::setw(11) << std::to_string(phys+1);       

      int initPos = 0;
      typename std::vector<Foo*>::iterator it = _mphys.begin();
      for(; it != _mphys.end(); it++, phys++) {
         
         realType absLInfNormGrad, absL2NormGrad, 
               absLInfNormdSol, absL2NormdSol, 
               relL2NormGrad, absLInfNormSol,
               relL2NormdSol, energy=0.0;

         // Number of degrees of freedom of each physics.
         int physDim = (*it)->getNumEquations();

         // Computing infinity norm of the solution and residual vector 
         // for each physics.
         absLInfNormSol = X.segment(initPos, physDim).template lpNorm<Eigen::Infinity>();
         absLInfNormGrad = G.segment(initPos, physDim).template lpNorm<Eigen::Infinity>();
         absLInfNormdSol = dX.segment(initPos, physDim).template lpNorm<Eigen::Infinity>();

         // Computing L2 norm of the solution and residual vector 
         // for each physics.
         absL2NormGrad = G.segment(initPos, physDim).norm();
         absL2NormdSol = dX.segment(initPos, physDim).norm();

         // Computing L2 relative norm of the residual vector
         // for each physics.
         relL2NormGrad = absL2NormGrad/_G0Norm(phys);
         relL2NormdSol = absL2NormdSol/_dX0Norm(phys);

         bool res_converged = false;
         res_converged = absL2NormGrad < _absResTol || 
                        absL2NormGrad < _relResTol*_G0Norm(phys);

         bool sol_converged = false;
         sol_converged = absL2NormdSol < _relSolTol*_dX0Norm(phys);

         // Convergence check
         physConvCheck[phys] = res_converged && sol_converged;
         
         // Updating global vector mapping for next physics.
         initPos += physDim; 

         // Saving norms values to the monitoring file for each
         // physics.
         line << std::fixed << std::setprecision(5) << std::scientific
                           << std::setw(15) << absLInfNormSol
                           << std::setw(15) << absLInfNormdSol
                           << std::setw(15) << absL2NormdSol
                           << std::setw(15) << absLInfNormGrad
                           << std::setw(15) << absL2NormGrad
                           << std::setw(15) << relL2NormdSol
                           << std::setw(16) << relL2NormGrad << "\n";
         
         if(it != _mphys.end()-1)
            // Writting in the monitoring file the next 
            // physics identifier.
            line << std::setw(18) << std::to_string(phys+2);         
      }

      _monitor->write(line.str());
      line.str(std::string());

      // return !(std::find(physConvCheck.begin(), physConvCheck.end(), false) != physConvCheck.end());
      return physConvCheck[0]==true && physConvCheck[1]==true;
   }

   /**
    * @brief Print a standardized preamble to the convergence monitoring file.
    * 
    */
   inline void monitorPreamble() {
      std::stringstream line;
      std::string aux;

      int skip, rest=0;
      aux = "ITER"; skip = 3 + aux.size();
      line << std::setw(skip) << aux;
      aux = "PHYSICS"; skip = aux.size();
      line << std::setw(skip + 4) << aux;
      aux = "|X|"; skip = 6 + 0.5*aux.size(); rest = 11 - skip;
      line << std::setw(skip + 4) << aux;
      aux = "|dX|"; skip = 5 + 0.5*aux.size(); rest = 11 - skip;
      line << std::setw(skip + 4 + rest) << aux;
      aux = "||dX||"; skip = 5 + 0.5*aux.size(); rest = 11 - skip;
      line << std::setw(skip + 5 + rest) << aux;
      aux = "|G|"; skip = 5 + 0.5*aux.size(); rest = 10 - skip;
      line << std::setw(skip + 4 + rest) << aux;
      aux = "||G||"; skip = 5 + 0.5*aux.size(); rest = 11 - skip;
      line << std::setw(skip + 4 + rest) << aux;
      aux = "||dX||/||dX0||"; skip = 5 + 0.5*aux.size(); rest = 11 - skip;
      line << std::setw(skip + 9 + rest) << aux;
      aux = "||G||/||G0||"; skip = 5 + 0.5*aux.size(); rest = 8 - skip;
      line << std::setw(skip + 7 + rest) << aux <<"\n\n";

      _monitor->write(line.str());
   }

   /** Variables **/

   SolverType _type;

   // List of physics problems
   std::vector<Foo*> _mphys;

   // Global number of Degrees of Freedom
   int _dim;

   // Maximum number of iterations allowed 
   int _nItMax;

   // Current number of iterations
   int _nIt;
   
   // Converge tolerances
   realType _absResTol;
   realType _relResTol;
   realType _relSolTol;

   // Convergence monitor
   Monitor* _monitor;

   // L2 norm of the initial residual and increment 
   // solution vector for each physics
   Vec<realType> _G0Norm;
   Vec<realType> _dX0Norm;

};

#endif // MINSOLVER_H
