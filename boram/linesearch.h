/**
 * @header linesearch.h
 * @brief This header contains the implementation of Line Search procedures.
 * 
 * @author C. Silva Ramos <caio.silva_@hotmail.com>
 * @copyright  2002-2004 C. Armando Duarte <caduarte@illinois.edu>
 * @version    0.1
 */

#ifndef LINESEARCH_H
#define LINESEARCH_H

#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <stdexcept>  // std::runtime_error

class Physics;

/**
 * @brief 
 * 
 * @tparam Foo: Generic object type that must allow access 
 *              to two methods:
 *              - setSolution()
 *              - computeGradient();
 */
template <typename Foo, typename realType = double>
class LineSearch {
protected:
   // Alias for a generic Eigen vector
   template<typename T>
   using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

   // Alias for a generic Eigen vector reference
   template<typename T>
   using RefVec = Eigen::Ref<Vec<T>>;

public:
   enum LineSearchType {ELSRelaxed, ELSQNR, ELSBacktrac, ELSNone};

   /**
    * @brief Construct a new Line Search object
    * 
    */
   LineSearch(): _lambda(1.0), _type(ELSNone) {}

   // Removing default copy constructor
   LineSearch(const LineSearch& obj) = delete;

   // Destructor
   virtual ~LineSearch() = default;

   inline LineSearchType type() const {return _type;}

   /**
    * @brief Get the current line search parameter. 
    * 
    * @return realType 
    */
   inline realType getLineSearchParameter() const {return _lambda;};

   /**
    * @brief Applies a user-defined line search parameter.
    * 
    * @param phys     
    *    Generic object to represent the problem to be solved.
    * @param Xold
    *    Eigen vector to represent the initial solution.
    * @param dX
    *    Eigen vector to represent the incremental solution.
    * @param Gold
    *    Eigen vector to represent the last residual before 
    *    solution be updated. 
    * @return int
    *    Number of iterations needed to compute the minimized line
    *    search parameter. 
    */
   inline virtual int applyLineSearch(Foo* phys, 
                                   RefVec<realType> Xold, 
                                   RefVec<realType> dX, 
                                   RefVec<realType> Gold) {
      
      // Updating solution
      Xold.noalias() = Xold + _lambda*dX;
      
      // Updating physic solution: X = Xold + lambda*dX
      phys->setSolution(Xold);

      // Computing new residual: G(X) = G(Xold + lambda*dX)
      phys->computeGradient(Gold);

      return 1;
   }

protected: 

    // line search parameter used to scale a given incremental 
   // solution vector.
   realType _lambda;

   LineSearchType _type;

};

/**
 * @brief 
 * 
 * @tparam Foo 
 */
template <typename Foo, typename realType = double>
class RelaxedLineSearch: public LineSearch<Foo, realType> {
public:
   /**
    * @brief Construct a new Relaxed Line Search object
    * 
    * @param relaxParam 
    */
   RelaxedLineSearch(const double relaxParam): LineSearch<Foo>() {
      this->_lambda = relaxParam;
      this->_type = LineSearch<Foo, realType>::ELSRelaxed;
   }

   // Removing default constructor.
   RelaxedLineSearch() = delete;

   // Removing default copy constructor.
   RelaxedLineSearch(const RelaxedLineSearch &obj) = delete;

   // Destructor.
   ~RelaxedLineSearch() = default;

   /**
    * @brief Set a line search parameter given by the user.
    * 
    * @param param 
    */
   inline void setLineSearchParameter(const double param) {this->_lambda = param;}
};

/**
 * @brief A energy-based Quasi-Newton Line Search procedure 
 * to compute a minimized parameter.
 * 
 */
template <typename Foo, typename realType = double>
class QNRLineSearch: public LineSearch<Foo, realType> {
private:
   // Alias for a generic Eigen vector
   template<typename T>
   using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

   // Alias for a generic Eigen vector reference
   template<typename T>
   using RefVec = Eigen::Ref<Vec<T>>;

   realType _convTol;
   realType _actTol;
   int  _nItMax; 
   bool _isEvaluate;

   void lineSearchEvaluation(Foo* phys, 
                             Vec<realType> Xold, 
                             Vec<realType> dX,
                             realType lambda) {

      printf("\nQNRLineSearch::lineSearchEvaluation(): Starting line search evaluation\n");
      
      int limiIt;
      realType startLambda, incrLambda;
      if(lambda > 1.0) {
         incrLambda = std::ceil(2.0*lambda)/20.0;
         limiIt=20;
         startLambda=0.0;
      } else if(lambda<0.0) {
         if(lambda>=-1.0){
            incrLambda = 0.1;
            limiIt=20;
            startLambda= -1.0;
         } else {
            incrLambda = -std::floor(2.0*lambda)/20.0;
            limiIt=20;
            startLambda= std::floor(2.0*lambda);
         }
      } else {
         incrLambda = 1.0/10.0;
         limiIt=10;
         startLambda=0.0;
      }

      Vec<realType> G = Vec<realType>::Zero(Xold.size());
      Vec<realType> X = Vec<realType>::Zero(Xold.size());

      std::vector<realType> alphas(limiIt+2), GtdXs(limiIt+2), energys(limiIt+2);
      for(int it=0; it<=limiIt; it++) {
         realType energy, GtdX;
         realType alpha = startLambda + it*incrLambda;

         X.noalias() = Xold + alpha*dX;
         phys->setSolution(X);

         phys->computeGradient(G);
         
         energy = phys->computeEnergy();

         // GtdX = E'(Xold + lambda_{k+1} * dX)
         GtdX = G.dot(dX);

         alphas[it] = alpha;
         GtdXs[it] = GtdX;
         energys[it] = energy;
      };

      X.noalias() = Xold + lambda*dX;
      phys->setSolution(X);

      phys->computeGradient(G);

      alphas[limiIt+1] = lambda; 
      energys[limiIt+1] = phys->computeEnergy();
      GtdXs[limiIt+1] = G.dot(dX);

      printf("\n\n============= LINE SEARCH EVALUATION =============\n");
      printf("    lambda    ;      energy    ;       GtdX      ;\n");
      for(int it=0; it<=limiIt+1; it++)
         printf ("%10.8e;  %10.8e;  %10.8e;\n", alphas[it], energys[it], GtdXs[it]);
      printf("==================================================\n\n");
      
   };

public:
   QNRLineSearch(): _convTol(1.0e-4), _actTol(1.0e-5), _nItMax(20) {
      this->_type = LineSearch<Foo, realType>::ELSQNR;
      this->_isEvaluate = false;
   };

   // Removing default copy constructor.
   QNRLineSearch(const QNRLineSearch &obj) = delete;

   // Destructor.
   ~QNRLineSearch() = default;

   inline void setConvTolerance(const realType convTol) {_convTol = convTol;}
   inline void setActTolerance(const realType actTol) {_actTol = actTol;}
   inline void setMaxNumIter(const int nIt) {_nItMax = nIt;}

   /**
    * @brief Compute the minimized line search parameter based on
    * quasi-newton method.
    * 
    * @param phys     
    *    Generic object to represent the problem to be solved.
    * @param Xold     
    *    Eigen vector to represent the initial solution.
    * @param dX
    *    Eigen vector to represent the incremental solution.
    * @param Gold
    *    Eigen vector to represent the last residual before 
    *    solution be updated. 
    * @return int 
    *    Number of iterations needed to compute the minimized 
    *    line search parameter. 
    */
   inline int applyLineSearch(Foo* phys, 
                              RefVec<realType> Xold, 
                              RefVec<realType> dX, 
                              RefVec<realType> Gold) override {

      int k=1;                                    
      Vec<realType> G, X;
      realType lambda=1.0, GtdX=0.0, GoldtdX=0.0;

      // Computing Energy value at current solution
      // realType Einit = phys->computeEnergy();

      // Computing first derivative of the potencial energy 
      // in relation to the line search parameter (Projection of 
      // gradient on search direction).
      // E'(Xold + lambda * dX) = dE(Xold + lambda * dX)/dLambda 
      //                        = Grad(Xold + lambda * dX)^T * dX. 
      // GoldtdX = Gold^T * dX.
      GoldtdX = Gold.dot(dX);

      // Checking if dX is not a descent direction
      if (GoldtdX > 0.0) {
         // throw std::logic_error("the moving direction increases the objective function value");
         printf("\nQNRLineSearch::applyLineSearch(): The moving direction "
             "increases the objective function value."
             "\n   Using direction of steepest descent.\n");

         // dX = -(1.0/Gold.norm()) * Gold;
      }
      
      // Updating the current solution.
      X.noalias() = Xold + lambda*dX;
      
      // Updating physic solution: X = Xold + lambda*dX
      phys->setSolution(X);

      // Computing new residual: G(X) = G(Xold + lambda*dX)
      G = Vec<realType>::Zero(Gold.size());
      phys->computeGradient(G);
      
      // Projection of gradient on search direction for updated solution
      GtdX = G.dot(dX);
      
      // The minimized lambda occurs when E'(Xold + lambda * dX) = 0.
      // Checking if it respect the user tolerance.
      printf("\nQNRLineSearch::applyLineSearch(): Checking if Line Search needs "
             "to be activated."
             "\n   lambda = 0.0   E'(lambda) = %10.8e"
             "\n   lambda = 1.0   E'(lambda) = %10.8e"
             "\n         |E'(1.0) - E'(0.0)| = %10.8e" 
             "\n                         tol = %2.1e\n", 
             GoldtdX, GtdX, std::abs(GtdX - GoldtdX), _actTol);

      if(std::abs(GtdX - GoldtdX)>_actTol) {
         
         printf("\nQNRLineSearch::applyLineSearch(): Line Search activated!\n");

         Vec<realType> Y;
         realType lambdaOld=0.0;

         // Applying secant method to computing a approximated
         // second derivative of potencial energy:
         // E"(Xold + lambda * dX) = E'(Xold + lambda_k * dX) - E'(Xold + lambda_{k-1} * dX)
         //                                        lambda_k - lambda_{k-1}
         //                        = (Grad(Xold + lambda_k * dX) - Grad(Xold + lambda_{k-1} * dX)) * dX
         //                                        lambda_k - lambda_{k-1}
         //                        = (Y_k * dX) / (lambda_k - lambda_{k-1});
         for( ; ; k++) {   

            // Y_k = Grad(Xold + lambda_k * dX) - Grad(Xold + lambda_{k-1} * dX)
            Y.noalias() = G - Gold;

            // Computing the new lambda:
            // lambda_{k+1} = lambda_k - (lambda_k - lambda_{k-1}) * (Grad(Xold + lamda_k * dX) * dX) / Y_k * dX 
            realType dXtY = dX.dot(Y);
            realType deltaLambda = -(lambda - lambdaOld)*GtdX/dXtY;
            lambdaOld = lambda;
            lambda += deltaLambda;

            // Updating physics solution: X = Xold + lambda_{k+1} * dX
            X.noalias() = Xold + lambda*dX;
            phys->setSolution(X);

            // Computing new residual: G(X) = G(Xold + lambda_{k+1} * dX)
            Gold.noalias() = G;
            phys->computeGradient(G);

            // GtdX = E'(Xold + lambda_{k+1} * dX)
            GtdX = G.dot(dX);

            // Checking maximum number of iterations
            if (k >= this->_nItMax) {
               
               // lambda = 1.0;

               // X.noalias() = Xold + lambda*dX;
               // phys->setSolution(X);

               // // Computing new residual: G(X) = G(Xold + lambda_{k+1} * dX)
               // Gold.noalias() = G;
               // phys->computeGradient(G);

               printf("\nQNRLineSearch::applyLineSearch(): The maximum number of "
                      "iterations was reached and the Line Search did not converge!"
                      "\nA Line Search parameter equal to 1.0 will be assumed!\n");
               // std::cerr << "\nQNRLineSearch::applyLineSearch(): The maximum number of "
               //       "iterations was reached and the problem did not converge\n";
               // throw std::exception();

               break;
            }
            
            printf("\nQNRLineSearch::applyLineSearch(): Solving iteration %d"
                   "\n   lambda = %10.8e   E'(lambda) = %10.8e\n", k, lambda, GtdX);

            // Convergence check
            if (std::abs(deltaLambda)<_convTol) {

               printf("\nQNRLineSearch::applyLineSearch(): The Line Search"
                      " was successful and converges with %d iterations\n", k);

               break;
            }
         } 

         if(_isEvaluate)
            this->lineSearchEvaluation(phys, Xold, dX, lambda);       
      }
      
      // Returning results
      Xold.noalias() = X;     
      Gold.noalias() = G; 
      dX.noalias() = lambda*dX;
      this->_lambda = lambda;

      // Checking Wolfe-Powell linesearch conditions
      // realType Eend = phys->computeEnergy();
      
      // // Wolfe parameters
      // realType sigma1 = 0.1;
      // realType sigma2 = 0.9;

      // // i - Wolfe Condition
      // if(!(Eend <= Einit + sigma1*lambda*GoldtdX)) {
      //    printf("\nQNRLineSearch::applyLineSearch(): "
      //    "(i) Wolfe condition was not respected for lambda = %10.8e\n", lambda);
      // };
      // // ii - Wolfe Condition 
      // if(!(GtdX >= sigma2*GoldtdX)) {
      //    printf("\nQNRLineSearch::applyLineSearch(): "
      //    "(ii) Wolfe condition was not respected for lambda = %10.8e\n", lambda);
      // };

      return k;
   }
};


/**
 * @brief The backtracking line search for L-BFGS.
 * 
 */
template <typename Foo, typename realType = double>
class BacktrackingLineSearch: public LineSearch<Foo, realType> {
private:
   // Alias for a generic Eigen vector
   template<typename T>
   using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

   // Alias for a generic Eigen vector reference
   template<typename T>
   using RefVec = Eigen::Ref<Vec<T>>;

   int  _nItMax; 

public:
   BacktrackingLineSearch(): _nItMax(20) {
      this->_type = LineSearch<Foo, realType>::ELSBacktrac;
   };

   // Removing default copy constructor.
   BacktrackingLineSearch(const BacktrackingLineSearch &obj) = delete;

   // Destructor.
   ~BacktrackingLineSearch() = default;

   inline void setMaxNumIter(const int nIt) {_nItMax = nIt;}

   /**
    * @brief Compute the minimized line search parameter based on
    * quasi-newton method.
    * 
    * @param phys     
    *    Generic object to represent the problem to be solved.
    * @param Xold     
    *    Eigen vector to represent the initial solution.
    * @param dX
    *    Eigen vector to represent the incremental solution.
    * @param Gold
    *    Eigen vector to represent the last residual before 
    *    solution be updated. 
    * @return int 
    *    Number of iterations needed to compute the minimized 
    *    line search parameter. 
    */
   inline int applyLineSearch(Foo* phys, 
                              RefVec<realType> Xold, 
                              RefVec<realType> dX, 
                              RefVec<realType> Gold) override {

      // Decreasing and increasing factors
      const realType dec = 0.5;
      const realType inc = 2.1;

      // Initial step size
      realType lambda=1.0;
      
      // Computing Energy value at current solution
      const realType Einit = phys->computeEnergy();
      // Projection of gradient on the search direction
      const realType GoldTdX = Gold.dot(dX);
      // Checking if dX is not a descent direction
      if (GoldTdX > 0.0)
         throw std::logic_error("\nBacktrackingLineSearch::applyLineSearch(): The moving "
             "direction increases the objective function value.");   

      const realType sigma1 = 1e-4;
      const realType sigma2 = 0.9;
      realType scale;

      int k=1;                                    
      Vec<realType> G, X;
      G = Vec<realType>::Zero(Gold.size());
      for( ; ; k++) {   

         // Updating the current solution.
         X.noalias() = Xold + lambda*dX;
         
         // Updating physic solution: X = Xold + lambda*dX
         phys->setSolution(X);

         // Computing Energy value at current solution
         realType Ener = phys->computeEnergy();

         // Computing new residual: G(X) = G(Xold + lambda*dX)
         phys->computeGradient(G);

         // Armijo condition
         if(Ener > Einit + sigma1*lambda*GoldTdX) {
            scale = dec;
         } else {
            const realType GTdX = G.dot(dX);
            // Regular Wolfe condition 
            if(GTdX < sigma2*GoldTdX) {
               scale = inc;
            } else {
               // if(GTdX > -sigma2*GoldTdX){
               //    scale = dec;
               // } else {
               //    // Strong Wolfe condition
                  break;
               // }
            }
         }

         lambda *= scale;

         // Checking maximum number of iterations
         if (k >= this->_nItMax)
            throw std::runtime_error("\nBacktrackingLineSearch::applyLineSearch(): The line search routine reached the maximum number of iterations");
      } 

      printf("\nBacktrackingLineSearch::applyLineSearch(): The Line Search"
               " was successful and converges for lambda = %10.8e\n", lambda);

      Xold.noalias() = X;     
      Gold.noalias() = G; 
      dX.noalias() = lambda*dX;
      this->_lambda = lambda;

      return k;
   }
};

#endif // LINESEARCH_H