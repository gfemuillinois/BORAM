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
 * @brief A gradient-based Line Search procedure 
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

public:
   QNRLineSearch(): _convTol(1.0e-4), _actTol(1.0e-5), _nItMax(20) {
      this->_type = LineSearch<Foo, realType>::ELSQNR;
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

      // Computing first derivative of the potential energy 
      // in relation to the line search parameter (Projection of 
      // gradient on search direction).
      // E'(Xold + lambda * dX) = dE(Xold + lambda * dX)/dLambda 
      //                        = Grad(Xold + lambda * dX)^T * dX. 
      // GoldtdX = Gold^T * dX.
      GoldtdX = Gold.dot(dX);

      // Checking if dX is not a descent direction
      if (GoldtdX > 0.0) {
         printf("\nQNRLineSearch::applyLineSearch(): The moving direction "
             "increases the objective function value."
             "\n   Using direction of steepest descent.\n");
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
      }
      
      // Returning results
      Xold.noalias() = X;     
      Gold.noalias() = G; 
      dX.noalias() = lambda*dX;
      this->_lambda = lambda;

      return k;
   }
};

#endif // LINESEARCH_H