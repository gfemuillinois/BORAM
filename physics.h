

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include "am.h"
#include "lbfgs.h"

using namespace Eigen;

// =============================== DATA STRUCTURES ===============================
// ===============================================================================
// Struct to represent a quadrature point
struct QuadraturePoint {
  double xi;      // xi coordinate
  double eta;     // eta coordinate
  double weight;  // weight associated with this point
};
struct MaterialParameters {
  double E;  // Young's modulus
  double nu; // Poisson's ratio
  double G;  // Strain energy release rate
  double l;  // Length scale parameter
};
std::vector<QuadraturePoint> create2x2QuadratureRule();
std::vector<QuadraturePoint> create2x2QuadratureRule();

struct Element {
  std::vector<int> node_ids;
};
struct Node {
  double x, y;
};

struct BC {
  int node;
  int type; // 0 dirichlet in x and y, 1 dirichlet in x, 2 dirichlet in y, 3 neumann
  double xval,yval;
};

struct Timer {
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<float> duration;

  Timer() {
    start = std::chrono::high_resolution_clock::now();
  }

  void elapsed(std::string message = "") {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    float seconds = duration.count();
    std::cout << "Timer for " << message << " took " << std::fixed << std::setprecision(1) << seconds << " seconds" << std::endl;
  }
};

// =============================== Physics Classes - Necessary for Boram solver =========================
// ====================================================================================================
/* 
* Physics class is a base class that contains the main methods that are necessary for the minSolver class
* to work. The Physics class is an abstract class, meaning that it has at least one pure virtual function.
* This means that the Physics class cannot be instantiated, but it can be used as a base class for other
* classes that inherit from it. In this case, we have two classes that inherit from Physics: PhysicsElas and
* PhysicsPF. These classes implement the pure virtual functions of the Physics class.
*/
class Physics{
protected:
    std::vector<Node> _nodes;
    std::vector<Element> _elements;
    MaterialParameters _material;

    Eigen::LLT<MatrixXd> _llt; // Factorized stiffness matrix
    MatrixXd _K; // Stiffness matrix
    VectorXd _U; // Solution vector
    VectorXd _F; // Force vector
    std::vector<QuadraturePoint> _intrule = create2x2QuadratureRule(); // Adopting 2x2 quadrature rule
    Physics* _p_otherPhysics = nullptr; // Pointer to the other physics class (Elas or PF)
    int _ndofs = 0;
    bool _isKcomputed = false;
    bool _isDecomposed = false;
    int _step = -1;

public:
    // Constructor
    Physics() = default;
    Physics(const Physics &obj) = delete; // not allowing copy construct
    virtual ~Physics() = default;

    void setNodes(const std::vector<Node> &nodes) {_nodes = nodes;}

    void setElements(const std::vector<Element> &elements) { _elements = elements;}

    void setMaterial(const MaterialParameters &material) {_material = material;}

    inline void setOtherPhysics(Physics* p_otherPhysics) {_p_otherPhysics = p_otherPhysics;}
    inline void setStep(int step) {
        if(step <= _step){
            std::cerr << "Step must be greater than the current step" << std::endl;
            throw std::exception();
        }
        _isKcomputed = false; // we will have to update K
        _step = step;
    }

    // --- Helper methods specific to this implementation of the Physics class ---
    inline void initializeStructures(const int ndofs) {
        _K = MatrixXd::Zero(ndofs, ndofs);
        _U = VectorXd::Zero(ndofs);
        _F = VectorXd::Zero(ndofs);
        _ndofs = ndofs;
    }

    inline void zeroLLT() { 
        _llt = Eigen::LLT<MatrixXd>();
    }

    inline MatrixXd& getK() {return _K;}
    inline VectorXd& getF() {return _F;}
    inline VectorXd& getU() {return _U;}

    virtual void computeElementStiffness(MatrixXd &Ke, VectorXd& Fe, const Element &element) = 0; // implemented in son classes
    virtual const int nState() const = 0; // implemented in son classes

    virtual void assembleGlobalStiffness(const bool withBC = true) {
        Timer time;

        _K.setZero();
        _F.setZero();
        const int nstate = nState();
        for (const auto &element : _elements) {
            // Element stiffness matrix (for simplicity, assume a 4-node quadrilateral element)
            const int nquadnodes = 4;
            const int ndofel = nstate * nquadnodes;
            MatrixXd Ke = MatrixXd::Zero(ndofel, ndofel);
            VectorXd Fe = VectorXd::Zero(ndofel);
            this->computeElementStiffness(Ke, Fe, element);

            // Assemble Ke into the global stiffness matrix K
            for (int i = 0; i < nquadnodes; ++i) {
            int row = nstate * element.node_ids[i];
            for (int k = 0; k < nstate; ++k) {
                _F[row + k] += Fe[nstate * i + k];
            }
            for (int j = 0; j < nquadnodes; ++j) {
                int col = nstate * element.node_ids[j];
                for (int k = 0; k < nstate; ++k) {
                for (int l = 0; l < nstate; ++l) {
                    _K(row + k, col + l) += Ke(nstate * i + k, nstate * j + l);
                }
                }
            }
            }
        }
        _isKcomputed = true;
        _isDecomposed = false;
        // time.elapsed("assembly");
    }

    inline void computeRequiredData(const Element &element, double& detjac, MatrixXd& J_inv) {
        const Node &n1 = _nodes[element.node_ids[0]];
        const Node &n2 = _nodes[element.node_ids[1]];
        const Node &n3 = _nodes[element.node_ids[2]];
        const Node &n4 = _nodes[element.node_ids[3]];

        // Jacobian matrix
        double base = n2.x - n1.x;
        double height = n4.y - n1.y;
        double area = base * height;  // base * height since it is a simple mesh
        detjac = area / 4.0;
        double dqsidx = 2.0 / base;    // for simple rectangular elements
        double dqsidy = 2.0 / height;  // for simple rectangular elements

        J_inv = MatrixXd::Zero(2, 2);
        J_inv(0, 0) = dqsidx;
        J_inv(1, 1) = dqsidy;
        }

    inline void factorize() {
        if(!_isDecomposed) zeroLLT();
        if(!_llt.cols()){ // If the factorization has not been done yet
            _llt.compute(_K); // do factorization
            if(_llt.info() != Eigen::Success) {
            std::cerr << "Error during the factorization of the stiffness matrix" << std::endl;
            throw std::exception();
            }
            _isDecomposed = true;
        }
    }  

    double computeEnergy() {
        assembleGlobalStiffness(false);
        return 0.5 * _U.dot(_K * _U);
    }

    // --------------------------------------------------
    // --- Access methods necessary for BORAM library ---
    // --------------------------------------------------
    inline const int getNumEquations() {return _ndofs;}
    inline void getSolution(RefEigenVec<double> U) {U = _U;}
    inline void setSolution(const EigenVec<double> U) {_U = U;}

    inline void computeGradient(RefEigenVec<double> grad) {   
        assembleGlobalStiffness();
        grad = _K * _U - _F;
    }

    inline void computeJacobian() {
        assembleGlobalStiffness();
        factorize();
    }

    inline void solve(RefEigenVec<double> R) {
        _llt.solveInPlace(R);  
        _U = R;
    }

    // Do we need this?
    inline void solveAnalysis() {
        computeJacobian();
        _U = _llt.solve(_F); 
    }
        
};
    
