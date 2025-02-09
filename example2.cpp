/*
MIT License

© 2025 Nathan Shauer

phasefield-jr-boram

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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
std::vector<QuadraturePoint> create3x3QuadratureRule();

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
// =============================== GLOBAL VARIABLES ==============================
// ===============================================================================
// This is not ideal, but since this is a simple example, it is acceptable.
MatrixXd D = MatrixXd::Zero(3, 3);
double pseudotime = 0.;
std::string basefilename = "output_ex2_", vtkextension = ".vtk";
std::ofstream outpdelta("pdelta_ex2.txt");

// =============================== FUNCTION DECLARATIONS =========================
// ===============================================================================
void createDoubleNodeMesh(std::vector<Node> &nodes, std::vector<Element> &elements, int num_elements_x, int num_elements_y, double length, double height);
void shapeFunctions(MatrixXd &N, MatrixXd &dN, const double qsi, const double eta, const int nstate);
void createB(MatrixXd &B, const MatrixXd &dN);
void generateVTKLegacyFile(const std::vector<Node> &nodes, const std::vector<Element> &elements, const std::string &filename, const VectorXd &Uelas, const VectorXd &Upf);
void solveStep(const std::vector<Node> &nodes, const std::vector<Element> &elements, MaterialParameters &material, minSolver<Physics> &solver,
               const int maxiter, const double stagtol, const int step);
void solveStepHardCode(const std::vector<Node> &nodes, const std::vector<Element> &elements, MaterialParameters &material, minSolver<Physics> &solver,
                       const int maxiter, const double stagtol);  
class PhysicsElas;
void computeReaction(PhysicsElas *physics_elas, const std::vector<Node> &nodes, const std::vector<Element> &elements, MaterialParameters& mat, double& reaction);

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

  void setNodes(const std::vector<Node> &nodes) {
    _nodes = nodes;
  }

  void setElements(const std::vector<Element> &elements) {
    _elements = elements;
  }

  void setMaterial(const MaterialParameters &material) {
    _material = material;
  }

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

  // --- Access methods necessary for minSolver class ---
  inline const int getNumEquations() {
    return _ndofs;
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
  
  inline void getSolution(RefEigenVec<double> U) {U = _U;}
  inline void setSolution(const EigenVec<double> U) {_U = U;}

  inline void computeGradient(RefEigenVec<double> grad) {
    if(!_isKcomputed) {
      std::cerr << "Stiffness matrix has not been computed yet. Residual is calculated using R = K*u - F" << std::endl;
      throw std::exception();
    }
    grad = _K * _U - _F;
  }

  inline void computeJacobian() {
    assembleGlobalStiffness();
    factorize();
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

  inline void solve(RefEigenVec<double> R) {
    _llt.solveInPlace(R);  
    _U = R;
  }

  // Do we need this?
  inline void solveAnalysis() {
    computeJacobian();
    _U = _llt.solve(_F); 
  }

  double computeEnergy() {
    assembleGlobalStiffness(false);
    return 0.5 * _U.dot(_K * _U);
  }

};

class PhysicsElas : public Physics {
protected:
  std::vector<BC> _bc_nodes;

public:

  PhysicsElas() = default;
  virtual ~PhysicsElas() = default;

  virtual const int nState() const {return 2;}
  void setBCs(const std::vector<BC> &bc_nodes) {_bc_nodes = bc_nodes;}

  void applyBoundaryConditions() {
    for (const auto &bc : _bc_nodes) {
      int row = 2 * bc.node;
      double xval = bc.xval * pseudotime;
      double yval = bc.yval * pseudotime;
      if (bc.type == 0) {
        // Dirichlet in x and y
        _F -= _K.col(row) * xval;
        _F -= _K.col(row + 1) * yval;
        _K.row(row).setZero();
        _K.col(row).setZero();
        _K.row(row + 1).setZero();
        _K.col(row + 1).setZero();
        _K(row, row) = 1.0;
        _K(row + 1, row + 1) = 1.0;
        _F(row) = xval;
        _F(row + 1) = yval;
      } else if (bc.type == 1) {
        // Dirichlet in x
        _F -= _K.col(row) * xval;
        _K.row(row).setZero();
        _K.col(row).setZero();
        _K(row, row) = 1.0;
        _F(row) = xval;
      } else if (bc.type == 2) {
        // Dirichlet in y
        _F -= _K.col(row + 1) * yval;
        _K.row(row + 1).setZero();
        _K.col(row + 1).setZero();
        _K(row + 1, row + 1) = 1.0;
        _F(row + 1) = yval;
      } else if (bc.type == 3) {
        // Neumann
        _F(row) += xval;
        _F(row + 1) += yval;
      }
    }
  }

  virtual void assembleGlobalStiffness(const bool withBC = true) {
    Physics::assembleGlobalStiffness(withBC);
    if (withBC){
      applyBoundaryConditions();      
    }
    
  }
                                    

  virtual void computeElementStiffness(MatrixXd &Ke, VectorXd& Fe, const Element &element) {

    double detjac;
    MatrixXd J_inv;
    computeRequiredData(element, detjac, J_inv);

    MatrixXd B;
    for (const auto &qp : _intrule) {
      MatrixXd N, dN;
      shapeFunctions(N, dN, qp.xi, qp.eta, nState());

      // Transform derivatives to global coordinates
      MatrixXd dN_xy = J_inv.transpose() * dN.transpose();

      // Create B matrix
      createB(B, dN_xy.transpose());

      MatrixXd Ddeteriorated = D;
      // Interpolate phase field solution at quadrature points
      double phase_field = 0.0;
      for (int i = 0; i < 4; ++i) phase_field += N(0, 2 * i) * _p_otherPhysics->getU()(element.node_ids[i]);  // N is repeated for x and y so we multiply i by 2

      // Deteriorate the material properties based on the phase field
      Ddeteriorated *= (1 - phase_field) * (1 - phase_field);

      // Compute element stiffness matrix contribution of this integration point
      Ke += B.transpose() * Ddeteriorated * B * qp.weight * detjac;
    }
  }

  void createB(MatrixXd &B, const MatrixXd &dN) {
    B = MatrixXd::Zero(3, 8);
    for (int i = 0; i < 4; ++i) {
      B(0, 2 * i) = dN(i, 0);
      B(1, 2 * i + 1) = dN(i, 1);
      B(2, 2 * i) = dN(i, 1);
      B(2, 2 * i + 1) = dN(i, 0);
    }
  }

  VectorXd &computeSigmaAtCenter(const Element &element, VectorXd &stress_vec) {
    // Calculate the derivative of the elastic solution using dN and Uelas
    double qsi = 0., eta = 0.;  // center of element

    double detjac;
    MatrixXd J_inv;
    computeRequiredData(element, detjac, J_inv);

    MatrixXd N, dN;
    shapeFunctions(N, dN, qsi, eta, 2);
    MatrixXd dN_xy = J_inv.transpose() * dN.transpose();

    MatrixXd dU = MatrixXd::Zero(2, 2);
    for (int i = 0; i < 4; ++i) {
      int index = 2 * element.node_ids[i];
      dU(0, 0) += dN_xy(0, i) * _U(index);      // duxdx
      dU(0, 1) += dN_xy(1, i) * _U(index);      // duxdy
      dU(1, 0) += dN_xy(0, i) * _U(index + 1);  // duydx
      dU(1, 1) += dN_xy(1, i) * _U(index + 1);  // duydy
    }

    // Calculate strain tensor
    MatrixXd strain = 0.5 * (dU + dU.transpose());

    // Convert strain to a vector
    VectorXd strain_vec(3);
    strain_vec << strain(0, 0), strain(1, 1), 2 * strain(0, 1);  // note the times 2 in the off-diagonal term

    double phase_field = 0.0;
    for (int i = 0; i < 4; ++i) phase_field += N(0, 2 * i) * _p_otherPhysics->getU()(element.node_ids[i]);  // N is repeated for x and y so we multiply by 2

    // Calculate stress
    double g = (1. - phase_field) * (1. - phase_field);
    stress_vec = g * D * strain_vec;

    return stress_vec;
  }
};        

class PhysicsPF : public Physics {
public:
  PhysicsPF() = default;
  virtual ~PhysicsPF() = default;

  virtual const int nState() const {return 1;}

  virtual void computeElementStiffness(MatrixXd &Ke, VectorXd& Fe, const Element &element) {

    double detjac;
    MatrixXd J_inv;
    computeRequiredData(element, detjac, J_inv);

    double G = _material.G, l = _material.l;
    double c0 = 2.;

    for (const auto &qp : _intrule) {
      MatrixXd N, dN;
      shapeFunctions(N, dN, qp.xi, qp.eta, nState());

      // Transform derivatives to global coordinates
      MatrixXd dN_xy = J_inv.transpose() * dN.transpose();

      double sigmaDotEps = calculateSigmaDotEps(element, dN_xy);

      for (int i = 0; i < 4; ++i) {
        Fe[i] += detjac * qp.weight * 0.5 * sigmaDotEps * N(0, i);
        for (int j = 0; j < 4; ++j) {
          Ke(i, j) += detjac * qp.weight * (G * l / c0 * (dN_xy(0, i) * dN_xy(0, j) + dN_xy(1, i) * dN_xy(1, j)) + (G / (l * c0) + 0.5*sigmaDotEps) * N(0, j) * N(0, i));
        }
      }
    }
  }

  double calculateSigmaDotEps(const Element &element, const MatrixXd &dN) {
    double sigmaDotEps = 0.;
    // Calculate the derivative of the elastic solution using dN and Uelas
    MatrixXd dU = MatrixXd::Zero(2, 2);
    for (int i = 0; i < 4; ++i) {
      int index = 2 * element.node_ids[i];
      dU(0, 0) += dN(0, i) * _p_otherPhysics->getU()(index);      // duxdx
      dU(0, 1) += dN(1, i) * _p_otherPhysics->getU()(index);      // duxdy
      dU(1, 0) += dN(0, i) * _p_otherPhysics->getU()(index + 1);  // duydx
      dU(1, 1) += dN(1, i) * _p_otherPhysics->getU()(index + 1);  // duydy
    }

    // Calculate strain tensor
    MatrixXd strain = 0.5 * (dU + dU.transpose());

    // Convert strain to a vector
    VectorXd strain_vec(3);
    strain_vec << strain(0, 0), strain(1, 1), 2 * strain(0, 1);  // note the times 2 in the off-diagonal term

    // Calculate stress
    VectorXd stress_vec = D * strain_vec;

    // Calculate sigma dot epsilon
    sigmaDotEps = stress_vec.dot(strain_vec);
    // std::cout << "Sigma dot Epsilon: " << std::scientific << sigmaDotEps << std::endl;

    return sigmaDotEps;
  }
};

//-------------------------------------------------------------------------------------------------
//   __  __      _      _   _   _
//  |  \/  |    / \    | | | \ | |
//  | |\/| |   / _ \   | | |  \| |
//  | |  | |  / ___ \  | | | |\  |
//  |_|  |_| /_/   \_\ |_| |_| \_|
//-------------------------------------------------------------------------------------------------
int main() {
  Timer simulation_time;
  // Define material properties
  double E = 210;    // Young's modulus in Pascals
  double nu = 0.3;  // Poisson's ratio
  double G = 2.7e-3;  // Strain energy release rate
  double l = 0.005; // Length scale parameter

  // Define mesh and time step parameters
  int num_elements_x = 60;
  int num_elements_y = 30; // has to be even number
  double length = 1.;
  double height = 1.;
  double dt = 0.01;
  double totaltime = 0.8;
  int maxsteps = 1e5; // maximum number of time steps (in case using adptative time step)
  int maxiter = 1000; // maximum number of iterations for the staggered scheme
  double stagtol = 1e-7; // tolerance to consider the staggered scheme converged  

  // Boundary conditions
  double imposed_displacement_y = 0.01; // such that we have nucleation at step 50

  // Data structures
  std::vector<Node> nodes;
  std::vector<Element> elements;
  std::vector<BC> bc_nodes;

  // Create material parameters struct and D matrix (assumed same for all elements)
  MaterialParameters material = {E, nu, G, l};
  double factor = E / (1 - nu * nu);  
  D(0, 0) = factor;
  D(0, 1) = factor * nu;
  D(1, 0) = factor * nu;
  D(1, 1) = factor;
  D(2, 2) = factor * (1 - nu) / 2.0;  

  createDoubleNodeMesh(nodes, elements, num_elements_x, num_elements_y, length, height);

  // Create boundary conditions  
  bool firstnode = true;
  for (int i = 0; i < nodes.size(); ++i) {
    if (fabs(nodes[i].y + 0.5) < 1.e-8) {
      if (firstnode) {
        bc_nodes.push_back({i, 0, 0.0, 0.0}); // Fix x and y displacement
        firstnode = false;
      }
      else bc_nodes.push_back({i, 2, 0.0, 0.0}); // Fix y displacement
    } else if (fabs(nodes[i].y - 0.5) < 1.e-8) {
      bc_nodes.push_back({i, 2, 0., imposed_displacement_y}); // Impose total y displacement on the top edge
    }
  }  

  // Initialize global stiffness matrix and force vector
  int nstate_elas = 2, nstate_pf = 1;
  int ndofs_elas = nstate_elas * nodes.size(), ndofs_pf = nstate_pf * nodes.size();

  PhysicsElas *physics_elas = new PhysicsElas;
  PhysicsPF *physics_pf = new PhysicsPF;
  physics_elas->setOtherPhysics(physics_pf);
  physics_pf->setOtherPhysics(physics_elas);
  physics_elas->initializeStructures(ndofs_elas);
  physics_pf->initializeStructures(ndofs_pf);
  physics_elas->setBCs(bc_nodes);
  alternMinSolver<Physics> mySolver;
  // LBFGSSolver<Physics> mySolver;  
  mySolver.setMaxNumIter(1000);
  mySolver.addPhysics(physics_elas);
  mySolver.addPhysics(physics_pf);
  mySolver.monitorName("AM_monitor.txt");

  // Set nodes, elements, and material for each physics
  for (auto &phys : mySolver.getPhysics()) {
    phys->setNodes(nodes);
    phys->setElements(elements);
    phys->setMaterial(material);
  }

  VectorXd residual;
  for (int step = 0; step < maxsteps; ++step) {    
    pseudotime += dt;
    if(pseudotime > totaltime) break;
    for (auto &phys : mySolver.getPhysics()) {phys->setStep(step);}
    std::cout << "******************** Time Step " << step << " | Pseudo time = " << std::fixed << std::setprecision(6) << pseudotime << " | Time step = " << dt << " ********************" << std::endl;
    // solveStepHardCode(nodes, elements, material, mySolver, maxiter, stagtol);
    solveStep(nodes, elements, material, mySolver, maxiter, stagtol, step);    
    
    std::string filename = basefilename + std::to_string(step) + vtkextension;
    generateVTKLegacyFile(nodes, elements, filename, physics_elas->getU(), physics_pf->getU());

    double reaction = 0.;
    computeReaction(physics_elas, nodes, elements, material, reaction);
    outpdelta << pseudotime*imposed_displacement_y << " " << reaction << std::endl;
  }

  std::cout << std::endl << "================> Simulation completed!" << std::endl;
  simulation_time.elapsed("complete simulation");
  return 0;
}

// =============================== FUNCTION IMPLEMENTATIONS ======================
// ===============================================================================
void createDoubleNodeMesh(std::vector<Node> &nodes, std::vector<Element> &elements, int num_elements_x, int num_elements_y, double length, double height) {
  double xsize = 0.001, ysize = 0.001;
  num_elements_y /= 2;
  int num_elements_x_small = 8, num_elements_y_small = 8;
  int num_elements_y_large = num_elements_y - num_elements_y_small;
  int num_elements_x_large = num_elements_x - num_elements_x_small;
  double y_small = ysize * num_elements_y_small;
  double y_largeel_size = (height/2 - y_small) / num_elements_y_large;
  double x_small = xsize * num_elements_x_small;
  double x_largeel_size = (length - x_small) / num_elements_x_large;

  for (int j = 0; j <= num_elements_y; ++j) {
    double xnow = -0.5;
    for (int i = 0; i <= num_elements_x; ++i) {
      double y;
      if (j <= num_elements_y_small) {
        y = j * ysize;
      } else {
        y = y_small + (j - num_elements_y_small) * y_largeel_size;
      }
      nodes.push_back({xnow, y});
      if (i < num_elements_x_large / 2 || i > num_elements_x_large / 2 + num_elements_x_small - 1) {
        xnow += x_largeel_size;
      } else {
        xnow += xsize;
      }
    }
  }

  // Generate elements
  for (int j = 0; j < num_elements_y; ++j) {
    for (int i = 0; i < num_elements_x; ++i) {
      int n1 = j * (num_elements_x + 1) + i;
      int n2 = n1 + 1;
      int n3 = n1 + num_elements_x + 1;
      int n4 = n3 + 1;
      elements.push_back({{n1, n2, n4, n3}});
    }
  }

  // Mirror the mesh to negative y
  int original_node_count = nodes.size();
  std::unordered_map<int, int> node_map;

  for (int i = 0; i < original_node_count; ++i) {
    if (fabs(nodes[i].y) > 1.e-8) {
      Node mirrored_node = nodes[i];
      mirrored_node.y = -mirrored_node.y;
      node_map[i] = nodes.size();
      nodes.push_back(mirrored_node);
    } else {
      if (nodes[i].x < -1.e-8) { // nodes until the tip
        Node duplicated_node = {nodes[i].x, nodes[i].y};
        node_map[i] = nodes.size();
        nodes.push_back(duplicated_node);
      } else {
        node_map[i] = i;
      }
    }
  }

  // Generate elements for the mirrored part
  int original_element_count = elements.size();
  for (int i = 0; i < original_element_count; ++i) {
    Element mirrored_element = elements[i];
    for (int &node_id : mirrored_element.node_ids) {
      node_id = node_map[node_id];
    }
    std::reverse(mirrored_element.node_ids.begin(), mirrored_element.node_ids.end());
    elements.push_back(mirrored_element);
  }
}

void shapeFunctions(MatrixXd &N, MatrixXd &dN, const double qsi, const double eta, const int nstate) {
  double phi1qsi = (1 + qsi) / 2.0;
  double phi0eta = (1 - eta) / 2.0;
  double phi1eta = (1 + eta) / 2.0;
  double phi0qsi = (1 - qsi) / 2.0;

  Vector4d shape;
  shape << phi0qsi * phi0eta,
      phi1qsi * phi0eta,
      phi1qsi * phi1eta,
      phi0qsi * phi1eta;

  N = MatrixXd::Zero(2, nstate*4);
  if (nstate == 1) {
    for (int i = 0; i < 4; ++i) {
      N(0, i) = shape(i);
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      N(0, 2 * i) = shape(i);
      N(1, 2 * i + 1) = shape(i);
    }
  }

  dN.resize(4, 2);
  dN << 0.25 * (-1 + eta), 0.25 * (-1 + qsi),
      0.25 * (1 - eta), 0.25 * (-1 - qsi),
      0.25 * (1 + eta), 0.25 * (1 + qsi),
      0.25 * (-1 - eta), 0.25 * (1 - qsi);
}

void createB(MatrixXd &B, const MatrixXd &dN) {
  B = MatrixXd::Zero(3, 8);
  for (int i = 0; i < 4; ++i) {
    B(0, 2 * i) = dN(i, 0);
    B(1, 2 * i + 1) = dN(i, 1);
    B(2, 2 * i) = dN(i, 1);
    B(2, 2 * i + 1) = dN(i, 0);
  }
}


void computeReaction(PhysicsElas *physics_elas, const std::vector<Node> &nodes, const std::vector<Element> &elements, MaterialParameters& mat, double& reaction) {
  
  physics_elas->assembleGlobalStiffness(false);
  VectorXd residual = physics_elas->getK() * physics_elas->getU(); // F is zero
  reaction = 0.;
  for (int i = 0; i < nodes.size(); ++i) {
    if (fabs(nodes[i].y + 0.5) < 1.e-8) {
      reaction += -residual(2 * i + 1);
    }
  }
}

void generateVTKLegacyFile(const std::vector<Node> &nodes, const std::vector<Element> &elements, const std::string &filename, const VectorXd& Uelas, const VectorXd& Upf) {
  std::ofstream vtkFile(filename);  

  vtkFile << "# vtk DataFile Version 2.0\n";
  vtkFile << "FEM results\n";
  vtkFile << "ASCII\n";
  vtkFile << "DATASET UNSTRUCTURED_GRID\n";

  // Write points
  vtkFile << "POINTS " << nodes.size() << " float\n";
  for (const auto &node : nodes) {
    vtkFile << node.x << " " << node.y << " 0.0\n";
  }

  // Write cells
  vtkFile << "CELLS " << elements.size() << " " << elements.size() * 5 << "\n";
  for (const auto &element : elements) {
    vtkFile << "4 " << element.node_ids[0] << " " << element.node_ids[1] << " " << element.node_ids[2] << " " << element.node_ids[3] << "\n";
  }

  // Write cell types
  vtkFile << "CELL_TYPES " << elements.size() << "\n";
  for (size_t i = 0; i < elements.size(); ++i) {
    vtkFile << "9\n";  // VTK_QUAD
  }

  // Write point data (displacements)
  vtkFile << "POINT_DATA " << nodes.size() << "\n";
  vtkFile << "VECTORS displacements float\n";
  for (size_t i = 0; i < nodes.size(); ++i) {
    vtkFile << Uelas(2 * i) << " " << Uelas(2 * i + 1) << " 0.0\n";
  }

  // Write point data (phase field)
  vtkFile << "SCALARS phasefield float 1\n";
  vtkFile << "LOOKUP_TABLE default\n";
  for (size_t i = 0; i < nodes.size(); ++i) {
    vtkFile << Upf(i) << "\n";
  }

  vtkFile.close();
}

void solveStepHardCode(const std::vector<Node> &nodes, const std::vector<Element> &elements, MaterialParameters &material, minSolver<Physics> &solver,
               const int maxiter, const double stagtol) {
  int iter = 0;  
  PhysicsElas *physics_elas = dynamic_cast<PhysicsElas*>(solver.getPhysics()[0]);
  PhysicsPF *physics_pf = dynamic_cast<PhysicsPF*>(solver.getPhysics()[1]);

  VectorXd residual = VectorXd::Zero(physics_elas->getNumEquations());
  VectorXd residualPF = VectorXd::Zero(physics_pf->getNumEquations());
  for (iter = 0; iter < maxiter; iter++) {
    std::cout << "------ Staggered Iteration " << iter << " ------" << std::endl;
    // Solve elasticity problem
    physics_elas->assembleGlobalStiffness();
    double norm;
    if (iter != 0) {
      physics_elas->computeGradient(residual);
      norm = residual.norm();
      std::cout << "Residual Elasticity Norm: " << std::scientific << std::setprecision(2) << norm << std::endl;
    }
    if (iter != 0 && norm < stagtol) {
      std::cout << "------> Staggered scheme converged in " << iter << " iterations." << std::endl;
      break;
    }

    physics_elas->factorize();
    physics_elas->solve(physics_elas->getF());

    // Solve phase field problem
    physics_pf->assembleGlobalStiffness();
    physics_pf->factorize();
    physics_pf->solve(physics_pf->getF());    
  }
  if (iter == maxiter) {
    std::cout << "------> Staggered scheme did not converge in " << maxiter << " iterations." << "\nAccepting current solution and continuing" << std::endl;
  }
}

void solveStep(const std::vector<Node> &nodes, const std::vector<Element> &elements, MaterialParameters &material, minSolver<Physics> &solver,
               const int maxiter, const double stagtol, const int step) {

  solver.monitorTimeRestart();
  std::vector<Physics*> phys = solver.getPhysics();
  std::vector<EigenVec<double>> resVecs(phys.size()), solVecs(phys.size()), dsolVecs(phys.size());

  int nEq = 0, initPos = 0, physIndex = 0;
  std::vector<Physics*>::iterator it;
  for(it = phys.begin(); it != phys.end(); it++) {

    int physDim = (*it)->getNumEquations();
    nEq += physDim;

    resVecs[physIndex].resize(physDim);
    solVecs[physIndex].resize(physDim);
    dsolVecs[physIndex].resize(physDim);

    (*it)->getSolution(dsolVecs[physIndex]);
    (*it)->assembleGlobalStiffness(false);
    (*it)->computeGradient(resVecs[physIndex]);
    (*it)->solveAnalysis();    
    (*it)->getSolution(solVecs[physIndex]);

    initPos += physDim;
    physIndex++;
  }

  // Configuring global vectors to be used in the external library
  EigenVec<double> sol = EigenVec<double>::Zero(nEq);
  EigenVec<double> dsol = EigenVec<double>::Zero(nEq);
  EigenVec<double> res = EigenVec<double>::Zero(nEq);

  // Initializing the global vector.
  initPos = 0, physIndex = 0;
  for(it = phys.begin(); it != phys.end(); it++) {
    
    // Number of degrees of freedom of each physics.
    int physDim = (*it)->getNumEquations();

    sol.segment(initPos, physDim) = solVecs[physIndex];
    dsol.segment(initPos, physDim) = dsolVecs[physIndex];
    res.segment(initPos, physDim) = resVecs[physIndex];
    
    // Updating global vector mapping for next physics.
    initPos += physDim;  
    physIndex++;
  }  

  // Computing the increment solution dX = X - Xold
  dsol.noalias() = sol - dsol;

  // Initializing the convergence criteria with the first 
  // residual (R_0) and increment BCs vector. 
  solver.initializeConvCriterion(res, dsol);

  // Updating the residual vector (R_1)
  solver.computeGradient(res);

  // Here we are just passing the information of the step to be 
  // solved to the convergence monitor of the external library, 
  // so we can have a nice view of the results of each step.
  std::string aux;
  int skip;
  std::stringstream line;
  aux = "S T E P"; skip = 37 + aux.size();
  line << std::setw(skip) << aux << std::setw(54 - skip) 
        << std::to_string(step+1) << "\n\n";
  solver.printOnConvMonitor(line.str());

  // Calling solver algorithm 
  int iSolve = solver.solveAnalysis(sol, dsol, res);

  if(iSolve < 0) {
    std::cout << "\nWARNING! Staggered scheme did not converge!!!" << std::endl;
    std::cout << "Accepting current solution and continuing" << std::endl;
  }
        
  // The problem converged as a linear problem
  if(iSolve==0) iSolve++;  

  PhysicsElas *physics_elas = dynamic_cast<PhysicsElas*>(solver.getPhysics()[0]);
}

// =============================== QUADRADURE RULES ==============================
// ===============================================================================
std::vector<QuadraturePoint> create2x2QuadratureRule() {
  // 2-point Gaussian quadrature positions and weights
  const double points[2] = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
  const double weights[2] = {1.0, 1.0};

  std::vector<QuadraturePoint> rule;

  // Create 4 quadrature points (2x2 grid)
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      QuadraturePoint qp;
      qp.xi = points[i];
      qp.eta = points[j];
      qp.weight = weights[i] * weights[j];
      rule.push_back(qp);
    }
  }

  return rule;
}

std::vector<QuadraturePoint> create3x3QuadratureRule() {
  // 3-point Gaussian quadrature weights and positions
  const double points[3] = {-std::sqrt(3.0 / 5.0), 0.0, std::sqrt(3.0 / 5.0)};
  const double weights[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};

  std::vector<QuadraturePoint> rule;

  // Create 9 quadrature points
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      QuadraturePoint qp;
      qp.xi = points[i];
      qp.eta = points[j];
      qp.weight = weights[i] * weights[j];
      rule.push_back(qp);
    }
  }

  return rule;
}