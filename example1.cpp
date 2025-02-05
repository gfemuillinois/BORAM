/*
MIT License

© 2023 Nathan Shauer

phasefield-jr

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

class Physics{
private:
  Eigen::LLT<MatrixXd> _llt; // Factorized stiffness matrix
  MatrixXd _K; // Stiffness matrix
  VectorXd _U; // Solution vector
  VectorXd _F; // Force vector

public:
  // Constructor
  Physics() = default;
  Physics(const Physics &obj) = delete; // not allowing copy construct
  ~Physics() = default;

  inline const int getNumEquations() {
    if (_K.rows() == 0) {
      std::cerr << "Stiffness matrix has not been initialized" << std::endl;
      throw std::exception();
    }
    return _K.rows();
  }
  
  inline void getSolution(RefEigenVec<double> U) {U = _U;}
  inline void setSolution(const EigenVec<double> U) {_U = U;}

  inline void computeGradient(RefEigenVec<double> grad) {
    grad = _K * _U - _F;
  }

  inline void computeJacobian() {
    if(!_llt.cols()){ // If the factorization has not been done yet
      _llt.compute(_K); // do factorization
      if(_llt.info() != Eigen::Success) {
        std::cerr << "Error during the factorization of the stiffness matrix" << std::endl;
        throw std::exception();
      }
    }
  }

  inline void solve(RefEigenVec<double> R) {
    _llt.solveInPlace(R);  
    _U = R;
  }

  // Do we need this?
  // inline void solveAnalysis() {
  //   const EPhysType physType = _p_mPhys->type(_shp_Analys->compMesh().get());
  //   _p_mPhys->setActiveProblemPhys(physType);

  //   _p_mPhys->assembleSolveMPhys(_step, physType);

  // }

  double computeEnergy() {
    throw std::runtime_error("computeEnergy() is not implemented.");
  }

};

// =============================== GLOBAL VARIABLES ==============================
// ===============================================================================
// This is not ideal, but since this is a simple example, it is acceptable.
VectorXd Uelas;
VectorXd Upf;
MatrixXd D = MatrixXd::Zero(3, 3);
double pseudotime = 0.;
std::string basefilename = "output_ex1_", vtkextension = ".vtk";
std::ofstream outpdelta("pdelta_ex1.txt");
std::vector<QuadraturePoint> intrule = create2x2QuadratureRule(); // Adopting 2x2 quadrature rule
Eigen::LLT<MatrixXd> lltOfKelas, lltOfKpf;

// =============================== FUNCTION DECLARATIONS =========================
// ===============================================================================

void createRectangularMesh(std::vector<Node> &nodes, std::vector<Element> &elements, int num_elements_x, int num_elements_y, double length, double height);
void assembleGlobalStiffness(MatrixXd &K, VectorXd& F, const std::vector<Element> &elements, const std::vector<Node> &nodes, MaterialParameters& material, int nstate);
void computeElementStiffness(MatrixXd &Ke, VectorXd& Fe, const std::vector<Node> &nodes, const Element &element, MaterialParameters& material, const int nstate);
void shapeFunctions(MatrixXd &N, MatrixXd &dN, const double qsi, const double eta, const int nstate);
void createB(MatrixXd &B, const MatrixXd &dN);
void applyBoundaryConditions(MatrixXd &K, VectorXd &F, const std::vector<BC> &bc_nodes);
void solveSystem(const MatrixXd &K, const VectorXd &F, VectorXd &U, const int nstatevar);
void generateVTKLegacyFile(const std::vector<Node> &nodes, const std::vector<Element> &elements, const std::string &filename);
double calculateSigmaDotEps(const Element &element, const MatrixXd &dN, MaterialParameters& mat);
VectorXd& computeSigmaAtCenter(const Element &element, std::vector<Node> &nodes, VectorXd& stress_vec);

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
  double E = 30;    // Young's modulus in Pascals
  double nu = 0.2;  // Poisson's ratio
  double G = 1.2e-4;  // Strain energy release rate
  double l = 10.; // Length scale parameter

  // Define mesh and time step parameters
  int num_elements_x = 50; // number of elements in x direction
  int num_elements_y = 5; // number of elements in y direction
  double length = 200.; // length of the domain
  double height = 20.; // height of the domain
  double dt = 0.02; // pseudo time step
  double totaltime = 1.5; // total simulation time
  int maxsteps = 1e5; // maximum number of time steps (in case using adptative time step)
  int maxiter = 600; // maximum number of iterations for the staggered scheme
  double stagtol = 1e-7; // tolerance to consider the staggered scheme converged  

  // Boundary conditions
  double sigma_peak_at2 = sqrt(27. * E * G / (256. * l));
  double u_peak_at2 = 16./9. * sigma_peak_at2 * length / E;
  std::cout << "Sigma peak: " << sigma_peak_at2 << std::endl;
  std::cout << "U peak: " << u_peak_at2 << std::endl;
  double imposed_displacement_x = u_peak_at2; // such that we have nucleation at step 50

  // Data structures
  std::vector<Node> nodes;
  std::vector<Element> elements;
  std::vector<BC> bc_nodes;

  // Create material parameters struct and D matrix (assumed same for all elements and therefore a global variable)
  MaterialParameters material = {E, nu, G, l};
  double factor = E / (1 - nu * nu);  
  D(0, 0) = factor;
  D(0, 1) = factor * nu;
  D(1, 0) = factor * nu;
  D(1, 1) = factor;
  D(2, 2) = factor * (1 - nu) / 2.0;  

  // Create regular rectangular mesh
  createRectangularMesh(nodes, elements, num_elements_x, num_elements_y, length, height);

  // Create boundary conditions  
  for (int i = 0; i < nodes.size(); ++i) {
    if (fabs(nodes[i].x) < 1.e-8) {
      bc_nodes.push_back({i, 0, 0.0, 0.0});     // Fix x displacement
    } else if (fabs(nodes[i].x - length) < 1.e-8) {
      bc_nodes.push_back({i, 1, imposed_displacement_x, 0.0});     // Impose total x displacemente on the right edge
    }
  }  

  // Initialize global stiffness matrix and force vector
  int nstate_elas = 2, nstate_pf = 1;
  int ndofs_elas = nstate_elas * nodes.size(), ndofs_pf = nstate_pf * nodes.size();
  MatrixXd Kelas(ndofs_elas, ndofs_elas);  
  VectorXd Felas = VectorXd::Zero(ndofs_elas);
  Uelas = VectorXd::Zero(ndofs_elas);
  
  MatrixXd Kpf = MatrixXd::Zero(ndofs_pf, ndofs_pf);
  VectorXd Fpf = VectorXd::Zero(ndofs_pf);
  Upf = VectorXd::Zero(ndofs_pf);

  VectorXd residual;
  for (int step = 0; step < maxsteps; ++step) {    
    pseudotime += dt;
    if(pseudotime > totaltime) break;
    std::cout << "******************** Time Step " << step << " | Pseudo time = " << std::fixed << std::setprecision(6) << pseudotime << " | Time step = " << dt << " ********************" << std::endl;
    int iter = 0;
    for(iter = 0 ; iter < maxiter ; iter++){
      std::cout << "------ Staggered Iteration " << iter << " ------" << std::endl;
      // Solve elasticity problem
      assembleGlobalStiffness(Kelas, Felas, elements, nodes, material, nstate_elas);
      applyBoundaryConditions(Kelas, Felas, bc_nodes);
      double norm;
      if(iter != 0){
        residual = Kelas * Uelas - Felas; // checking if last U satisfies the equilibrium with updated phase field      
        norm = residual.norm();
        std::cout << "Residual Elasticity Norm: " << std::scientific << std::setprecision(2) << norm << std::endl;        
      }
      if (iter != 0 && norm < stagtol) {
        std::cout << "------> Staggered scheme converged in " << iter << " iterations." << std::endl;
        break;
      }            
      
      lltOfKelas = Eigen::LLT<MatrixXd>();
      lltOfKpf = Eigen::LLT<MatrixXd>();
      solveSystem(Kelas, Felas, Uelas, 2);

      // Solve phase field problem    
      assembleGlobalStiffness(Kpf, Fpf, elements, nodes, material, nstate_pf);      
      solveSystem(Kpf, Fpf, Upf, 1);
    }
    if(iter == maxiter){
      std::cout << "------> Staggered scheme did not converge in " << maxiter << " iterations." << "\nAccepting current solution and continuing" << std::endl;     
    }
    std::string filename = basefilename + std::to_string(step) + vtkextension;
    generateVTKLegacyFile(nodes, elements, filename);

    VectorXd sig;
    computeSigmaAtCenter(elements[50],nodes,sig);
    outpdelta << pseudotime << " " << sig[0] << std::endl;
  }

  std::cout << std::endl << "================> Simulation completed!" << std::endl;
  simulation_time.elapsed("complete simulation");
  return 0;
}

// =============================== FUNCTION IMPLEMENTATIONS ======================
// ===============================================================================

void createRectangularMesh(std::vector<Node> &nodes, std::vector<Element> &elements, int num_elements_x, int num_elements_y, double length, double height) {
  // Generate nodes
  for (int j = 0; j <= num_elements_y; ++j) {
    for (int i = 0; i <= num_elements_x; ++i) {
      nodes.push_back({i * length / num_elements_x, j * height / num_elements_y});
    }
  }

  // Print the nodes
  // std::cout << "Nodes:" << std::endl;
  // for (const auto &node : nodes) {
  //   std::cout << "(" << node.x << ", " << node.y << ")" << std::endl;
  // }  

  // Generate elements
  for (int j = 0; j < num_elements_y; ++j) {
    for (int i = 0; i < num_elements_x; ++i) {
      int n1 = j * (num_elements_x + 1) + i;
      int n2 = n1 + 1;
      int n3 = n1 + num_elements_x + 1;
      int n4 = n3 + 1;
      elements.push_back({{n1, n2, n4, n3}});
      // Print the coordinates of each node in the element
      // std::cout << "Element nodes: ";
      // for (int node_id : elements.back().node_ids) {
      //   std::cout << "(" << nodes[node_id].x << ", " << nodes[node_id].y << ") ";
      // }
      // std::cout << std::endl;
    }
  }
}

void assembleGlobalStiffness(MatrixXd &K, VectorXd& F, const std::vector<Element> &elements, const std::vector<Node> &nodes, MaterialParameters& mat, int nstate) {

  Timer time;

  K.setZero();
  F.setZero();
  for (const auto &element : elements) {
    // Element stiffness matrix (for simplicity, assume a 4-node quadrilateral element)
    const int nquadnodes = 4;
    const int ndofel = nstate * nquadnodes;
    MatrixXd Ke = MatrixXd::Zero(ndofel, ndofel);
    VectorXd Fe = VectorXd::Zero(ndofel);
    computeElementStiffness(Ke, Fe, nodes, element, mat, nstate);

    // Assemble Ke into the global stiffness matrix K
    for (int i = 0; i < nquadnodes; ++i) {
      int row = nstate * element.node_ids[i];
      for (int k = 0; k < nstate; ++k) {
        F[row + k] += Fe[nstate * i + k];
      }
      for (int j = 0; j < nquadnodes; ++j) {        
        int col = nstate * element.node_ids[j];
        for (int k = 0; k < nstate; ++k) {
          for (int l = 0; l < nstate; ++l) {
            K(row + k, col + l) += Ke(nstate * i + k, nstate * j + l);
          }
        }
      }
    }
  }
  // time.elapsed("assembly");
}

void computeElementStiffness(MatrixXd &Ke, VectorXd& Fe, const std::vector<Node> &nodes, const Element &element, MaterialParameters& mat, const int nstate) {
  // Extract node coordinates
  const Node &n1 = nodes[element.node_ids[0]];
  const Node &n2 = nodes[element.node_ids[1]];
  const Node &n3 = nodes[element.node_ids[2]];
  const Node &n4 = nodes[element.node_ids[3]];

  // Jacobian matrix
  double base = n2.x - n1.x;
  double height = n4.y - n1.y;
  double area = base * height;  // base * height since it is a simple mesh
  double detjac = area / 4.0;
  double dqsidx = 2.0 / base;    // for simple rectangular elements
  double dqsidy = 2.0 / height;  // for simple rectangular elements

  // Create diagonal matrix with dqsidx and dqsidy
  MatrixXd J_inv = MatrixXd::Zero(2, 2);
  J_inv(0, 0) = dqsidx;
  J_inv(1, 1) = dqsidy;

  if (nstate == 2) { // elasticity stiff
    // Compute B matrix
    MatrixXd B;
    for (const auto &qp : intrule) {
      MatrixXd N, dN;
      shapeFunctions(N, dN, qp.xi, qp.eta, nstate);

      // Transform derivatives to global coordinates
      MatrixXd dN_xy = J_inv.transpose() * dN.transpose();

      // Create B matrix
      createB(B, dN_xy.transpose());

      MatrixXd Ddeteriorated = D;
      // Interpolate phase field solution at quadrature points
      double phase_field = 0.0;
      for (int i = 0; i < 4; ++i) phase_field += N(0, 2 * i) * Upf(element.node_ids[i]); // N is repeated for x and y so we multiply by 2
      
      // Deteriorate the material properties based on the phase field
      Ddeteriorated *= (1 - phase_field) * (1 - phase_field);

      // Compute element stiffness matrix contribution of this integration point
      Ke += B.transpose() * Ddeteriorated * B * qp.weight * detjac;
    }
  }
  else if (nstate == 1) { // phase field stiff
    double G = mat.G, l = mat.l;
    double c0 = 2.;

    for (const auto &qp : intrule) {
      MatrixXd N, dN;
      shapeFunctions(N, dN, qp.xi, qp.eta, nstate);

      // Transform derivatives to global coordinates
      MatrixXd dN_xy = J_inv.transpose() * dN.transpose();

      double sigmaDotEps = calculateSigmaDotEps(element, dN_xy, mat);

      for (int i = 0; i < 4; ++i) {
        Fe[i] += detjac * qp.weight * 0.5 * sigmaDotEps * N(0, i);
        for (int j = 0; j < 4; ++j) {
          Ke(i, j) += detjac * qp.weight * (G * l / c0 * (dN_xy(0, i) * dN_xy(0, j) + dN_xy(1, i) * dN_xy(1, j)) + (G / (l * c0) + 0.5*sigmaDotEps) * N(0, j) * N(0, i));
        }
      }
    }
  }
  else {
    throw std::bad_exception();
  }

  // Print the element stiffness matrix Ke
  // std::cout << "Element stiffness matrix Ke: \n"
  //           << Ke << std::endl;
}

double calculateSigmaDotEps(const Element &element, const MatrixXd &dN, MaterialParameters& mat) {
  double sigmaDotEps = 0.;
  // Calculate the derivative of the elastic solution using dN and Uelas
  MatrixXd dU = MatrixXd::Zero(2, 2);
  for (int i = 0; i < 4; ++i) {
    int index = 2 * element.node_ids[i];
    dU(0, 0) += dN(0, i) * Uelas(index); // duxdx
    dU(0, 1) += dN(1, i) * Uelas(index); // duxdy
    dU(1, 0) += dN(0, i) * Uelas(index + 1); // duydx
    dU(1, 1) += dN(1, i) * Uelas(index + 1); // duydy
  }

  // Calculate strain tensor
  MatrixXd strain = 0.5 * (dU + dU.transpose());

  // Convert strain to a vector
  VectorXd strain_vec(3);
  strain_vec << strain(0, 0), strain(1, 1), 2 * strain(0, 1); // note the times 2 in the off-diagonal term

  // Calculate stress
  VectorXd stress_vec = D * strain_vec;

  // Calculate sigma dot epsilon
  sigmaDotEps = stress_vec.dot(strain_vec);
  // std::cout << "Sigma dot Epsilon: " << std::scientific << sigmaDotEps << std::endl;

  return sigmaDotEps;
}

VectorXd& computeSigmaAtCenter(const Element &element, std::vector<Node> &nodes, VectorXd& stress_vec) {  
  // Calculate the derivative of the elastic solution using dN and Uelas
  double qsi = 0., eta = 0.; // center of element
  const Node &n1 = nodes[element.node_ids[0]];
  const Node &n2 = nodes[element.node_ids[1]];
  const Node &n3 = nodes[element.node_ids[2]];
  const Node &n4 = nodes[element.node_ids[3]];  
  double base = n2.x - n1.x;
  double height = n4.y - n1.y;
  double area = base * height;  // base * height since it is a simple mesh
  double detjac = area / 4.0;
  double dqsidx = 2.0 / base;    // for simple rectangular elements
  double dqsidy = 2.0 / height;  // for simple rectangular elements

  // Create diagonal matrix with dqsidx and dqsidy
  MatrixXd J_inv = MatrixXd::Zero(2, 2);
  J_inv(0, 0) = dqsidx;
  J_inv(1, 1) = dqsidy;  
  MatrixXd N, dN;
  shapeFunctions(N, dN, qsi, eta, 2);
  MatrixXd dN_xy = J_inv.transpose() * dN.transpose();

  MatrixXd dU = MatrixXd::Zero(2, 2);
  for (int i = 0; i < 4; ++i) {
    int index = 2 * element.node_ids[i];
    dU(0, 0) += dN_xy(0, i) * Uelas(index); // duxdx
    dU(0, 1) += dN_xy(1, i) * Uelas(index); // duxdy
    dU(1, 0) += dN_xy(0, i) * Uelas(index + 1); // duydx
    dU(1, 1) += dN_xy(1, i) * Uelas(index + 1); // duydy
  }

  // Calculate strain tensor
  MatrixXd strain = 0.5 * (dU + dU.transpose());

  // Convert strain to a vector
  VectorXd strain_vec(3);
  strain_vec << strain(0, 0), strain(1, 1), 2 * strain(0, 1); // note the times 2 in the off-diagonal term

  double phase_field = 0.0;
  for (int i = 0; i < 4; ++i) phase_field += N(0, 2 * i) * Upf(element.node_ids[i]);  // N is repeated for x and y so we multiply by 2

  // Calculate stress
  double g = (1. - phase_field) * (1. - phase_field);
  stress_vec = g * D * strain_vec;

  return stress_vec;
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

void applyBoundaryConditions(MatrixXd &K, VectorXd &F, const std::vector<BC> &bc_nodes) {
  for (const auto &bc : bc_nodes) {
    int row = 2 * bc.node;
    double xval = bc.xval * pseudotime;
    double yval = bc.yval * pseudotime;
    if (bc.type == 0) {
      // Dirichlet in x and y
      F -= K.col(row) * xval;
      F -= K.col(row+1) * yval;
      K.row(row).setZero();
      K.col(row).setZero();
      K.row(row+1).setZero();
      K.col(row+1).setZero();
      K(row, row) = 1.0;      
      K(row + 1, row + 1) = 1.0;
      F(row) = xval;
      F(row + 1) = yval;
    } else if (bc.type == 1) {
      // Dirichlet in x      
      F -= K.col(row) * xval;
      K.row(row).setZero();
      K.col(row).setZero();      
      K(row, row) = 1.0;
      F(row) = xval;
    } else if (bc.type == 2) {
      // Dirichlet in y
      F -= K.col(row+1) * yval;
      K.row(row+1).setZero();
      K.col(row+1).setZero();      
      K(row + 1, row + 1) = 1.0;
      F(row + 1) = yval;
    } else if (bc.type == 3) {
      // Neumann
      F(row) += xval;
      F(row + 1) += yval;
    }
  }
}

void solveSystem(const MatrixXd &K, const VectorXd &F, VectorXd &U, const int nstatevar) {
  Timer time;

  // U = K.llt().solve(F);
  if(nstatevar == 2){
    if(!lltOfKelas.cols()){
      lltOfKelas.compute(K);
    }
    U = lltOfKelas.solve(F);
  }else{
    if(!lltOfKpf.cols()){
      lltOfKpf.compute(K);
    }
    U = lltOfKpf.solve(F);
  }
  // time.elapsed("solve");
}

void generateVTKLegacyFile(const std::vector<Node> &nodes, const std::vector<Element> &elements, const std::string &filename) {
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