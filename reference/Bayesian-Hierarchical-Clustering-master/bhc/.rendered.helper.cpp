#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::log;
using std::lgamma;

MatrixXd scale_matrix(MatrixXd X, int N, int k, double r, MatrixXd m, MatrixXd S) {
    MatrixXd xsum = X.colwise().sum();
    MatrixXd res = S + X.transpose() * X
                     + r * N / (N + r) * m * m.transpose()
                     - 1/(N+r) * (xsum.transpose() * xsum)
                     - (r / (N + r)) * (m * xsum + xsum.transpose() * m.transpose());
    return res;
}

double signprod(VectorXd dU){
    double res = 1;
    for(int i = 0; i < dU.size(); i++){
        if(dU[i] < 0){
            res *= -1;
        }
    }
    return res;
}

double logdet(MatrixXd S){
    Eigen::PartialPivLU<MatrixXd> lu(S);
    MatrixXd U = lu.matrixLU();
    VectorXd dU = U.diagonal();
    MatrixXd P = lu.permutationP();
    double c = P.determinant() * signprod(dU);
    double d = log(c) + dU.array().abs().log().sum();
    return d;
}

double niw(MatrixXd X, MatrixXd m, MatrixXd S, double r){
    const double PI  =3.141592653589793238463;
    double N = (double)X.rows();
    double k = (double)X.cols();
    // v = k;
    double vprime = k + N;

    MatrixXd Sprime = scale_matrix(X, N, k, r, m, S);

    double num = vprime*k/2*log(2);
    double den = k * k / 2 * log(2);
    for(int i = 0; i < k; i++){
        num += lgamma((double)(vprime - i)/2);
        den += lgamma((double)(k - i)/2);
    }

    double lml = - N*k/2 * (log(2) + log(PI))
                 + k/2*(log(r)-log(N + r))
                 + k/2*logdet(S)
                 - vprime/2*logdet(Sprime)
                 + num - den;
    return lml;
}

PYBIND11_PLUGIN(helper){
    pybind11::module m("helper", "helper functions");
    m.def("scale_matrix", &scale_matrix);
    m.def("niw", &niw);
    m.def("logdet", &logdet);
    return m.ptr();
}
