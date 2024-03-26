# include <iostream>
# include "Eigen/Eigen"
using namespace Eigen;

double palu(const MatrixXd& A, const VectorXd& b, const VectorXd& x_esatta)
{
    VectorXd x_palu = A.lu().solve(b);
    double err_palu = (x_palu - x_esatta).norm()/x_esatta.norm();
    return err_palu;
}

double qr(const MatrixXd& A, const VectorXd& b, const VectorXd& x_esatta)
{
    HouseholderQR<Matrix2d> qr(A);
    Vector2d x_qr = qr.solve(b);
    double err_qr = (x_qr - x_esatta).norm()/x_esatta.norm();
    return err_qr;
}

int main()
{
    Vector2d x_esatta;
    x_esatta << -1.0e+0, -1.0e+00;

    Matrix2d A_1;
    A_1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b_1;
    b_1 << -5.169911863249772e-01, 1.672384680188350e-01;
    double err_palu_1 = palu(A_1, b_1, x_esatta);
    double err_qr_1 = qr(A_1,b_1,x_esatta);
    std::cout << "Errore 1 con PA=LU: " << err_palu_1 << " | " << "Errore 1 con QR: " << err_qr_1 << std::endl;


    Matrix2d A_2;
    A_2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b_2;
    b_2 << -6.394645785530173e-04, 4.259549612877223e-04;
    double err_palu_2 = palu(A_2, b_2, x_esatta);
    double err_qr_2 = qr(A_2,b_2,x_esatta);
    std::cout << "Errore 2 con PA=LU: " << err_palu_2 << " | " << "Errore 2 con QR: " << err_qr_2 << std::endl;


    Matrix2d A_3;
    A_3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d b_3;
    b_3 << -6.400391328043042e-10, 4.266924591433963e-10;
    double err_palu_3 = palu(A_3, b_3, x_esatta);
    double err_qr_3 = qr(A_3,b_3,x_esatta);
    std::cout << "Errore 3 con PA=LU: " << err_palu_3 << " | " << "Errore 3 con QR: " << err_qr_3 << std::endl;


    return 0;
}
