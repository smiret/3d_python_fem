# 3d_python_fem
3D Python Finite Element Code

This code is a three-dimensional finite element solver of the heat equation implemented in Python. The code includes the setup of the equation into matrix form by computing various integrals. The matrices are then fed into a sparse matrix system, which is then solved to give the resulting temperature distribution in the body. The resulting temperature distribution is written to an xml file that can be viewed in a rendering software, such as Paraview.
