# n-Pendulum Dynamical Simulations

This software implements several numerical schemes to solve the equations of motions for multiple pendulum systems.
 
For double pendulums (N=2), it uses a faster (and more direct) acceleration function and the 4th order Runge-Kutta method by default.

For N > 2, it also includes a matrix-based method to compute the acceleration of arbitrary length pendulums provided in "Parallel simulation of large scale multibody systems" by Kloppel at al. (https://onlinelibrary.wiley.com/doi/pdf/10.1002/pamm.201510327)

Since multiple pendulums are highly chaotic and become increasingly unstable with additional links, the package includes two methods that prioritize stability over truncation error: the 2nd order Crank-Nicholson method and the 3rd order Strong Stability Preserving Runge-Kutta 3 (SSPRK3) method. 

The example notebook includes multiple animation options including an embedded MPEG video, an interactive HTML5 animation or an exported GIF file.

This example shows a double pendulum using the dedicated acceleration function and the RK4 method.
![An animation of the double pendulum](example_2.gif)

This demonstrates the motion of a pendulum with 20 links using the general acceleration function and the SSPRK3 method
![An animation of a 20-link pendulum](example_20.gif)
