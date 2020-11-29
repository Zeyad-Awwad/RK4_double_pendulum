# Double Pendulum Dynamics (4th Order Runge-Kutta)

 A dynamic model that solves the equations of motion for a double pendulum using the 4th order Runge-Kutta method.
 
 The example notebook includes multiple animation options including an embedded MPEG video, an interactive HTML5 animation or an exported GIF file.
 
![An animation of the double pendulum](example_2.gif)

I've also included functions to test out n-pendulum dynamics, implementing a matrix-based method for arbitrary length pendulums provided in "Parallel simulation of large scale multibody systems" by Kloppel at al. (https://onlinelibrary.wiley.com/doi/pdf/10.1002/pamm.201510327)

Since multiple pendulums are highly chaotic and become increasingly unstable with additional links, I also included two numerical methods designed to increase stability at the cost of accuracy: the 2nd order Crank-Nicholson method and the 3rd order Strong Stability Preserving Runge-Kutta 3 (SSPRK3) method. 

The following example demonstrates the motion a pendulum with 20 links using the SSPRK3 method
![An animation of a 20-link pendulum](example_20.gif)
