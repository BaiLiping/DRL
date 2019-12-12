Review of the course material:
1. ![Synaptic_Model](Synaptic_Model.png)
Key take away, not only does this model have the activation, it also has inhibition, which has veto power. I think this is omitted in the current incarnation of neural network?

2. ![Perceptron](Perceptron.png)
the visualization of hyperplane is actually quite helpful in terms of undersanding multi-layer perceptron as a universal function approximator

3. ![Decision Bountdaries Constructed via Multiple Perceptron](Multiple_Perceptron.png) ![Second](2.png)
Every perceptron defines the boundary of an alphine plane, with multiple spetrons, you are effectively linearly combining the boundaries. With the addition of activation functioin, which is a gated value for perceptron, you also have differnet values for different regions of the plane.

If you define the desired region with threshod value of 5 then you effectively have a pentagon.This is how you constructor shape out of perceptron.

4. ![take it to the limit](3.png) ![4](4.png) ![5](5.png)
when you have infinite amount of perceptrons to define the boundaries, then you can define a cylinder, where inside the cylinder has value of N, out side of the cylinder have the value of N/2.  However, if you offset it with a bias, you have the cylinder height of N/2, out side height of 0. This cylinder is the basic building block for approximating other functions.

5. ![time series](Multiple_Perceptron_for_Time_Series)
Not only can perceptrons be used to model any 2D shape, if you tile them up in temporal axis, you can use it to model any time series.

6. ![0](AddingCircles.png)![7](7.png) ![8](8.png)

7. ![another way of building the basic unit](9.png) ![10](10.png) ![11](11.png)




