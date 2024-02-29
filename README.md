# Reinforcement Learning for Solving the Vehicle Routing Problem in the Aquaculture 


This code is an adaptation of the NIPS 2018 paper by Nazari et.al.  https://arxiv.org/abs/1802.04240v2
![Vehicle Routing Problem](https://www.researchgate.net/profile/Savvas-Pericleous/publication/319754352/figure/fig1/AS:631655517659162@1527609819407/The-Capacitated-Vehicle-Routing-Problem-CVRP.png)
The architecture and the main structure of the code is similar to the paper of Nazari et .al. However the framework is focused on adapting this work to the aquaculture sector with some requirements as:
*The depot must be at (0,0) to resemble it to a usual scenario of the Aquaculture (offshore problem)
*The static tensor is modified to use fixed locations. (Just in the validation tensor and in the test tensor)
*We add some KPIs in the Main to undertand the evolution of the knowledge of the Agent in an easier way 
*Finally, we add a new implementation to test the model using the parameters obteined in the training.
## Dependancies
 * Pytorch==0.4.1 
 * matplotlib
 * tqdm

