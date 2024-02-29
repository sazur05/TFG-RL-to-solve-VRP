# Reinforcement Learning for Solving the Vehicle Routing Problem in the Aquaculture 

This code is an adaptation of the NIPS 2018 paper by Nazari et.al.  https://arxiv.org/abs/1802.04240v2
![Vehicle Routing Problem](https://www.researchgate.net/profile/Savvas-Pericleous/publication/319754352/figure/fig1/AS:631655517659162@1527609819407/The-Capacitated-https://github.com/pullsVehicle-Routing-Problem-CVRP.png)

The architecture and the main structure of the code is similar to the paper of Nazari et .al. However the framework is focused on adapting this work to the aquaculture sector with some requirements as:
*The depot must be at (0,0) to resemble it to a usual scenario of the Aquaculture (offshore problem)
*The static tensor is modified to use fixed locations. (Just in the validation tensor and in the test tensor)
*We add some KPIs in the Main to undertand the evolution of the knowledge of the Agent in an easier way 
*Finally, we add a new implementation to test the model using the parameters obteined in the training.

There exists a tesis with much more detail of the framework, where it is explained all the history of the Aquaculture, why an implemetation of Deep Reinforcement Learning is interesting (also an explanation of its competitors and an overview of Deep Learning and Reinforcement Learning), a description of the model and a set of experiments to optimize the variables used in the model.
## Dependancies
certifi==2023.5.7
charset-normalizer==3.1.0
cycler==0.11.0
fonttools==4.38.0
idna==3.4
kiwisolver==1.4.4
matplotlib==3.5.3
numpy==1.21.6
packaging==23.1
Pillow==9.5.0
pyparsing==3.1.0
python-dateutil==2.8.2
requests==2.31.0
six==1.16.0
torch==1.13.1
torchaudio==0.13.1
torchvision==0.14.1
typing-extensions==4.6.3
urllib3==2.0.3
