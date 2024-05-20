# PaN
Prediction and noise can implement flexible reward-seeking behavior.

## Demo notebook
The demo notebook includes walkthroughs of the base PaN algorithm and generates plots in Appendix Figure 12. These are experiments with a PaN network with 30 hidden neurons and three possible actions with reward values (0,0,.5). The notebook has dependencies on data in the `Noise sweep` folder.

## Videos

1. Two-neuron trajectory. This video animates every N points for a setup with two neurons and one connecting weight, plotted in the full parameter space $(x_0, x_1, W)$ where $x_i$ denotes activity for neuron $i$ and $W$ denotes the connecting weight.
2. Open-field track. Every 100th point is plotted for a full 500k-timestep track in the open-field search task in Figure 6b. The track animated is the one plotted in Figure 6c.

## Scripts
Scripts are labelled with their corresponding figure and a description. 

### Figure 3
`Fig3_TwoNeuronNoiseSweep.py` runs a two-neuron system for 500k timesteps, 100 trials per activity noise and weight noise setting. Networks can choose between two actions which give 0 or 0.5 reward. If the motor neuron activity is above 0 at any given time $t$, then the next timestep's sensory input is the corresponding reward, $s(t+1)$. 10 activity noise settings, 10 weight noise settings, 100 seeds each. Saves all parameters and action choices.
          
          python Fig3_TwoNeuronNoiseSweep.py s, where s is an integer in 0-9999
          
### Figure 4
Figure 4 uses data from Figure 3.

### Figure 5
`Fig5_6LeverBandit.py` runs a network with one input neuron, 30 hidden neurons, and 6 output neurons. Networks choose between actions corresponding to fixed rewards that go from 0 to 0.5, in 0.1 increments. Code runs 25 trials. Saves first layer weights, second layer weights, and actions.
          
          python Fig5_6LeverBandit.py s, where s is an integer in 0-24
          
### Figure 6
Landscape experiments where networks' actions correspond to movements in the landscape. Each script is designed to run 1000 trials for 500k timesteps each, where each trial is initialized at a random location.
       
`Fig6I_3PeaksRange.py` runs agents on landscapes with three local maxima and translational controls.

        python Fig6I_3PeaksRange.py s, where s is an integer in 0-999
        
`Fig6II_SwitchingPeaks.py` runs agents on landscapes with one local maximum that switches locations every 50k timesteps.

        python Fig6II_SwitchingPeaks.py s, where s is an integer in 0-999
        
Three local maxima. Rotational controls for the first 500k timesteps; translational controls for the last 500k timesteps.

        python Fig6III_Metamorphosis.py s, where s is an integer in 0-999
          
### Figure 7
All conditions run for 500k timesteps.

`Fig7I_TwoTaskRandom.py` runs agents with two input neurons and three output neurons. 500 random seeds for Figure 7c-d.
        
        python Fig7I_TwoTaskRandom.py s, where s is an integer in 0-499

`Fig7II_TwoTask_cap50.py` runs agents with two input neurons and three output neurons. Runs 50 random initializations with 50 noise patterns each for Figure 7e.

        python Fig7II_TwoTask_cap50.py s, where s is an integer in 0-2499

`Fig7III_ConnTogether.py` runs agents with 2 input neurons, 30 hidden neurons, and 3 output neurons. Input connectivities are changed through 7 densities as described in figure legend. 50 trials for each connection density. Figure 7f-g.

        python Fig7III_ConnTogether.py s, where s is an integer in 0-349

`Fig7IV_LearningRates.py` runs networks with 2 input neurons, 30 hidden neurons, and 3 output neurons. Action 1 causes a change in activity and weight learning rates, 10 settings each, 50 seeds per condition. Figure 7h-j.

        python Fig7IV_LearningRates.py s, where s is an integer in 0-4999
          
### Figure 8

All conditions run for 500k timesteps.

`Fig8I_FoodWater.py` runs agents with two input neurons, 30 hidden neurons, and four output motor neurons that are linked to translational movement. 1000 random seeds for Figure 8a-c.
        
        python Fig8I_FoodWater.py s, where s is an integer in 999

`Fig8II_FoodWaterConn.py` runs agents with two input neurons, 30 hidden neurons, and four output motor neurons that are linked to translational movement. One input neuron is connected to only 3 hidden neurons; the other is connected to all 30 hidden neurons. 1000 random seeds for Figure 8e-f.
        
        python Fig8II_FoodWaterConn.py s, where s is an integer in 999

`Fig8III_FoodWaterLR.py` runs agents with two input neurons, 30 hidden neurons, and four output motor neurons that are linked to translational movement. Both input neurons are fully connected. Learning rates change so that activity learning rates are low when the change in one reward is positive; learning rate modulation changes to the other reward type every 50k timesteps for 500k timesteps total. 1000 random seeds for Figure 8g-i.
        
        python Fig8III_FoodWaterLR.py s, where s is an integer in 999

