NetworkAttackSimulator-master(23.03.22):
This project gets the correct result for all benchmark problems.
-modify the bug since of numpy version.
-modify the reinforcement learning.
	--> firstly i improve the reward part. Original rl sets the reward as zero when action successed, however i modify this
		to reward positive value when action successed.
	    Reward for successing exploit, i set the reward as 4 to activate the rl with exploiting.
	    Reward for successing escalation, i set the reward as 2.
	    It is better to be smaller value than success exploiting since if it is greater than success exploiting, rl may
	    be in a infinite loop.
	    It is also better to be greater value than other success actions to get the goal quickly.
	-->secondly, i modify the action selection part.
	    Although, we learn the rl using updated reward, the net can be in a infinite loop when the remained nodes are small.
	    To prevent it, i altered epsilon greedy part.

NetworkAttackSimulator-master(23.03.23.v1):
This project updates the visualization of result.
visualize the results with positive reward only.

NetworkAttackSimulator-master(23.03.23.v1.1): 
This is the new version of nas.
To get the optimal path, we make all the reward are to be negative.
To do it, i changed line 212 of envs/environment.py as
reward = action_obs.value - action.cost -4
And i tested this algorithm with training steps as 1000000.
To find the optimal path, we should set reward correctly.
The problem is that if we select the random path, training can't be converged since there are too many actions to be able to select 
for each step.
To solve this, we should set the reward as different for the success and false actions.

NetworkAttackSimulator-master(23.03.23.v1.2):
This is a test version. I select the reward as
reward = action_obs.value - action.cost -4.
This is same as v1.1 however i used reward for success codes as same as in the original code.
Test shows that if the reward for success are greater than that for false, we can get the success attack path.
However, it is not a optimal path but has a long path. 

