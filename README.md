[![arXiv](https://img.shields.io/badge/arXiv-2206.09843-b31b1b.svg)](https://arxiv.org/abs/2306.13554)

Official repository of the iMuJoCo (iMitation MuJoCo) dataset, an offline dataset for imitation learning. Presented in: 

*"Comparing the Efficacy of Fine-Tuning and Meta-Learning for Few-Shot Policy Imitation", Patacchiola M., Sun M., Hofmann K., Turner R.E., Conference on Lifelong Learning Agents - CoLLAs 2023* [[arXiv]](https://arxiv.org/abs/2306.13554)

**Overview**: iMuJoCo builds on top of OpenAI-Gym MuJoCo providing a heterogeneous benchmark for training and testing imitation learning methods and offline RL methods. Heterogeneity is achieved by producing a large number of variants of three base environments: Hopper, Halfcheetah, and Walker2d. For each variant a policy has been trained via SAC, then the policy has been used to generate 100 offline trajectories. 

**What's included?** iMuJoCo includes (1) 100 trajectories from pretrained policies for each environment variant (`./imujoco/dataset` folder), (2) pretrained SAC policies for each variant (`./imujoco/policies` folder), and (3) XML files builder for each environment (`./imujoco/xml` folder). The user can access the environment variant (via the OpenAI-Gym API and the XML configuration file), the offline trajectories (via a Python/Pytorch data loader), and the underlying SAC policy network (using the Stable Baselines API).

The overall structure of iMuJoCo is the following:

```
./imujoco
    dataset
        sac-halfcheetah-jointdec_25_bfoot.npz
        sac-halfcheetah-jointdec_25_bshin.npz
        ...
        
    policies
        sac-halfcheetah-jointdec_25_bfoot.zip
        sac-halfcheetah-jointdec_25_bshin.zip
        ...
        
    xml
        halfcheetah-jointdec_25_bfoot.xml
        halfcheetah-jointdec_25_bshin.xml
        ...
```


Difference with previous benchmarks
----------------------------------

A few benchmarks have been proposed to address meta-learning and offline learning in RL, such as [Meta-World](https://meta-world.github.io/), [Procgen](https://github.com/openai/procgen), and [D4RL](https://arxiv.org/abs/2004.07219). However, differently from the standard meta-learning setting, in imitation learning we need a large variety of offline trajectories, collected from policies trained on heterogeneous environments. 

Existing benchmarks are not suited for this case as they: do not provide pretrained policies and their associated trajectories (e.g. Meta-World and Procgen), lack in diversity (Meta-World and D4RL), or do not support continuous control problems (e.g. Procgen).

Environment variants
--------------------

Each environment variant falls into one of these four categories:

- **mass**: increase or decrease the mass of a limb by a percentage; e.g. if the mass is 2.5 and the percentage is 200% then the new mass for that limb will be 7.5.
- **joint**: limit the mobility of a joint by a percentage range, e.g. if the joint range is 180 degrees and the percentage is -50% then the maximum range of motion becomes 90 degrees.
- **length**: increase or decrease the length of a limb by a percentage; e.g. if the length of a limb is 1.5 and the percentage is 150% then the new length will be 3.75.
- **friction**: increase or decrease the friction by a percentage (only for body parts that are in contact with the floor); e.g. if the friction is 1.9 and the percentage is -50% then the new friction will be 0.95.

Note that each environment has unique dynamics and agent configurations, resulting in different numbers of variants. Specifically, we have 37 variants for Hopper, 53 for Halfcheetah, and 64 for Walker2d, making a total of 154 variants.

Installation
------------

1. Requirements: there are no particular requirements, you need to install Numpy and Pytorch to use the sampler, [OpenAI-Gym](https://github.com/openai/gym) and [StableBaselines3](https://stable-baselines3.readthedocs.io) for loading the environment/policies.

2. Clone the repository `git clone https://github.com/mpatacchiola/imujoco.git` and set it as current folder with `cd imujoco`

3. Download the dataset files (approximately **2.7 GB**) from our page on [zenodo.org](https://zenodo.org/record/7971395):
 
```
wget https://zenodo.org/record/7971395/files/xml.zip
wget https://zenodo.org/record/7971395/files/policies.zip
wget https://zenodo.org/record/7971395/files/dataset.zip
```

4. Unzip the files into the `imujoco` folder: 

```
unzip xml.zip
unzip policies.zip
unzip dataset.zip
```

Usage
------

**Sampling offline trajectories**

In iMuJoCo there are a set of trajectories collected by agents trained using SAC. There is a total of 100 trajectores per each environment variant, which are stored as numpy compressed files (npz) into the `./dataset` folder. The following is an example of how to use [sampler.py](./sampler.py) to sample offline trajectories (pytorch).

```python
import os
from sampler import Sampler

env_name = "Hopper-v3" # can be: 'Hopper-v3', 'HalfCheetah-v3', 'Walker2d-v3'.

# Here we simply accumulate all the npz files for Hopper-v3.
files_list = list()
for filename in os.listdir("./dataset"):
    if filename.endswith(".npz") and env_name.lower()[0:-2] in filename: 
             files_list.append(os.path.abspath("./dataset"+filename))
    print("\n", files_list, "\n")

# Defining train/test samplers by allocating 75% of the 
# trajectories for training and 25% for testing.
train_sampler = Sampler(env_name=env_name, data_list=files_list, portion=(0.0,0.75))
test_sampler = Sampler(env_name=env_name, data_list=files_list, portion=(0.75,1.0))

# Sampling 5 trajectories (without replacement) using the train sampler
# The sampler returns the states/actions tensor for the sequences.
x, y = train_sampler.sample(tot_shots=5, replace=False)
```

**Loading one of the environment variants**

Each environment variant can be loaded as a standard OpenAI-Gym env, by using the XML file associated to it. For instance, here we load an environment for Hopper where the mass of the leg has been decreased by 25%:

```python
import gym
import os

env_name = "Hopper-v3"
xml_file = os.path.abspath("./xml/hopper-massdec_25_leggeom.xml")

# Generate and reset the env.
env = gym.make(env_name, xml_file=xml_file)
env.reset()

# Move in the env with a random policy for one episode.
for _ in range(1000):
    action = env.action_space.sample() 
    observation, reward, done, _ = env.step(action)
    if done: break

env.close()
```

**Loading a pretrained SAC policy**

Each environment variant has an associated SAC policy that has been trained on it. For this stage we used [Stable Baselines v3](https://stable-baselines3.readthedocs.io).

Here is an example on how to load a pretrained policy for its associated environment and evaluate its performance:

```python
import os
from stable_baselines3 import SAC
import gym

def evaluate_policy(env, model, tot_episodes, deterministic=True): 
    model.policy.actor.eval()
    episode_reward_list = list()
    for episode in range(tot_episodes):
        obs_t = env.reset()
        cumulated_reward = 0.0
        for i in range(1000):            
            action, _states = model.predict(obs_t, deterministic=deterministic)
            obs_t1, reward, done, info = env.step(action)
            cumulated_reward += reward
            if done: obs_t1 = env.reset()   
            obs_t = obs_t1           
        episode_reward_list.append(cumulated_reward)
    return episode_reward_list


env_name = "Hopper-v3"
eval_episodes = 10
xml_file = os.path.abspath("./xml/hopper-massdec_25_leggeom.xml")
policy_file = os.path.abspath("./policies/sac-hopper-massdec_25_leggeom.zip")

# Create the Gym env.
env = gym.make(env_name, xml_file=xml_file)

# Create the SAC env and load the pretrained policy
model = SAC("MlpPolicy", env, verbose=1)
model = SAC.load(policy_file)

# Evaluate the policy on the associated environment
reward_list = evaluate_policy(env, model, tot_episodes=eval_episodes)
print(f"Average reward on {eval_episodes} episodes .... {sum(reward_list)/len(reward_list)}")
```

Citation
--------

```bibtex
@inproceedings{patacchiola2023comparing,
  title={Comparing the Efficacy of Fine-Tuning and Meta-Learning for Few-Shot Policy Imitation},
  author={Patacchiola, Massimiliano and Sun, Mingfei and Hofmann, Katja and Turner, Richard E},
  booktitle={Conference on Lifelong Learning Agents},
  year={2023}
}
```

