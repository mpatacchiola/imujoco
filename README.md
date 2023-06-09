Official repository of the iMuJoCo (iMitation MuJoCo) dataset, an offline dataset for imitation learning.

Overview
--------

A few benchmarks have been proposed to address meta-learning and offline learning in RL, such as [Meta-World](https://meta-world.github.io/), [Procgen](https://github.com/openai/procgen), and [D4RL](https://arxiv.org/abs/2004.07219). However, differently from the standard meta-learning setting, in imitation learning we need a large variety of offline trajectories, collected from policies trained on heterogeneous environments. Existing benchmarks are not suited for this case as they: do not provide pretrained policies and their associated trajectories (e.g. Meta-World and Procgen), lack in diversity (Meta-World and D4RL), or do not support continuous control problems (e.g. Procgen).

In order to satisfy these requirements, we created a variant of OpenAI-Gym MuJoCo that we called iMuJoCo (iMitation MuJoCo). The iMuJoCo dataset builds on top of MuJoCo providing a heterogeneous benchmark for training and testing imitation learning methods and offline RL methods. Heterogeneity is achieved by producing a large number of variants of three base environments: Hopper, Halfcheetah, and Walker2d. For each variant a policy has been trained via SAC, then the policy has been used to generate 100 offline trajectories. The user can access the environment variant (via the OpenAI-Gym API and a XML configuration file), the offline trajectories (via a Python data loader), and the underlying SAC policy network (using the Stable Baselines API). Each environment variant falls into one of these four categories:

- **mass**: increase or decrease the mass of a limb by a percentage; e.g. if the mass is 2.5 and the percentage is 200% then the new mass for that limb will be 7.5.
- **joint**: limit the mobility of a joint by a percentage range, e.g. if the joint range is 180 degrees and the percentage is -50% then the maximum range of motion becomes 90 degrees.
- **length**: increase or decrease the length of a limb by a percentage; e.g. if the length of a limb is 1.5 and the percentage is 150% then the new length will be 3.75.
- **friction**: increase or decrease the friction by a percentage (only for body parts that are in contact with the floor); e.g. if the friction is 1.9 and the percentage is -50% then the new friction will be 0.95.

Note that each environment has unique dynamics and agent configurations, resulting in different numbers of variants. Specifically, we have 37 variants for Hopper, 53 for Halfcheetah, and 64 for Walker2d, making a total of 154 variants.

Installation
------------

1. Clone the repository `git clone https://github.com/mpatacchiola/imujoco.git` and set it as current folder with `cd imujoco`

2. Download the dataset files (approximately **3.4 GB**) from our page on [zenodo.com](https://zenodo.org/):
 
```
COMING SOON
```

3. Unzip the files into the `imujoco` folder: 

```
unzip dataset.zip
unzip policies.zip
unzip xml.zip
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



