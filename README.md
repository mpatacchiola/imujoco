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


