# REINFORCEMENT LEARNING MODEL FOR GAMING
## Everything in a nutshell

Welcome aboard friends, the focus of the project was to implement an RL algorithm to create an AI agent capable of playing the popular Super Mario Bros game. Leveraging the OpenAI Gym environment, I used the Proximal Policy Optimization (PPO) algorithm to train the agent. 

![WhatsApp Image 2023-06-11 at 3 30 09 PM](https://github.com/nikhilgrad/super_mario/assets/117857370/d189594b-bd5c-42ea-8aa3-d98a46e1ab17)
                                      *A frame from Super Mario Bros. Environment*


The primary objective of the project was to explore the application of reinforcement learning techniques in gaming scenarios. By developing this AI agent, I aimed to gain a deeper understanding of the underlying concepts and challenges associated with reinforcement learning.
To accomplish this, I first familiarized myself with the fundamentals of reinforcement learning, including concepts such as rewards, policies, and value functions. I then proceeded to build the agent's environment using OpenAI Gym, which provided the necessary tools and libraries for the Mario game.
Next, I implemented the Proximal Policy Optimization algorithm, which is a state-of-the-art reinforcement learning technique. This algorithm allows for continuous updates of the agent's policy, striking a balance between exploration and exploitation to optimize performance. Through iterations of training and fine-tuning, I observed improvements in the agent's ability to navigate the game environment and achieve higher scores.

## Let’s dive into the code

**First we need to install and import all the dependencies** like the gaming environment itself  and `nes-py` which is an emulator to play Mario.

```
!pip install gym_super_mario_bros==7.3.0 nes_py
```

Now that we have finished installing, we need to import the game in the jupyter notebook. Also we need to import the `JoypadSpace` which acts like a wrapper and feeds the action to the environment like a joystick. The SIMPLE_MOVEMENT is needed to simplify the controls so that our agent has to choose only from a small set of movements. 

```
import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
```

Install `Pytorch` (check the version suitable for your system) and install `stable-baselines3` library which contains many RL algorithms which we need to train our model

```
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

!pip install stable-baselines3[extra]
```

`GrayScaleObservation` changes the game from colour image (RGB) to grayscale so that our processing becomes faster as we need to deal with less data. `VecFrameStack` allows us to work with our stacked enviroments first by letting us know the information of previous frames and second by stacking the frames. `DummyVecEnv` transforms our model so that we can pass it to our AI model. 

```
from gym.wrappers import GrayScaleObservation

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
```
**Now little pre-processing** is needed which includes setting up the environment and doing the below given steps.
1. Make the base game environment
```
env = gym_super_mario_bros.make('SuperMarioBros-v0')
```

2. Load the Simplified controler with Joypad wrapper in our game so that we just have few actions to take care of
```
env = JoypadSpace(env, SIMPLE_MOVEMENT)
```

3. Grayscale the environment to make our processing faster
```
env = GrayScaleObservation(env, keep_dim=True)
```

4. Wrap inside the Dummy environment
```
env = DummyVecEnv([lambda:env])
```

5. Stack 4 frames of our environment and `channels_order="last"` is for stacking along the last dimension
```
env = VecFrameStack(env, 4, channels_order="last")
```

Now to train our RL model(Our AI) we are going to use **PPO (Proximal Policy Optimization) algorithm**. 
PPO (Proximal Policy Optimization) is an algorithm which belongs to the family of policy gradient methods, which are a class of reinforcement learning algorithms that optimize the policy directly. PPO uses an approach called "on-policy" learning, where it collects experience by interacting with the environment using the current policy and updates the policy based on the collected data. The policy is updated in a way that maximizes the expected reward while also ensuring that the policy update does not deviate too far from the previous policy.

### Build and Train the AI (RL Model)

Now we need to import `os` for file path management and `import PPO` algorithm to train our model, additionally we have to `import Base Callback` for saving models and to continue from where we stopped.

```
import os 

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import BaseCallback
````

To do the callback and saving of the training and log files we need to specify a folder 

```
CHECKPOINT_DIR = './train'

LOG_DIR = './logs'
```
**Now we will set 2 hyperparameters, namely, the clip range and learning rate**

The clipping range is used to restrict the extent to which the updated policy can deviate from the old policy. The value of the clipping range is typically set as a small positive number, such as 0.1 or 0.2. A smaller clipping range restricts the policy update more tightly, while a larger range allows for more significant updates. The specific value of the clipping range depends on the problem domain, the characteristics of the environment, and the desired trade-off between exploration and exploitation.

The learning rate determines the step size at each update during optimization. A higher learning rate can lead to faster convergence but may risk overshooting and instability. A lower learning rate can provide more stable updates but may slow down convergence. The ideal learning rate depends on the problem and network architecture and is typically tuned through experimentation. Now, I tried different learning rates some of them quite large like 0.0001(this gave really bad results) to very small values like 0.000001 and at the end I settled for 0.000003. 

As these are some of the many hyperparameters, they can be tuned according to the environment and by experimentation. 

```
def custom_clip_range(a):
    a = 0.2
    return a  

def custom_lr_schedule(lr):
    lr = 0.000003
    return lr  
```

We need function to start the training, to save and also to callback the saved model. The below class `TrainAndLoggingCallback` was taken from a similar model made by [Nicholas Renotte](https://github.com/nicknochnack/MarioRL). His program helped me a lot in making this one. One of the key difference is that his program does not have a  code to resume the training again from where it was stopped and that is because he didn't need to do that as he might have a GPU and can train the model quite a few million times in one single go. But this was not certainly my case as I had to train the model over CPU:( and this would have taken days of uninterrupted run if not weeks. So a way out was to train the model in multiple turns and for that purpose I changed the code to suit my needs. We also changed a very crucial thing, again to suit our needs but more on this later.

```
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Save the model and track training progress
        if self.num_timesteps % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.num_timesteps))
            self.model.save(model_path)

        return True
```

To modify the given code to resume training from where it stopped, we need to incorporate the following changes:

1. Save the total number of steps completed during training. This value will be used to continue training from the same point later.
2. Add code to load the previously trained model if it exists.
3. Adjust the starting step count and the total number of training steps based on the previous training progress.

```
# Check if a previously trained model exists
if os.path.exists('./train/best_model.zip'):

    # Load the pre-trained model
    model_start = PPO.load('./train/best_model.zip', env, tensorboard_log=LOG_DIR, custom_objects={'clip_range': custom_clip_range, 'learning_rate': custom_lr_schedule})
    
    # Get the total number of steps completed during the previous training
    total_steps_completed = model_start.num_timesteps
    
    model = PPO.load('./train/best_model.zip', env, tensorboard_log=LOG_DIR, custom_objects={'clip_range': custom_clip_range, 'learning_rate': custom_lr_schedule})

    # Adjust the starting step count and the total number of training steps
    starting_step = total_steps_completed + 1
    total_training_steps = starting_step + 100000  # Resume training for 100,000 steps
    
else:
    # Create a new model if no pre-trained model exists
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=custom_lr_schedule, n_steps=512)
   
    # Set the starting step count and the total number of training steps
    starting_step = 1
    total_training_steps = 100000  # Train for 100,000 steps

# Call back the trained and logged model after every 5000 steps (takes 150MB space for one run logged data for 5k steps) and save to CHECKPOINT_DIR.
callback = TrainAndLoggingCallback(check_freq=5000, save_path=CHECKPOINT_DIR)
```

**And now we train our model.**

```
model.learn(total_timesteps=total_training_steps, callback=callback, reset_num_timesteps=False)
```

This is another change from program written by [Nicholas Renotte](https://github.com/nicknochnack/MarioRL) as we are resuming from where the previous model stopped, we have set the `reset_num_timesteps` to **False** which by default is always set to **True**.

### Combining to 2 different models

I added this part because the model was not performing upto the mark even after training the model 1.7 million times which made sense but what I felt bad about was the inefficiency of my CPU which took around 6 days to train the model this many times. I would like everyone to see how my models faired upto 500k training steps.



https://github.com/nikhilgrad/super_mario/assets/117857370/6ad4880c-cd15-445f-82bf-09f8f7855349

*This is the model after 10k trainings*


                                   
https://github.com/nikhilgrad/super_mario/assets/117857370/1eb3d3a1-47d7-446d-9b48-03ee4a5b0bdb

*This is the model after 500k trainings*



Now the model from here were a bit distracted which means there is a good chance that due to more exploration and learning on an inefficient hardware(which reduced the frames per second to as low as 6) they might have undergone catastrophic forgetting. Catastrophic forgetting refers to a situation where a neural network or learning algorithm forgets previously learned information when it is trained on new data or tasks. This can be problematic in reinforcement learning when an agent trained on one task starts learning a new task and loses the knowledge or performance gained from the previous task. The low value of FPS(Frames Per Second) also results in slow learning which can also be a reason for the model not giving good result.

On multiple obsevations I saw one very important difference between the 500k model and the 1.7M model and that was, the 500k model was good at jumping and many a times without any good reason. This might be due to the fact that the model learnt, which is, if Mario(the Agent) would just keep on jumping and move right it would gather more rewards. On the other hand the 1.7M model though not as good a jumper as 500k, was better at taking decisions like jumping over those rhombohedral monsters and tackling the bricks to get gold coins and in turn increasing the reward. So my idea was, what if we could combine both the models and check if the combined model with both the properties was any better. And so the below code atually shows how to combine 2 different models.

For combining 2 different models we need to:
- Load models we want to combine
- Assign weights for the models
- Get the policy parameters from the models
- Combine the policy parameters with the specified weights
- Create a new model with the combined policy parameters
- Save the new combined model in the train directory

```
model1 = PPO.load('./train/best_model.zip', env, custom_objects={'clip_range': custom_clip_range, 'learning_rate': custom_lr_schedule})
model2 = PPO.load('./train/best_model_500000.zip', env, custom_objects={'clip_range': custom_clip_range, 'learning_rate': custom_lr_schedule})

weight_model1 = 0.6  
weight_model2 = 0.4  

policy_params1 = model1.policy.state_dict()
policy_params2 = model2.policy.state_dict()

combined_policy_params = {}
for param_name in policy_params1.keys():
    combined_policy_params[param_name] = weight_model1 * policy_params1[param_name] + weight_model2 * policy_params2[param_name]

combined_model = PPO('CnnPolicy', env=model1.env) 
combined_model.policy.load_state_dict(combined_policy_params)

combined_model.save("./train/combined_model_best*500000.zip")
```

The result of the combined model is shown below as I was unable to upload the complete video the video shows just the small part of the model run and also keep an eye on the score in the top-left corner.



https://github.com/nikhilgrad/super_mario/assets/117857370/1ba4b368-c27e-44a0-b751-8ab09125bd4b

*This is the combined model which was made by combining the 1.7M model and the 500k model with weights 0.6 and 0.4 respectively*



The weights given to each model is also a hyperparameter i.e it can also be tuned according to our needs. I tried several different weights for both the models and also tried combining 3 models with weights assigned to each one of them. But in my case I found that combining 2 models and that too with the weights of 0.6 and 0.4 actually gives the best possible result. The video shows Mario (our agent) falling in the hole, this was actually happening continously even after running(not Training) the model several times. One possible reason could be that since both the models used to create the combined model never had the chance to train over a hole the resulting combined model also performs poorly as it has no prior experience of crossing it. So to solve this issue I trained the  combined model quite a few more times.

The below code is needed to render, that is, to show our game environment on our screen which is possible only after loading the model whose performance we want to see.

```
# Load the new combined model
combined_model = PPO.load('./train/combined_model_best*500000', custom_objects={'clip_range': custom_clip_range, 'learning_rate': custom_lr_schedule})

#Starting our game
state = env.reset()

#Loop through the game
while True:
    # we get two values of which we need only one, so we put a underscore to neglect the extra value
    action, _ = combined_model.predict(state)
    action, reward, done, info = env.step(action)
    env.render()
```

To stop the environment running in a loop, that is  to stop the game, press the **"interrupt the kernel"** button next to **"Run"** and run the below code.

```
#To close the game environment
env.close()
```


**Below is the final model that I got after 7 days of this learning journey:)**



https://github.com/nikhilgrad/super_mario/assets/117857370/61c9d0e4-6762-4fbf-89e3-1274f9c16a8b

*The final model*











