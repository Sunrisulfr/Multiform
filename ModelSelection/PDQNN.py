import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import random 

# env = gym.make("CartPole-v0")
# env = env.unwrapped
NUM_ACTIONS = 5 #env.action_space.n
NUM_STATES = 13 #env.observation_space.shape[0] #AVsonar+AVBearing 
ENV_A_SHAPE = 0   
class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, netStructure):
        super(Net, self).__init__()
        netType = netStructure
        self.act_type = []
        self.l = []
        self.Ffunc(netType+1)
        self.Nodes(netType)

        self.nettype = netType
        if netType == 1:
            layer1 = 2**(self.l[0])
            layer2 = 2**(self.l[1])
            # layer1 = 32*(l+1)
            # layer2 = 16*(l+1)
            
            self.fc1 = nn.Linear(NUM_STATES, layer1) 
            self.fc1.weight.data.normal_(0,0.1) 
            self.fc2 = nn.Linear(layer1,layer2) 
            self.fc2.weight.data.normal_(0,0.1)
            self.out = nn.Linear(layer2,NUM_ACTIONS)
            self.out.weight.data.normal_(0,0.1)
        elif netType == 2:
            layer1 = 2**(self.l[0])
            layer2 = 2**(self.l[1])
            layer3 = 2**(self.l[2])
            self.fc1 = nn.Linear(NUM_STATES, layer1) 
            self.fc1.weight.data.normal_(0,0.1)
            self.fc2 = nn.Linear(layer1,layer2)  
            self.fc2.weight.data.normal_(0,0.1)
            self.fc3 = nn.Linear(layer2,layer3)  
            self.fc3.weight.data.normal_(0,0.1)
            self.out = nn.Linear(layer3,NUM_ACTIONS)
            self.out.weight.data.normal_(0,0.1)

        elif netType == 3:
            layer1 = 2**(self.l[0])
            layer2 = 2**(self.l[1])
            layer3 = 2**(self.l[2])
            layer4 = 2**(self.l[3])
           
            self.fc1 = nn.Linear(NUM_STATES, layer1)
            self.fc1.weight.data.normal_(0,0.1) 
            self.fc2 = nn.Linear(layer1,layer2) 
            self.fc2.weight.data.normal_(0,0.1)
            self.fc3 = nn.Linear(layer2,layer3)  
            self.fc3.weight.data.normal_(0,0.1)
            self.fc4 = nn.Linear(layer3,layer4) 
            self.fc4.weight.data.normal_(0,0.1)
            
            self.out = nn.Linear(layer4,NUM_ACTIONS)
            self.out.weight.data.normal_(0,0.1)

        elif netType == 4:
            layer1 = 2**(self.l[0])
            layer2 = 2**(self.l[1])
            layer3 = 2**(self.l[2])
            layer4 = 2**(self.l[3])
            layer5 = 2**(self.l[4])
            self.fc1 = nn.Linear(NUM_STATES, layer1) 
            self.fc1.weight.data.normal_(0,0.1)
            self.fc2 = nn.Linear(layer1,layer2)  
            self.fc2.weight.data.normal_(0,0.1)
            self.fc3 = nn.Linear(layer2,layer3) 
            self.fc3.weight.data.normal_(0,0.1)
            self.fc4 = nn.Linear(layer3,layer4) 
            self.fc4.weight.data.normal_(0,0.1)
            self.fc5 = nn.Linear(layer4,layer5) 
            self.fc5.weight.data.normal_(0,0.1)
            self.out = nn.Linear(layer5,NUM_ACTIONS)
            self.out.weight.data.normal_(0,0.1)

        else:
            layer1 = 2**(self.l[0])
            self.fc1 = nn.Linear(NUM_STATES, layer1) 
            self.fc1.weight.data.normal_(0,0.1) 
            self.out = nn.Linear(layer1,NUM_ACTIONS)
            self.out.weight.data.normal_(0,0.1)

    def Ffunc(self, x):
        for i in range(x):
            if random.randint() == 1:
                self.act_type.append(F.relu)
            if self.act_type[i] == 2:
                self.act_type.append(F.leaky_relu)
            if self.act_type[i] == 3:
                self.act_type.append(F.sigmoid)
            if self.act_type[i] == 4:
                self.act_type.append(F.tanh)
        
    def Nodes(self, x):
        for i in range(x):
            self.l.append(random.randint(0,8))
                
    def forward(self,x):
        if self.nettype == 1:
            x = self.fc1(x)
            x = self.act_type[0](x)
            x = self.fc2(x)
            x = self.act_type[1](x)
        
        elif self.nettype == 2:
            x = self.fc1(x)
            x = self.act_type[0](x)
            x = self.fc2(x)
            x = self.act_type[1](x)
            x = self.fc3(x)
            x = self.act_type[2](x)
        
        elif self.nettype == 3:
            x = self.fc1(x)
            x = self.act_type[0](x)
            x = self.fc2(x)
            x = self.act_type[1](x)
            x = self.fc3(x)
            x = self.act_type[2](x)
            x = self.fc4(x)
            x = self.act_type[3](x)
        
        elif self.nettype == 4:
            x = self.fc1(x)
            x = self.act_type[0](x)
            x = self.fc2(x)
            x = self.act_type[1](x)
            x = self.fc3(x)
            x = self.act_type[2](x)
            x = self.fc4(x)
            x = self.act_type[3](x)
            x = self.fc5(x)
            x = self.act_type[4](x)
            
        else:
            x = self.fc1(x)
            x = self.act_type[0](x)
        
        action_prob = self.out(x)
        return action_prob


def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


class PDQN():
    # hyper-parameters
    BATCH_SIZE = 64
    LR = 0.00024 
    GAMMA = 0.9 
    
    MEMORY_CAPACITY = 5000
    Q_NETWORK_ITERATION = 100
    ENV_A_SHAPE = 0

    __instance = False

    __detect_loop = False
    __look_ahead = False
    __Trace = False

    # eval_net, target_net = Net(), Net()



    def __init__ (self, av_ID, netType) : 
        super(PDQN,self).__init__()
        self.__agentID = av_ID
        self.__numSpace=4  # 0-State 1-Action 2-Reward 3-New State 
        self.__numSonarInput = 5
        self.__positionInput = 2
        self.__numBearingInput = 8
        self.__numState = self.__numSonarInput + self.__numBearingInput
        self.__numAction=5
        self.__numReward=1

        self.EPISILO_START = 0.9
        self.EPISILO_END = 0.05
        self.EPISILO = 0.99
        self.EPISILO_DECAY = 400

        self.__prevReward = 0
        self.__end_state = False
        self.__currentBearing = 0

        self.__numInput = [0] * self.__numSpace    # self.__numSpace:0-State 1-Action 2-Reward 3-New State 
        self.__numInput[0] = self.__numState
        self.__numInput[1] = self.__numAction #5
        self.__numInput[2] = self.__numReward #2
        self.__numInput[3] = self.__numInput[0]

        # print("!!!")
        
        self.eval_net, self.target_net = Net(netType), Net(netType)
        

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.__numState * 2 + 2)) 
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss() #均方损失函数 loss_i = (x_i - y_i)^2


    def choose_action(self, state):
        # print (state)
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        # if np.random.randn() <= self.EPISILO:# greedy policy
        #     action_value = self.eval_net.forward(state)
        #     action = torch.max(action_value, 1)[1].data.numpy() #max(input, dim) 
        #     action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
        # else: # random policy
        #     action = np.random.randint(0,self.__numAction)
        #     action = action if self.ENV_A_SHAPE ==0 else action.reshape(self.ENV_A_SHAPE)
        action_value = self.eval_net.forward(state)
        action = torch.max(action_value, 1)[1].data.numpy() #max(input, dim) 
        action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
        # print(action)
        return action


    def store_transition(self, state, action, reward, next_state):
        # print(memory_counter)
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition

        self.memory_counter += 1

    def saveAction(self):
        if self.memory_counter > self.MEMORY_CAPACITY:
            np.savetxt("predictedActions1.txt", self.memory)
            exit(0)

    def loadAction(self):
        a = np.loadtxt("predictedActions1.txt")
        # print(a.shape)
        self.memory[0:3000,:] = a[0:3000,:]
        self.memory_counter = 3000
        # print(self.memory[0])
    
    def predictAction(self):
        batch_state = torch.FloatTensor(self.memory[:, :self.__numState])
        # batch_action = torch.LongTensor(self.memory[:, self.__numState:self.__numState+1].astype(int))
        # batch_reward = torch.FloatTensor(self.memory[:, self.__numState+1:self.__numState+2])
        # batch_next_state = torch.FloatTensor(self.memory[:,-self.__numState:])
        predictedActions = []
        for i in range(self.MEMORY_CAPACITY // 10):
            state = batch_state[i]
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy() #max(input, dim) 
            action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
            predictedActions.append(action)
        return predictedActions

    def learn(self):

        #update the parameters
        if self.learn_step_counter % self.Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) # return weight && bias
        self.learn_step_counter+=1

        #sample batch from memory
        # sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        sample_index = np.random.choice(min(self.memory_counter,self.MEMORY_CAPACITY), self.BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.__numState])
        batch_action = torch.LongTensor(batch_memory[:, self.__numState:self.__numState+1].astype(int)) 
        batch_reward = torch.FloatTensor(batch_memory[:, self.__numState+1:self.__numState+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-self.__numState:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action) #gather(input, dim, index, out=None, sparse_grad=False) → Tensor
        q_next = self.target_net(batch_next_state).detach() 
        q_target = batch_reward + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()#d_weights = [0] * n 
        loss.backward()
        self.optimizer.step()

    def checkAgent (self, outfile): #outfile是文件名
        # PrintWriter pw_agent = null 
        # boolean invalid
        try:
            # pw_agent = new PrintWriter (new FileOutputStream(outfile),True)
            pw_agent = open(outfile, "a+")
        except IOError:  
            print("Open"+outfile+"file error")
        pw_agent.write("Number of Memory : " + str(self.memory_counter) + "\n") 

        for j in range(self.memory_counter):
            pw_agent.write("Memory " + str(j) + "\n")
            for k in range(self.__numSpace):
                pw_agent.write("Space " + str(k) + " : ") 
                for i in range(self.__numInput[k]):
                    pw_agent.write(str(self.memory[j,i]) + ", ", end = '')
                pw_agent.write("\n") 
        pw_agent.close ()

    def saveAgent (self, outfile):
        try:
            pw_agent = open(outfile, "w+")
        except IOError:
            print("Open"+outfile+"file error")

        pw_agent.write("Number of Memory : " + str(self.memory_counter) + "\n")
        for j in range(self.memory_counter):
            print(self.memory_counter)
            pw_agent.write ("Memory " + str(j) + "\n")
            for k in range(self.__numSpace):
                pw_agent.write("Space " + str(k) + " : ")
                for i in range(self.__numInput[k]):
                    pw_agent.write(str(self.memory[j,i]) + ", ")
                pw_agent.write("\n")
        pw_agent.close ()

    
    def savePreAgent(self):
        torch.save(self.target_net.state_dict(),"./preparameter.pkl")
        print("Save Model Success!")
    
    def loadPreAgent(self):
        self.target_net.load_state_dict(torch.load('./preparameter.pkl'))
        self.eval_net.load_state_dict(torch.load('./preparameter.pkl'))
        print("Load  Prediction Model Success!")


    def getMemory_counter(self):
        return( self.memory_counter )

    def getCapacity (self):
        return (self.MEMORY_CAPACITY)

    def setTrace (self, t):
        self.__Trace = t

    def setPrevReward (self, r):
        self.__prevReward = r

    def getPrevReward (self):
        return (self.__prevReward)

    def turn(self, d : int ):
        self.__currentBearing = ( self.__currentBearing  +  d  +  8 ) % 8
