import os
import sys
import json
import logging
import time
import gym
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tf_util



def train_val_split(data, train_size=0.9):
    n = data['observations'].shape[0]
    indices = np.random.permutation(n)
    train_id, val_id = indices[:int(n * train_size)], indices[int(n * train_size):]
    data_train = {'observations': data['observations'][train_id], 'actions': data['actions'][train_id]}
    data_val = {'observations': data['observations'][val_id], 'actions': data['actions'][val_id]}
    return data_train, data_val

def merge_data(data1, data2):
    return {
        'observations':np.concatenate((data1['observations'], data2['observations']), axis=0),
        'actions':np.concatenate((data1['actions'], data2['actions']), axis=0)
    }

def get_batch_generator(data, batch_size, shuffle=False):
    n = data['observations'].shape[0]
    if shuffle:
        indices = np.random.permutation(n)
        data = {'observations': data['observations'][indices], 'actions': data['actions'][indices]}
    for i in range(0, n, batch_size):
        yield {'observations':data['observations'][i:i + batch_size], 'actions': data['actions'][i:i + batch_size]}

def write_summary(value, tag, summary_writer, global_step): #生成标量图
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)

class Model(nn.Module):
    def __init__(self,FLAGS, algorithm, expert_returns=None, expert_policy_fn=None):
        super(Model,self).__init__()
        if not algorithm.strip().lower() in ['behavioral_cloning','dagger']:
            raise NotImplementedError('Algorithm {} not implement,check the initinization of model'.format(algorithm))
        self.FLAGS = FLAGS
        self.dropout = 1.0 - self.FLAGS['dropout']
        self.algorithm = algorithm
        self.expert_returns = expert_returns
        self.expert_policy_fn = expert_policy_fn
        #if self.algorithm == 'dagger':
        #    raise ValueError('No expect policy found')
        self.scope = self.algorithm + ' ' + time.strftime('%Y-%m-%d-%H-%M-%S')
        #print(self.scope)
        self.inputlayer=nn.Sequential(nn.Linear(self.FLAGS['input_dim'],
                                                       self.FLAGS['hidden_dims'][0]),nn.ReLU())
        self.layers=[]
        for i in range(1,len(self.FLAGS['hidden_dims'])):
            self.layers.append(nn.Sequential(nn.Linear(self.FLAGS['hidden_dims'][i-1],self.FLAGS['hidden_dims'][i]),
                                             nn.ReLU()))
        self.outputlayer=nn.Linear(self.FLAGS['hidden_dims'][-1],self.FLAGS['output_dim'],nn.Dropout(p=self.dropout))
        if self.FLAGS['loss'].strip().lower()=='l2':
            self.lossfun=nn.MSELoss(reduction='mean')
        elif self.FLAGS['loss'].strip().lower()=='smooth_l1':
            self.lossfun=nn.SmoothL1Loss(reduction='mean')
        else:
            raise NotImplementedError('loss function is not in definition')
        self.lr=self.FLAGS['learning_rate']



    def forward(self, x,y):
        out = x
        out = self.inputlayer(out)
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        out=self.outputlayer(out)
        loss = self.lossfun(out,y)
        return loss
    def get_val_loss(self,x,y):
        self.eval()
        out = x
        out = self.inputlayer(out)
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        out = self.outputlayer(out)
        loss = self.lossfun(out, y)
        self.train()
        return loss

    def get_predictions(self,observations):
        self.eval()
        out=torch.tensor(observations,dtype=torch.float)
        out = torch.unsqueeze(out,0)
        out = self.inputlayer(out)
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        out = self.outputlayer(out)
        out = torch.squeeze(out, dim=0)
        self.train()
        return out.detach().numpy()
    def evalutions(self,num_rollouts,max_steps,env):
        returns=[]
        observations = []
        for i in range(num_rollouts):
            obs = env.reset()
            done=False
            total = steps = 0
            while not done:
                action = self.get_predictions(obs)
                observations.append(obs)
                obs,r,done,_ = env.step(action)
                total+=r
                steps+=1
                if steps >= max_steps:
                    break
            print('reward of a run {} step {}'.format(total,steps))
            returns.append(total)
        return returns,observations
    def compare(self,num_rollouts,max_steps,env,expert_policy_fn):
        returns_exp=[]
        return_expert=[]
        observations = []
        with tf.Session():
            tf_util.initialize()
            for i in range(num_rollouts):
                obs = env.reset()
                #print('obs',obs,type(obs),obs[None,:])
                done=False
                total = steps = 0
                while not done:
                    action = self.get_predictions(obs)
                    #action = action[np.newaxis, :]
                    #print('action of exp ',action,type(action))

                    observations.append(obs)
                    #print('action type',type(action),action)
                    obs,r,done,_ = env.step(action)
                    total+=r
                    steps+=1
                    if steps >= max_steps:
                        break
                returns_exp.append((total,steps))
                obs = env.reset()
                done = False
                total = steps = 0
                while not done:
                    action = expert_policy_fn(obs[None,:])
                    #print('obs',obs)
                    action=action[0]
                    #print('action of expert ', action,type(action))
                    observations.append(obs)
                    #print('action type',type(action),action)
                    obs,r,done,_ = env.step(action)
                    total+=r
                    steps+=1
                    if steps >= max_steps:
                        break
                return_expert.append((total,steps))
        print('compartion: \n   exp:{}\n    expert:{}'.format(returns_exp,return_expert))

        return returns_exp,observations



























