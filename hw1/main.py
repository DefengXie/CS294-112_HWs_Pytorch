import os
import pickle
import json
import logging
import time
import numpy as np
import tensorflow as tf
import gym
import argparse

import load_policy
import torch
import torch.nn as nn
import tf_util
from Model import Model,train_val_split,get_batch_generator,merge_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)


# High-level options
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str,default='Hopper-v2')
parser.add_argument('--algorithm', type=str, default='behavioral_cloning',
                    help='Available algorithms: behavioral_cloning / dagger')
parser.add_argument('--gpu', type=int, default=0, help='Which gpu to use')
parser.add_argument('--mode', type=str, default='all', help='Available modes: all / test')
parser.add_argument('--reload', type=str, default='no')
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[100, 100,100])
parser.add_argument('--loss', type=str, default='l2')


# Hyperparameters for the model
parser.add_argument('--num_epochs', type=int, default=400,
                    help='Number of epochs to train. 0 means train indefinitely')
parser.add_argument('--learning_rate', type=float, default=0.00002)
parser.add_argument('--max_gradient_norm', type=float, default=10.0,
                    help='Clip gradients to this norm')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eval_every', type=int, default=5,
                    help='How many epochs to do per simulation / dagger')


# Hyperparameters for simulation / evaluation
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=10,
                    help='Number of expert roll outs')

FLAGS = vars(parser.parse_args())

def train(model,data_train,data_val,expert_policy_fn):
    env = gym.make(FLAGS['env_name'])
    num_rollouts = FLAGS['num_rollouts']
    max_steps = FLAGS['max_timesteps'] or 1000
    epoch=0
    returns_all=[]
    optim = torch.optim.Adam(model.parameters(),lr=FLAGS['learning_rate'], betas=(0.8, 0.999), eps=1e-7,weight_decay=0.0000)
    while FLAGS['num_epochs'] == 0 or epoch < FLAGS['num_epochs']:
        epoch+=1
        #start_time=time.time()
        for batch in get_batch_generator(data_train,FLAGS['batch_size'],shuffle=True):
            x=batch['observations']
            y=batch['actions']
            x=torch.tensor(x,dtype=torch.float)
            y=torch.tensor(y,dtype=torch.float)
            y=torch.squeeze(y,dim=1)
            #print('shap print..',x.shape,y.shape)
            train_loss=model(x,y)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=FLAGS['max_gradient_norm'])
            train_loss.backward()
            optim.step()
        if epoch%5 == 0:
            print("epoch {} train_loss : {}".format(epoch,train_loss))
        if epoch % FLAGS['eval_every'] == 0:
            x_val = torch.tensor(data_val['observations'],dtype=torch.float)
            y_val = torch.tensor(data_val['actions'],dtype=torch.float)
            y_val = torch.squeeze(y_val, dim=1)
            loss_val = model.get_val_loss(x_val,y_val)
            print("the {} epoch val loss is {}".format(epoch,loss_val))
            curr_returns,curr_observations = model.evalutions(num_rollouts,max_steps,env)
            returns_all.append(curr_returns)
            if FLAGS['algorithm']=='dagger':
                data_train, data_val = update_expert_data(np.array(curr_observations), data_train, data_val,expert_policy_fn)

    return returns_all

def update_expert_data(observations, data_train, data_val,expert_policy_fn):
    with tf.Session():
        tf_util.initialize()
        actions=expert_policy_fn(observations)
        actions_new=[]
        for item in actions:
            items=[]
            items.append(item)
            actions_new.append(items)

        d1, d2 = train_val_split({
            'observations': observations,
            'actions': np.array(actions_new)
        })
        return merge_data(data_train, d1), merge_data(data_val, d2)


DATA_DIR = os.path.join('.','expert_data')
EXPERT_POLICY_DIR = os.path.join('.','experts')
EXPERIMENT_DIR = os.path.join('.','experiments')

def main():
    #curr_dir = os.path.join(EXPERIMENT_DIR,FLAGS['env_name'])
    #if not os.path.exists(curr_dir):
    #    os.makedirs(curr_dir)

    with open(os.path.join(DATA_DIR,FLAGS['env_name']+'.pkl'),'rb') as f:
        data=pickle.load(f)

    FLAGS['input_dim'] = data['observations'].shape[-1]
    FLAGS['output_dim'] = data['actions'].shape[-1]

    expert_policy_fn = load_policy.load_policy(
        os.path.join(EXPERT_POLICY_DIR,FLAGS['env_name']+'.pkl')
    )


    with open(os.path.join(DATA_DIR, FLAGS['env_name'] + '.json'), 'r') as f:
        expert_returns = json.load(f)


    data_train, data_val = train_val_split(data)
    #print('data',data_train,data_val)


    if FLAGS['mode'] in ['all']:
        if FLAGS['algorithm'] == 'behavioral_cloning':
            model=Model(FLAGS,'behavioral_cloning')
        elif FLAGS['algorithm'] == 'dagger':
            model = Model(FLAGS, 'dagger')
        returns_all=train(model,data_train,data_val,expert_policy_fn)
        print(len(returns_all),len(expert_returns['returns']))
        mean=[]
        std=[]

        result_record = {
            'name': FLAGS['env_name'],
            'algorithm': FLAGS['algorithm'],
            'expert_mean_return': expert_returns['mean_return'],
            'expert_std_return': expert_returns['std_return']
        }
        for i in range(int(len(returns_all)/2),len(returns_all)):
            mean.append(np.mean(returns_all[i]))
            std.append(np.std(returns_all[i]))

        i=np.argmax(np.array(mean))
        result_record['exp_mean'] = mean[i]
        result_record['exp_std'] = std[i]
        print("best mean and std:",i,mean[i],std[i])
        #if len(best_mean)!= 0:
        #    i=np.argmax(best_mean)
        #    print('best point mean',best_mean[i],best_std[i])
        print('best return all is ',mean,std)
        print('expert return ',expert_returns)

        with open(os.path.join(EXPERIMENT_DIR,'result.json'), 'a') as f:
            json.dump(result_record,f)



if __name__ == '__main__':
    main()

