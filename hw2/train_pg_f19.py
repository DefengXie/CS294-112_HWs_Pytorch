import numpy as np
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd

class policeNN(nn.Module):
    def __init__(self,input_dim, output_dim, n_layers,
                 hide_dim,activation='tanh', output_activation=None):
        super(policeNN,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.n_layers=n_layers
        self.hide_dim=hide_dim
        #self.dropout_prob=dropout_prob
        if activation not in ['relu','tanh']:
            raise NameError('activation function is not exist')
        #if output_activation not in ['relu', 'tanh','softmax']:
        #    raise NameError('output_activation function is not exist')
        self.activation=activation
        #self.output_activation=output_activation
        self.layers=[]
        self.input_layer=nn.Sequential(nn.Linear(self.input_dim,self.hide_dim),nn.ReLU() if activation == 'relu' else nn.Tanh())
        for i in range(1,n_layers):
            self.layers.append(nn.Sequential(nn.Linear(self.hide_dim,self.hide_dim),nn.ReLU() if activation == 'relu' else nn.Tanh()))
        self.outlayer=nn.Linear(self.hide_dim,self.output_dim)
        #self.dropout=nn.Dropout(p=self.dropout_prob)
    def forward(self,x):
        x=torch.tensor(x,dtype=torch.float)
        #y=torch.tensor(y)
        out=self.input_layer(x)
        for i in range(len(self.layers)):
            out=self.layers[i](out)
        out=self.outlayer(out)
        #out=self.dropout(out)
        return out


class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        self.pg_step = computation_graph_args['pg_step']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_return_args['gamma']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.nn_baseline = estimate_return_args['nn_baseline']
        self.normalize_advantages = estimate_return_args['normalize_advantages']

        self.policy=policeNN(self.ob_dim,self.ac_dim, self.n_layers, self.size)
        #self.policy_baseline = policeNN(self.ob_dim, 1, self.n_layers, self.size)
        self.baseline_value=policeNN(self.ob_dim,1, self.n_layers, self.size)
        self.optimizer_policy = optim.Adam(self.policy.parameters(),lr=self.learning_rate)
        self.optimizer_baseline_value=optim.Adam(self.baseline_value.parameters(),lr=self.learning_rate)
        #self.optimizer_policy_baseline = optim.Adam(self.policy_baseline.parameters(), lr=self.learning_rate)
        print('initial the agent : observation dim:{},action dim:{},discrete:{}'.format(self.ob_dim,self.ac_dim,self.discrete),)
        if self.discrete:
            self.sy_logstd = nn.Parameter(torch.randn(self.ac_dim,requires_grad=True))

    def process(self,sy_ob_no,sy_ac_na,sy_q_n,sy_adv_n):
        target_n = normalize(sy_q_n)
        target_n = torch.tensor(target_n,dtype=torch.float)
        sy_ob_no=torch.tensor(sy_ob_no,dtype=torch.float)
        sy_ac_na=torch.tensor(sy_ac_na,dtype=torch.float)
        sy_adv_n=torch.tensor(sy_adv_n,dtype=torch.float)
        #print('target_n target_n',target_n.shape)

        #sy_sampled_ac = torch.squeeze(torch.multinomial(sy_logits_na, 1), dim=1)#sample action??
        #sy_logits_na = sy_logits_na

        #print('sy_logits_na shape',sy_logits_na.shape,sy_ac_na.shape)
        #print('shape of adv:',sy_adv_n.shape)
        for i in range(self.pg_step):
            sy_logits_na = self.policy(sy_ob_no)
            if self.discrete:
                sy_logprob_n = nn.functional.cross_entropy(sy_logits_na, sy_ac_na.long(),reduction='none')
            else:
                sy=(sy_ac_na-sy_logits_na)/torch.exp(self.sy_logstd)
                sy_logprob_n=0.5 * torch.sum(sy.mul(sy),dim=1)
            #print('print sy_logprob_n.shape,sy_adv_n.shape ',sy_logprob_n.shape,sy_adv_n.shape)
            #temp=sy_logprob_n.mul(sy_adv_n)


            loss = torch.mean(sy_logprob_n.mul(sy_adv_n))
            #print('sy_logprob_n shape:',sy_logprob_n.shape)
            #print('loss shape and loss :',temp.shape,loss.shape,loss)
            print('have a loss:{}'.format(loss.item()))
            loss.backward()
            self.optimizer_policy.step() #update

        if self.nn_baseline:
            value_estimate=self.baseline_value(sy_ob_no)
            #value_estimate=torch.squeeze(value_estimate,dim=1)
            target_n=torch.unsqueeze(target_n,dim=1)
            #print('value_estimate,target_n',value_estimate.shape,target_n.shape)
            baseline_loss = torch.nn.functional.mse_loss(value_estimate,target_n)
            baseline_loss.backward()
            self.optimizer_baseline_value.step()




    def sample_trajectories(self,itr,env,epison): #sample the trajectory of policy choose
        timesteps_this_batch = 0
        paths = []

        while True:
            animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode,epison)
            paths.append(path)
            timesteps_this_batch += len(path["reward"])
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch
    def sample_trajectory(self, env, animate_this_episode,epison):
        ob = env.reset()
        obs, acs, rewards = [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            #====================================================================================#
            #                           ----------PROBLEM 3----------
            #====================================================================================#
            # YOUR CODE HERE

            sy_logits_na = self.policy(ob.reshape(1, -1))
            #print(sy_logits_na,sy_logits_na.shape)
            temp=torch.squeeze(sy_logits_na,dim=0)
            #gap=torch.max(torch.abs(0-temp[0]),temp[1]))

            #gapA=torch.abs(temp[0]) if temp[0].item()<0 else torch.tensor([0.])
            #gapB=torch.abs(temp[1]) if temp[1].item()<0 else torch.tensor([0.])
            #gap=torch.max(gapA,gapB)
            #temp=torch.norm(temp,dim=0)
            #temp=torch.tensor([temp[0]+gap,temp[1]+gap],dtype=torch.float)
            #print(temp)
            #ac = torch.multinomial(temp, 1)  # sample action??
            if self.discrete:
                temp = torch.squeeze(sy_logits_na, dim=0)
                a=np.random.random_sample()
                #print(a)
                if  a <= epison:
                    ac = torch.argmax(temp)
                else:
                    ac = torch.tensor(np.random.randint(0,2))
            else:
                ac=temp+torch.exp(self.sy_logstd).mul(torch.randn_like(temp))

            #print(ac)
            #print('pro',temp,'ac',ac)

            #ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob.reshape(1, -1)})
            #ac = ac[0]   #only the first one will be saved
            #if self.discrete:
            #    ac_input=ac.item
            #else:
            #    ac_input=ac.float()
            acs.append(ac.numpy())
            ob, rew, done, _ = env.step(ac.numpy())
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"observation" : np.array(obs, dtype=np.float32),
                "reward" : np.array(rewards, dtype=np.float32),
                "action" : np.array(acs, dtype=np.float32)}
        return path

    def sum_of_rewards(self, re_n):
        #num_paths = len(re_n)
        sum_of_path_lengths = sum(len(r) for r in re_n)
        q_n = np.empty(sum_of_path_lengths)
        i = 0
        for r in re_n:
            l = len(r)
            q_n[i + l - 1] = r[-1]
            for j in range(l - 2, -1, -1):
                q_n[i + j] = r[j] + self.gamma * q_n[i + j + 1]
            i += l
        if not self.reward_to_go:
            i = 0
            for r in re_n:
                l = len(r)
                q_n[i:i + l] = q_n[i]
                i += l
        return q_n

    def compute_advantage(self, ob_no, q_n):
        #q_n=torch.tensor(q_n,dtype=torch.float)
        ob_no=torch.tensor(ob_no,dtype=torch.float)
        if self.nn_baseline:
            b_n = self.baseline_value(ob_no)
            b_n = torch.squeeze(b_n,dim=1)
            b_n = b_n.detach().numpy()
            #print(b_n.shape,b_n)
            b_n = normalize(b_n, q_n.mean(), q_n.std())
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()
        return adv_n

    def estimate_return(self, ob_no, re_n):

        q_n = self.sum_of_rewards(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        if self.normalize_advantages:
            adv_n = normalize(adv_n)
        return q_n, adv_n



def train_PG(
        exp_name,
        env_name,
        n_iter,
        gamma,
        min_timesteps_per_batch,
        max_path_length,
        learning_rate,
        reward_to_go,
        animate,
        logdir,
        normalize_advantages,
        nn_baseline,
        seed,
        n_layers,
        size,
        pg_step):

    env = gym.make(env_name)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    max_path_length = max_path_length or env.spec.max_episode_steps
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
        'pg_step':pg_step
    }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'nn_baseline': nn_baseline,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)

    total_timesteps = 0
    epison=0.9
    data=[]
    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env,epison)
        #if epison<1:epison+= 5e-3
        total_timesteps += timesteps_this_batch
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [path["reward"] for path in paths]

        q_n, adv_n = agent.estimate_return(ob_no, re_n)
        agent.process(ob_no,ac_na,q_n,adv_n)
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        print('average return {} max return {} min return {} return std'.format(np.mean(returns),np.max(returns),np.min(returns),np.std(returns)))
        data.append(np.mean(returns))
    paint(data)
def paint(data):
    sns.set(style="darkgrid")
    sns.tsplot(data=data,color='g')
    plt.show()
    plt.savefig('./baseline.jpg')
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,default='CartPole-v0')
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.)
    parser.add_argument('--n_iter', '-n', type=int, default=400)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.002)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true',default=True)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true',default=True)
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=3)
    parser.add_argument('--size', '-s', type=int, default=128)
    parser.add_argument('--pg_step', '-ps', type=int, default=1)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline,
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                pg_step=args.pg_step
                )
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()


    for p in processes:
        p.join()





def pathlength(path):
    return len(path["reward"])

def normalize(values, mean=0., std=1.):
    values = (values - values.mean()) / (values.std() + 1e-8)
    return mean + (std + 1e-8) * values










if __name__ == "__main__":
    main()