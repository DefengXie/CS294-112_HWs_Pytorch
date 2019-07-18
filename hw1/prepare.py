import os
#os.system('sh demo.bash')
'''
import gym
import time
env = gym.make('Hopper-v2')
print(env.spec)
env.reset()
env.render()
time.sleep(100)
'''
'''
import mujoco_py
import os
mj_path, _ = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
sim.step()
print(sim.data.qpos)
'''


'''
import gym
import gym
import time
env = gym.make('Hopper-v2')

max_steps =1000
returns = []
observations = []
actions = []
for i in range(20):
    print('iter', i)
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
        action = 1
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        env.render()
        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
        if steps >= max_steps:
            break
    print(totalr)

import json

data = { 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5 }

jsons = json.dumps(data)
print(jsons)
print(jsons[5])
str=json.loads(jsons)

print(str)
print(str['a'])
'''
import os
import json
#os.system('bash run_experiment.sh')
with open('experiments/result.json','r') as f:
    test=f.read()
test=test.split('\n')
jsonall=[]
for item in test:
    f2=open('experiments/temp.json', 'w')
    f2.write(item)
    f2.close()
    f2 = open('experiments/temp.json', 'r')
    jsonall.append(json.load(f2))
    f2.close()
f3=open('experiments/output.md', 'a')
f3.write('|Task|Algorithm|Mean return |STD |Mean return(expert) |STD(expert)|\n|---|---|---|---|---|---|\n')
for item in jsonall:
    str="|{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|\n".format(item['name'],item['algorithm'],item['exp_mean'],item['exp_std'],item['expert_mean_return'],item['expert_std_return'])
    print(str)
    f3.write(str)
f3.close()



