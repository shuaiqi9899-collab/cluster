"""Two numeric-coordinate DBSCAN search, with complete audited TD3 training."""
import numpy as np
import torch
from ac_joint.network import Skylark_TD3,ReplayBuffer,FastActor,FastCritic


def make_agent(frozen):
 agent=Skylark_TD3(7,4,5,1.,'cpu');agent.actor.__class__=agent.actor_target.__class__=FastActor;agent.critic.__class__=agent.critic_target.__class__=FastCritic
 count=[0]
 if frozen:agent.actor_optimizer.step=lambda *a,**k:None
 else:
  step=agent.actor_optimizer.step
  def counted(*a,**k):count[0]+=1;return step(*a,**k)
  agent.actor_optimizer.step=counted
 return agent,ReplayBuffer(5,max_size=4096),count


def warmup(ctx):
 for x,y in [(0.5,0.5),(.25,.25),(.75,.25),(.25,.75)]:ctx.query(x,y)


def episode(ctx,agent,buffer,center,step,bounds,number,train):
 x,y=center;labels=ctx.query(x,y,'train' if train else 'detect');state=ctx.state(labels,x,y,bounds);transitions=[];stopped=bump=False
 for i in range(29):
  if (i>=2 and stopped) or bump:break
  action=agent.select_action(state)
  if train and number!=1:action=(action+np.random.normal(0,agent.expl_noise,size=5)).clip(0,1)
  else:action=action.clip(0,1)
  choice=int(np.argmax(action));point=[x,y];stopped=choice==4
  if choice<4:
   dim=choice//2;raw=point[dim]+(-step if choice%2==0 else step);point[dim]=float(np.clip(raw,*bounds[dim]));bump=raw<bounds[dim][0] or raw>bounds[dim][1]
  x,y=point;labels=ctx.query(x,y,'train' if train else 'detect');next_state=ctx.state(labels,x,y,bounds)
  if train:transitions.append([state,action,next_state,ctx.score()])
  state=next_state
 if train and transitions:
  future=endpoint=transitions[-1][3];rewards=[]
  for t in reversed(transitions):future=max(future,t[3]);rewards.append(.2*endpoint+.8*future)
  for i,(t,reward) in enumerate(zip(transitions,reversed(rewards))):
   if buffer.size>=buffer.max_size:raise RuntimeError('Replay wrapped')
   buffer.add(t[0],t[1],t[2],reward,float(i==len(transitions)-1))
  if buffer.size>=16:
   for _ in transitions:agent.learn(buffer,16)
 return x,y,len(transitions)


def train_local(ctx,method,seed,path):
 policies=[make_agent('FrozenActor' in method) for _ in range(3)]
 initial=[{k:v.detach().clone() for k,v in a.actor.state_dict().items()} for a,_,_ in policies]
 warmup(ctx);best=max(ctx.events,key=lambda e:e['score']);test_start=tuple(best['coords'])
 for (agent,buffer,_),step in zip(policies,[.2,.04,.008]):
  bounds=[(max(0,c-2*step),min(1,c+2*step)) for c in test_start];x,y,_=episode(ctx,agent,buffer,test_start,step,bounds,1,False);test_start=(x,y)
 logs=[]
 for layer,((agent,buffer,_),step) in enumerate(zip(policies,[.2,.04,.008])):
  best=max(ctx.events,key=lambda e:e['score']);center=tuple(best['coords']);bounds=[(max(0,c-2*step),min(1,c+2*step)) for c in center]
  baseline=best['score'];history=[baseline];improved=False
  for number in range(1,15):
   before=len(ctx.events);_,_,transitions=episode(ctx,agent,buffer,center,step,bounds,number,True)
   current=max(e['score'] for e in ctx.events);improved=improved or current>baseline;history.append(current)
   episode(ctx,agent,buffer,center,step,bounds,number,False)
   logs.append(dict(layer=layer,episode=number,training_requests=len(ctx.events)-before,transitions=transitions,critic_updates=agent.total_iteration,best_support_reward=current))
   if len(history)>3 and history[-1]==history[-2]==history[-3] and history[-1]!=history[0]:break
  if not improved:break
 models=[]
 for i,(agent,buffer,count) in enumerate(policies):
  unchanged=all(torch.equal(v,initial[i][k]) for k,v in agent.actor.state_dict().items())
  if 'FrozenActor' in method and not unchanged:raise AssertionError('Frozen Actor changed')
  models.append(dict(actor=agent.actor.state_dict(),critic=agent.critic.state_dict(),actor_updates=count[0],critic_updates=agent.total_iteration,actor_unchanged=unchanged,replay_size=buffer.size))
 torch.save(models,path)
 return dict(episodes=logs,actor_updates=[m['actor_updates'] for m in models],critic_updates=[m['critic_updates'] for m in models],actor_unchanged=[m['actor_unchanged'] for m in models])


def baseline_local(ctx,method,seed,budget=30):
 warmup(ctx);rng=np.random.default_rng(seed);remaining=budget-4
 if method in ['Rand','MA-Random']:
  for _ in range(remaining):
   p=rng.choice(.5+np.arange(-62,63)*.008,size=2) if method=='MA-Random' else rng.uniform(size=2);ctx.query(*p)
 elif method in ['BO-TPE','Anneal']:
  from hyperopt import fmin,hp,tpe,anneal
  from hyperopt.fmin import generate_trials_to_calculate
  initial=[dict(x=e['coords'][0],y=e['coords'][1]) for e in ctx.events];trials=generate_trials_to_calculate(initial);replay=[0]
  def objective(p):
   if replay[0]<4:e=ctx.events[replay[0]];replay[0]+=1;return -e['score']
   ctx.query(p['x'],p['y']);return -ctx.score()
  fmin(objective,dict(x=hp.uniform('x',0,1),y=hp.uniform('y',0,1)),algo=tpe.suggest if method=='BO-TPE' else anneal.suggest,max_evals=remaining,trials=trials,rstate=rng,show_progressbar=False,verbose=False)
 else:
  class BudgetComplete(Exception):pass
  calls=[0]
  def objective(p):
   if calls[0]>=remaining:raise BudgetComplete()
   calls[0]+=1;ctx.query(*np.clip(np.asarray(p).reshape(-1),0,1));return -ctx.score()
  try:
   if method=='PSO':
    from sko.PSO import PSO
    optimizer=PSO(objective,n_dim=2,pop=10,max_iter=10,lb=[0,0],ub=[1,1],w=.8,c1=.5,c2=.5)
   elif method=='GA':
    from sko.GA import GA
    optimizer=GA(objective,n_dim=2,size_pop=10,max_iter=10,lb=[0,0],ub=[1,1],precision=[.008,.008],prob_mut=.001)
   elif method=='DE':
    from sko.DE import DE
    optimizer=DE(objective,n_dim=2,size_pop=10,max_iter=10,lb=[0,0],ub=[1,1],F=.5,prob_mut=.3)
   else:raise ValueError(method)
   optimizer.run()
  except BudgetComplete:pass
  assert calls[0]==remaining
 assert len(ctx.events)==budget,(method,len(ctx.events))
 return dict(episodes=[],actor_updates=[],critic_updates=[],actor_unchanged=[])
