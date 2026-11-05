"""Mixed linkage/threshold search; training receives Context support labels only."""
import numpy as np
import torch

from .network import make_agent
from .context import warmup
from .trees import LINKAGES


def move(linkage,u,action,step,bounds):
    choice=int(np.argmax(action));bump=False
    if choice==0 or choice==1:
        raw=u+(-step if choice==0 else step)
        new=float(np.clip(raw,*bounds));bump=raw<bounds[0] or raw>bounds[1]
        return linkage,new,bump,choice==6
    if 2<=choice<=5:linkage=choice-2
    return linkage,u,bump,choice==6


def episode(ctx,agent,buffer,center,step,bounds,episode_number,train=True):
    linkage,u=center
    labels=ctx.query(linkage,u,'train' if train else 'detect')
    state=ctx.state(labels,linkage,u,bounds)
    transitions=[];stopped=bump=False;last_choice=-1
    for i in range(30):
        if (i>=2 and stopped) or bump or i==29:break
        action=agent.select_action(state)
        if train and episode_number!=1:
            action=(action+np.random.normal(0,agent.expl_noise,size=7)).clip(0,1)
        else:action=action.clip(0,1)
        linkage,u,bump,stopped=move(linkage,u,action,step,bounds)
        labels=ctx.query(linkage,u,'train' if train else 'detect')
        next_state=ctx.state(labels,linkage,u,bounds)
        if train:transitions.append([state,action,next_state,ctx.score()])
        state=next_state
    if train and transitions:
        future=transitions[-1][3];endpoint=future
        rewards=[]
        for transition in reversed(transitions):
            future=max(future,transition[3]);rewards.append(.2*endpoint+.8*future)
        rewards.reverse()
        for j,(transition,reward) in enumerate(zip(transitions,rewards)):
            if buffer.size>=buffer.max_size:raise RuntimeError('Replay unexpectedly wrapped')
            buffer.add(transition[0],transition[1],transition[2],reward,float(j==len(transitions)-1))
        if buffer.size>=16:
            for _ in transitions:agent.learn(buffer,16)
    return linkage,u,len(transitions)


def train_local(ctx,method,seed,path):
    # The user explicitly selected complete default training with retrospective
    # first-30-request display, not a 30-query hard training budget.
    policies=[make_agent('FrozenActor' in method) for _ in range(3)]
    initials=[{k:v.detach().clone() for k,v in a.actor.state_dict().items()} for a,_,_ in policies]
    warmup(ctx,seed)
    best=max(ctx.events,key=lambda e:e['score']);start=(best['linkage_id'],best['u'])
    test_start=start
    for (agent,buffer,_),step in zip(policies,[.2,.04,.008]):
        bounds=(max(0,test_start[1]-2*step),min(1,test_start[1]+2*step))
        a,u,_=episode(ctx,agent,buffer,test_start,step,bounds,1,False);test_start=(a,u)
    episode_logs=[]
    for layer,((agent,buffer,_),step) in enumerate(zip(policies,[.2,.04,.008])):
        best=max(ctx.events,key=lambda e:e['score']);center=(best['linkage_id'],best['u'])
        bounds=(max(0,center[1]-2*step),min(1,center[1]+2*step))
        baseline=best['score'];history=[baseline];improved=False
        for number in range(1,15):
            before=len(ctx.events)
            _,_,transitions=episode(ctx,agent,buffer,center,step,bounds,number,True)
            current=max(e['score'] for e in ctx.events)
            improved=improved or current>baseline
            history.append(current)
            episode(ctx,agent,buffer,center,step,bounds,number,False)
            episode_logs.append(dict(layer=layer,episode=number,training_requests=len(ctx.events)-before,
                                     transitions=transitions,critic_updates=agent.total_iteration,
                                     best_support_reward=current))
            if len(history)>3 and history[-1]==history[-2]==history[-3] and history[-1]!=history[0]:break
        if not improved:break
    models=[]
    for i,(agent,buffer,count) in enumerate(policies):
        unchanged=all(torch.equal(v,initials[i][k]) for k,v in agent.actor.state_dict().items())
        if 'FrozenActor' in method and not unchanged:raise AssertionError('Frozen actor changed')
        models.append(dict(actor=agent.actor.state_dict(),critic=agent.critic.state_dict(),
                           actor_updates=count[0],critic_updates=agent.total_iteration,
                           actor_unchanged=unchanged,replay_size=buffer.size))
    torch.save(models,path)
    return dict(episodes=episode_logs,actor_updates=[m['actor_updates'] for m in models],
                critic_updates=[m['critic_updates'] for m in models],
                actor_unchanged=[m['actor_unchanged'] for m in models])


def baseline_local(ctx,method,seed,budget=30):
    order=np.roll(np.arange(4),seed%4);warmup(ctx,seed)
    rng=np.random.default_rng(seed)
    if method in ['Rand','MA-Random']:
        for _ in range(budget-4):
            linkage=int(rng.integers(4))
            u=float(rng.choice(.5+np.arange(-62,63)*.008)) if method=='MA-Random' else float(rng.uniform())
            ctx.query(linkage,u)
    elif method in ['BO-TPE','Anneal']:
        from hyperopt import fmin,hp,tpe,anneal,Trials
        from hyperopt.fmin import generate_trials_to_calculate
        # Seed both the surrogate and total budget with exactly the four common probes.
        initial=[{'linkage':int(l),'u':.5} for l in order]
        trials=generate_trials_to_calculate(initial)
        start_index=len(ctx.events)
        replay_index=[0]
        def objective(p):
            linkage=int(p['linkage']);u=float(p['u'])
            if replay_index[0]<4:
                event=ctx.events[replay_index[0]]
                if event['linkage_id']!=linkage or event['u']!=u:raise AssertionError('Hyperopt warmup order differs')
                replay_index[0]+=1;return -event['score']
            ctx.query(linkage,u);return -ctx.score()
        fmin(objective,{'linkage':hp.choice('linkage',[0,1,2,3]),'u':hp.uniform('u',0,1)},
             algo=tpe.suggest if method=='BO-TPE' else anneal.suggest,max_evals=budget-4,
             trials=trials,rstate=rng,show_progressbar=False,verbose=False)
    else:
        # No ordinal encoding of categorical linkage: enumerate its four values,
        # then optimize u independently within each, pooling support incumbents.
        per=[budget//4+(i<budget%4) for i in range(4)]
        for index,linkage in enumerate(order):
            remaining=per[index]-1
            if method=='Grid-30':
                values=[]
                for value in np.r_[np.linspace(0,1,per[index]-1),np.linspace(0,1,2*per[index])]:
                    if abs(value-.5)>1e-15 and value not in values:values.append(float(value))
                    if len(values)==remaining:break
                for u in values:ctx.query(int(linkage),u)
                continue
            class BudgetComplete(Exception):pass
            counter=[0]
            def objective(params):
                if counter[0]>=remaining:raise BudgetComplete()
                counter[0]+=1;ctx.query(int(linkage),float(np.clip(np.asarray(params).reshape(-1)[0],0,1)))
                return -ctx.score()
            try:
                if method=='PSO':
                    from sko.PSO import PSO
                    optimizer=PSO(objective,n_dim=1,pop=4,max_iter=10,lb=[0.],ub=[1.],w=.8,c1=.5,c2=.5)
                elif method=='GA':
                    from sko.GA import GA
                    optimizer=GA(objective,n_dim=1,size_pop=4,max_iter=10,lb=[0.],ub=[1.],precision=.008,prob_mut=.001)
                elif method=='DE':
                    from sko.DE import DE
                    optimizer=DE(objective,n_dim=1,size_pop=4,max_iter=10,lb=[0.],ub=[1.],F=.5,prob_mut=.3)
                else:raise ValueError(method)
                optimizer.run()
            except BudgetComplete:pass
            if counter[0]!=remaining:raise AssertionError('Optimizer exhausted early')
    if len(ctx.events)!=budget:raise AssertionError(f'Baseline budget mismatch: {method}, {len(ctx.events)}')
    return dict(episodes=[],actor_updates=[],critic_updates=[],actor_unchanged=[])
