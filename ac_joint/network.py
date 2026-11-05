"""Vectorized ragged attention, algebraically identical to released TD3 nets."""
import sys
import numpy as np
import torch
import torch.nn.functional as F

from hyperbolic.prepare import ROOT
sys.path.insert(0,str(ROOT/'author_source'))
from model.TD3 import Actor,Critic,Skylark_TD3,ReplayBuffer


def encode(net,global_states,local_states):
    counts=[len(x) for x in local_states];maximum=max(counts)
    padded=np.zeros((len(counts),maximum,net.U.in_features),dtype=np.float32)
    mask=np.zeros((len(counts),maximum+1,1),dtype=np.float32);mask[:,0]=1
    for i,(count,values) in enumerate(zip(counts,local_states)):
        padded[i,:count]=values;mask[i,1:count+1]=1
    g=net.W(global_states[:,0,:])
    local=net.U(torch.as_tensor(padded,device=global_states.device))
    tokens=torch.cat((g[:,None,:],local),dim=1)
    global_tokens=g[:,None,:].expand_as(tokens)
    scores=F.leaky_relu(net.att(torch.cat((global_tokens,tokens),dim=-1)))
    scores=scores*torch.as_tensor(mask,device=scores.device)
    weights=scores/scores.sum(dim=1,keepdim=True)
    result=F.relu(torch.cat((weights[:,0,:]*g,(weights[:,1:,:]*local).sum(dim=1)),dim=-1))
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite attention state')
    return result


class FastActor(Actor):
    def forward(self,global_states,local_states):
        state=encode(self,global_states,local_states)
        return self.max_action*torch.sigmoid(self.l3(F.relu(self.l2(F.relu(self.l1(state))))))


class FastCritic(Critic):
    def forward(self,global_states,local_states,actions):
        sa=torch.cat((encode(self,global_states,local_states),actions),dim=1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(sa))))),self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
    def Q1(self,global_states,local_states,actions):
        sa=torch.cat((encode(self,global_states,local_states),actions),dim=1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))


def make_agent(frozen=False):
    agent=Skylark_TD3(8,4,7,1.,'cpu')
    agent.actor.__class__=agent.actor_target.__class__=FastActor
    agent.critic.__class__=agent.critic_target.__class__=FastCritic
    counter=[0]
    if frozen:
        agent.actor_optimizer.step=lambda *a,**kw:None
    else:
        step=agent.actor_optimizer.step
        def counted(*a,**kw):counter[0]+=1;return step(*a,**kw)
        agent.actor_optimizer.step=counted
    return agent,ReplayBuffer(7,max_size=4096),counter
