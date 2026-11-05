import copy
import unittest
import numpy as np
import torch

from .experiment import ROOT
from .context import GeometryBank,Context
from .network import Actor,Critic,FastActor,FastCritic
from .search import baseline_local


class SearchTests(unittest.TestCase):
    def test_vectorized_attention_outputs_and_gradients(self):
        for cls,fast in [(Actor,FastActor),(Critic,FastCritic)]:
            torch.manual_seed(4)
            net=cls(8,4,7,1.,'cpu') if cls is Actor else cls(8,4,7,'cpu')
            accelerated=copy.deepcopy(net);accelerated.__class__=fast
            g=torch.rand(3,1,8)
            local=[np.random.default_rng(i).random((n,4)).tolist() for i,n in enumerate([1,7,50])]
            args=(g,local) if cls is Actor else (g,local,torch.rand(3,7))
            a,b=net(*args),accelerated(*args)
            if cls is Critic:a,b=torch.cat(a),torch.cat(b)
            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
            a.sum().backward();b.sum().backward()
            for p,q in zip(net.parameters(),accelerated.parameters()):
                torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)

    def test_all_baseline_budgets_include_four_common_probes(self):
        source=ROOT/'results/hyperbolic';out=ROOT/'results/ac_joint'
        bank=GeometryBank(source,out,'Pathbased','E');y=np.load(source/'inputs/Pathbased/canonical.npz')['y']
        for seed in [0,1,9]:
            frozen=np.load(source/'inputs/Pathbased'/f'seed{seed}.npz')
            supports=frozen['sample_ids'][frozen['reward_ids']]
            for method in ['Rand','MA-Random','BO-TPE','Anneal','PSO','GA','DE','Grid-30']:
                ctx=Context(bank,bank.index['global_key'],supports,y[np.sort(supports)])
                np.random.seed(seed)
                baseline_local(ctx,method,seed)
                self.assertEqual(len(ctx.events),30,(seed,method))
                self.assertEqual([e['linkage_id'] for e in ctx.events[:4]],np.roll(np.arange(4),seed%4).tolist())
                self.assertTrue(all(e['u']==.5 for e in ctx.events[:4]))


if __name__=='__main__':unittest.main()
