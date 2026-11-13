import sys, time, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from common import *
from config import SEEDS, TRAIN_END, VAL_END

torch.set_num_threads(2)
model_name = sys.argv[1]
max_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else len(SEEDS)
B = load_base(); y, t = B['y'], B['t']
tr, va, te = splits(B)
sc = StandardScaler().fit(B['X'][t <= TRAIN_END])
Z = np.clip(sc.transform(B['X']), -10, 10)
Z = np.hstack([np.nan_to_num(Z, nan=0.0), np.isnan(Z).any(1, keepdims=True).astype(np.float32)])
Xall = torch.tensor(Z, dtype=torch.float32)
src = np.r_[B['src'], B['dst']]; dst = np.r_[B['dst'], B['src']]
HID, DROP = 128, 0.3

def subgraph(mask):
    nodes = np.where(mask)[0]
    loc = -np.ones(len(t), dtype=np.int64); loc[nodes] = np.arange(len(nodes))
    keep = mask[src] & mask[dst]
    return dict(nodes=nodes, loc=loc, ei=torch.tensor(np.vstack([loc[src[keep]], loc[dst[keep]]]), dtype=torch.long), n=len(nodes))

def prep_graph(g, kind):
    n, ei = g['n'], g['ei']
    if kind == 'GCN':
        loops = torch.arange(n); e = torch.cat([ei, torch.stack([loops, loops])], 1)
        deg = torch.zeros(n).index_add_(0, e[1], torch.ones(e.shape[1]))
        w = deg[e[0]].pow(-0.5) * deg[e[1]].pow(-0.5)
        g['A'] = torch.sparse_coo_tensor(torch.stack([e[1], e[0]]), w, (n, n)).coalesce().to_sparse_csr()
    elif kind == 'SAGE':
        deg = torch.zeros(n).index_add_(0, ei[1], torch.ones(ei.shape[1])).clamp(min=1)
        w = 1.0 / deg[ei[1]]
        g['A'] = torch.sparse_coo_tensor(torch.stack([ei[1], ei[0]]), w, (n, n)).coalesce().to_sparse_csr()
    elif kind == 'GAT':
        loops = torch.arange(n); g['e'] = torch.cat([ei, torch.stack([loops, loops])], 1)
    return g

class GCNL(nn.Module):
    def __init__(s, i, o): super().__init__(); s.l = nn.Linear(i, o)
    def forward(s, x, g): return torch.sparse.mm(g['A'], s.l(x))
class SAGEL(nn.Module):
    def __init__(s, i, o): super().__init__(); s.a = nn.Linear(i, o); s.b = nn.Linear(i, o, bias=False)
    def forward(s, x, g): return s.a(x) + s.b(torch.sparse.mm(g['A'], x))
class GATL(nn.Module):
    def __init__(s, i, o, h=4):
        super().__init__(); s.h, s.c = h, o // h
        s.W = nn.Linear(i, o, bias=False); s.as_ = nn.Parameter(torch.randn(h, s.c) * 0.1); s.ad = nn.Parameter(torch.randn(h, s.c) * 0.1)
    def forward(s, x, g):
        n = x.shape[0]; z = s.W(x).view(n, s.h, s.c); e = g['e']
        a = F.leaky_relu((z * s.as_).sum(-1)[e[0]] + (z * s.ad).sum(-1)[e[1]], 0.2)
        mx = torch.full((n, s.h), -1e30).scatter_reduce(0, e[1].unsqueeze(1).expand(-1, s.h), a, 'amax', include_self=True)
        a = torch.exp(a - mx[e[1]])
        den = torch.zeros(n, s.h).index_add_(0, e[1], a)
        a = a / den[e[1]].clamp(min=1e-16)
        a = F.dropout(a, DROP, s.training)
        out = torch.zeros(n, s.h, s.c).index_add_(0, e[1], a.unsqueeze(-1) * z[e[0]])
        return out.reshape(n, s.h * s.c)

class Net(nn.Module):
    def __init__(s, kind, d):
        super().__init__(); s.kind = kind
        L = {'GCN': GCNL, 'SAGE': SAGEL, 'GAT': GATL}.get(kind)
        if kind == 'MLP': s.layers = nn.ModuleList([nn.Linear(d, 256), nn.Linear(256, 128)]); s.out = nn.Linear(128, 1)
        else: s.layers = nn.ModuleList([L(d, HID), L(HID, HID)]); s.out = nn.Linear(HID, 1)
    def forward(s, x, g=None):
        for l in s.layers:
            x = F.dropout(x, DROP if s.kind != 'MLP' else 0.2, s.training)
            x = l(x) if s.kind == 'MLP' else l(x, g)
            x = F.elu(x) if s.kind == 'GAT' else F.relu(x)
        return s.out(F.dropout(x, DROP, s.training)).squeeze(-1)

def graph_for(mask):
    g = subgraph(mask); return prep_graph(g, model_name) if model_name != 'MLP' else g
gtr = graph_for(t <= TRAIN_END); gva = graph_for((t > TRAIN_END) & (t <= VAL_END)); gte = graph_for(t > VAL_END)
def sel(g, idx): return torch.tensor(g['loc'][idx], dtype=torch.long)
tr_l, va_l, te_l = sel(gtr, tr), sel(gva, va), sel(gte, te)
Xtr, Xva, Xte = Xall[gtr['nodes']], Xall[gva['nodes']], Xall[gte['nodes']]
ytr = torch.tensor(y[tr], dtype=torch.float32)

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(model_name, Xall.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=5e-3 if model_name != 'MLP' else 1e-3, weight_decay=5e-4)
    best, bstate, bad = -1, None, 0; t0 = time.time()
    for ep in range(max_epochs):
        net.train(); opt.zero_grad()
        if model_name == 'MLP':
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 1024):
                b = perm[i:i + 1024]; opt.zero_grad()
                F.binary_cross_entropy_with_logits(net(Xtr[tr_l[b]]), ytr[b]).backward(); opt.step()
        else:
            F.binary_cross_entropy_with_logits(net(Xtr, gtr)[tr_l], ytr).backward(); opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad(): pv = torch.sigmoid(net(Xva, gva)[va_l] if model_name != 'MLP' else net(Xva[va_l])).numpy()
            s = ap(y[va], pv)
            if s > best: best, bstate, bad = s, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= 8: break
    ttr = time.time() - t0
    net.load_state_dict(bstate); net.eval()
    with torch.no_grad():
        pv = torch.sigmoid(net(Xva, gva)[va_l] if model_name != 'MLP' else net(Xva[va_l])).numpy()
        t0 = time.time(); pt = torch.sigmoid(net(Xte, gte)[te_l] if model_name != 'MLP' else net(Xte[te_l])).numpy(); tin = time.time() - t0
    return pv, pt, ttr, tin, sum(p.numel() for p in net.parameters()), ep + 1

PV, PT, TT, TI, ep_list = [], [], [], [], []
for s in SEEDS[:n_seeds]:
    pv, pt, ttr, tin, npar, ep = run(s)
    PV.append(pv); PT.append(pt); TT.append(ttr); TI.append(tin); ep_list.append(ep)
    print(model_name, 'seed', s, 'val AP %.4f' % ap(y[va], pv), 'test AP %.4f' % ap(y[te], pt), 'epochs', ep, 'train s %.1f' % ttr, flush=True)
if True:
    save_preds(f'{model_name}__TX', PV, PT, TT, TI, dict(model=model_name, features='TX', n_features=int(Xall.shape[1]), params=int(npar), epochs=ep_list,
               hidden=HID, dropout=DROP, note='inductive: trained on steps<=29 subgraph; validation on steps 30-34 subgraph; test on steps>=35 subgraph'))
