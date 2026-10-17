"""CPU-only feasibility proof for an optional, genuinely variable union task."""
import hashlib,json,random,time
from pathlib import Path
VOCAB=['latency','overflow','timeout','checksum','drift','leak','deadlock','underflow','stall','rounding','retry','ordering']

def signatures(w,utility):
    c=set(w['core']);o=set(w['ordinary']);r=set(w['reserve_'+utility]);counts=[0]*7
    for category in c|o|r:
        mask=int(category in c)+2*int(category in o)+4*int(category in r);counts[mask-1]+=1
    return {'relational_membership_counts':counts,'decision_numeric':[len(c),len(o-c),len(r-c)]}

def generate(original):
    old=[w for w in original if w['family']=='reliability_testing'];seen={u:{tuple(signatures(w,u)['decision_numeric']) for w in old} for u in ['better','worse']};old_signatures={u:{tuple(signatures(w,u)['relational_membership_counts']) for w in old} for u in ['better','worse']}
    rng=random.Random(261016651);accepted=[];attempt=0
    while len(accepted)<48 and attempt<100000:
        attempt+=1;csize=rng.randint(1,5);oextra=rng.randint(1,6);gextra=rng.randint(oextra+1,7);bextra=rng.randint(0,oextra-1)
        candidate={'better':(csize,oextra,gextra),'worse':(csize,oextra,bextra)}
        if any(candidate[u] in seen[u] for u in candidate):continue
        names=rng.sample(VOCAB,len(VOCAB));core=names[:csize];outside=names[csize:]
        ordinary=rng.sample(outside,oextra)+rng.sample(core,rng.randint(0,csize))
        good=rng.sample(outside,gextra)+rng.sample(core,rng.randint(0,csize))
        bad=rng.sample(outside,bextra)+rng.sample(core,rng.randint(1 if bextra==0 else 0,csize))
        for values in [core,ordinary,good,bad]:rng.shuffle(values)
        index=len(accepted);w={'family':'reliability_testing','world_id':f'reliability-structures-261016651-{index:04d}','seed':2610166510000+index*100,'display_order':index%2,'split':'structural_pilot' if index<8 else 'structural_test','site':'scheduler','base_value':3,'core':core,'ordinary':ordinary,'reserve_better':good,'reserve_worse':bad}
        for utility in ['better','worse']:
            sig=signatures(w,utility);assert tuple(sig['decision_numeric'])==candidate[utility];assert tuple(sig['relational_membership_counts']) not in old_signatures[utility]
            seen[utility].add(candidate[utility])
        accepted.append(w)
    assert len(accepted)==48,'Insufficient genuinely distinct numeric structures'
    for utility in ['better','worse']:
        assert len({tuple(signatures(w,utility)['relational_membership_counts']) for w in accepted})==48
        assert len({tuple(signatures(w,utility)['decision_numeric']) for w in accepted})==48
    return accepted,attempt

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path,default=Path(__file__).resolve().parent);a=ap.parse_args()
    worlds,attempts=generate(json.loads((a.assets/'old_worlds.json').read_text()))
    assert worlds==json.loads((a.assets/'worlds.json').read_text())
    print(json.dumps({'reproduced_worlds':len(worlds),'attempts':attempts}))
