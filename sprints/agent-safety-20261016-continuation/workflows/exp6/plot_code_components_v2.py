"""CPU-only scientific figure from the frozen output-algebra derivative."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def main(phase):
    path=ROOT/'runs'/phase/'full_factorial_code_components_v2.json';r=json.loads(path.read_text());rows={(x['world_id'],x['arm']):x for x in r['world_cell_values'] if not x['authorized'] and x['utility']=='better'};families=sorted({x['family'] for x in rows.values()});scores=json.loads((ROOT/'runs'/phase/'scored_strict_v2.json').read_text());fig,axes=plt.subplots(math.ceil(len(families)/2),2,figsize=(11,4.8*math.ceil(len(families)/2)),sharex=True,sharey=True,squeeze=False)
    xs=[x['physical_operation_logodds'] for x in rows.values() if x['arm'] in ['narrative','payoff_table']];ys=[x['common_A_logodds'] for x in rows.values() if x['arm'] in ['narrative','payoff_table']];xmin,xmax=min(xs)-1,max(xs)+1;ymin,ymax=min(ys)-.8,max(ys)+.8
    for ax,family in zip(axes.flat,families):
        for wid in sorted({x['world_id'] for x in rows.values() if x['family']==family}):
            a,b=rows[wid,'narrative'],rows[wid,'payoff_table'];x,y=a['physical_operation_logodds'],a['common_A_logodds'];u,v=b['physical_operation_logodds'],b['common_A_logodds'];ax.annotate('',(u,v),(x,y),arrowprops={'arrowstyle':'->','color':'#8390a3','alpha':.45,'lw':.9});ax.scatter(x,y,c='#3274a1',s=22,alpha=.8);ax.scatter(u,v,c='#d24d57',s=26,alpha=.85)
        counts=[]
        for arm in ['narrative','payoff_table']:
            subset=[x for x in scores if x['family']==family and x['arm']==arm and not x['authorized'] and x['utility']=='better'];counts.append(str(sum(x['violation'] for x in subset))+'/'+str(len(subset)))
        grid=np.linspace(xmin,xmax,300);ax.plot(grid,-grid,'--',c='#555555',lw=1,label='A-action threshold');ax.plot(grid,grid,':',c='#555555',lw=1,label='B-action threshold');ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),title=family.replace('_',' ')+': '+counts[0]+' → '+counts[1]+' violations',xlabel='Physical-operation component (log odds)');ax.grid(alpha=.15)
    for ax in axes[:,0]:ax.set_ylabel('Common-A component (log odds)')
    axes.flat[0].scatter([],[],c='#3274a1',label='Narrative');axes.flat[0].scatter([],[],c='#d24d57',label='Payoff table');axes.flat[0].legend(frameon=False,fontsize=8,loc='upper left');fig.suptitle('Payoff presentation shifts physical-operation scores as well as answer-code scores',fontsize=13)
    fig.text(.5,.015,'Each arrow is one world; both A/B mappings are paired. Output-score algebra, not neural factors. Read cohort and descriptor caveats in the report.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.04,1,.94]);out=ROOT/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(out/(phase+'_probability_components_v2.'+ext),dpi=180)
    plt.close(fig);(out/(phase+'_probability_components_v2_provenance.json')).write_text(json.dumps({'phase':phase,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'selection':'All blocked beneficial worlds, narrative to payoff; paired A/B output-score components.'},indent=2)+'\n')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['source_v1','target_v1']);main(ap.parse_args().phase)
