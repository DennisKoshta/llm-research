import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'runs/diagnostic_v2';a=json.loads((p/'audit.json').read_text());stats=a['bundle_intervals']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
for ax,keys,labels,title in [
    (axes[0],['channel_absent','channel_A','channel_B'],['No default','Default A','Default B'],'Effect of flipping the delivered plan'),
    (axes[1],['default_live','default_flipped','default_delayed'],['Live plan','Flipped plan','Delayed plan'],'Effect of changing the advisory default')]:
    means=[stats[k]['mean']*100 for k in keys]
    errors=[[means[i]-stats[k]['bundle_bootstrap95'][0]*100 for i,k in enumerate(keys)],
            [stats[k]['bundle_bootstrap95'][1]*100-means[i] for i,k in enumerate(keys)]]
    ax.bar(range(3),means,color='#287c8e',width=.65,yerr=errors,capsize=4)
    ax.set_xticks(range(3),labels);ax.set_ylim(0,110);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Paired choices that changed (%)');ax.set_title(title,fontsize=12)
fig.suptitle('EXP4: advisory defaults interfere with message use',fontsize=15)
fig.supxlabel('12 descriptor bundles; descriptive bundle-bootstrap intervals. Original pilot remains failed.',fontsize=10)
for ext in ['png','pdf','svg']:fig.savefig(p/('diagnostic_figure.'+ext),dpi=180)
