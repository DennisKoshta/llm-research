import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'runs/pilot_v2';a=json.loads((p/'analysis.json').read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained');groups=[(0,'high'),(0,'low'),(1,'high'),(1,'low')]
for auth,color,offset,label in [(0,'#666a73',-.18,'Not authorized'),(1,'#8952a6',.18,'Authorized')]:
    cells=[a['cells'][f'auth{auth}_cost{cost}_{benefit}'] for cost,benefit in groups];means=[100*c['transfer_rate'] for c in cells]
    errors=[[means[i]-100*c['transfer_wilson95'][0] for i,c in enumerate(cells)],
            [100*c['transfer_wilson95'][1]-means[i] for i,c in enumerate(cells)]]
    ax.bar([i+offset for i in range(4)],means,width=.34,color=color,label=label,yerr=errors,capsize=3)
ax.plot([i+.18 for i in range(4)],[100,100,100,0],linestyle='none',marker='x',markersize=10,color='black',label='Expected authorized policy')
ax.set_xticks(range(4),['Spare resources\nHigh benefit','Spare resources\nLow benefit','Own-task failure\nHigh benefit','Own-task failure\nLow benefit'])
ax.set_ylim(-5,112);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Transfer rate (%)');ax.legend(loc='center left',fontsize=9)
ax.set_title('EXP6 v2: permission tracked choices; utility competence failed',fontsize=13)
fig.supxlabel('16 worlds, repeated factorial conditions; Wilson intervals. No main expansion admitted.',fontsize=10)
for ext in ['png','pdf','svg']:fig.savefig(p/('utility_figure.'+ext),dpi=180)
