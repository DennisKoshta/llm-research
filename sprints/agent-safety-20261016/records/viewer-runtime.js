
'use strict';
const data=JSON.parse(document.getElementById('evidence').textContent);
const world=document.getElementById('world'),condition=document.getElementById('condition');
for(const [i,w] of data.worlds.entries()){const o=document.createElement('option');o.value=w.world_id;o.textContent=`${String(i+1).padStart(2,'0')} · ${w.own_job} → ${w.peer_job} · ${w.unit}`;world.append(o);}
for(const auth of [0,1])for(const cost of [0,1])for(const benefit of ['high','low']){const o=document.createElement('option');o.value=`auth${auth}_cost${cost}_${benefit}`;o.textContent=`${auth?'Permitted':'Forbidden'} · ${cost?'costly':'spare capacity'} · ${benefit} benefit`;condition.append(o);}
const titles={baseline:'Narrative',payoff:'Payoff table',payoff_permission:'Table + permission'};
function element(tag,text,cls){const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e;}
function draw(){const selected=data.rows.filter(r=>r.world===world.value&&r.condition===condition.value);const parent=document.getElementById('pairs');parent.replaceChildren();
document.getElementById('context').textContent=`World ${world.value} · TRANSFER = ${selected[0].transfer_label} · table order begins with ${selected[0].table_first}. Group points are completed-task totals; forbidden options remain forbidden regardless of points.`;
for(const arm of ['baseline','payoff','payoff_permission']){const r=selected.find(x=>x.arm===arm),card=element('article','',`decision ${r.violation?'bad':'good'}`);card.append(element('h3',titles[arm]),element('div',r.transfer?'TRANSFER':'KEEP','action'),element('div',r.violation?'Permission violation':'Permission respected','badge'),element('p',`Completed points: ${r.points} (${r.own_points} own + ${r.peer_points} peer).`),element('p',r.correct?'Matches the constrained objective.':'Does not match the constrained objective.','muted'));
const detail=document.createElement('details');detail.append(element('summary','Exact prompt and output'));const text=r.messages.map(m=>`[${m.role}]\n${m.content}`).join('\n\n')+`\n\n[raw output]\n${r.raw}\n\n[prompt SHA256]\n${r.hash}`;detail.append(element('pre',text));card.append(detail);parent.append(card);}}
world.addEventListener('change',draw);condition.addEventListener('change',draw);draw();
for(const button of document.querySelectorAll('nav button'))button.addEventListener('click',()=>{for(const b of document.querySelectorAll('nav button'))b.setAttribute('aria-pressed',String(b===button));for(const p of document.querySelectorAll('section.panel'))p.hidden=p.id!==button.dataset.tab;});
