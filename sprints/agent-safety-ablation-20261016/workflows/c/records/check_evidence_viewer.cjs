// Reuse the project's existing local Playwright/Chromium installation.
const {chromium}=require('/home/dennis/projects/Cognanimus/node_modules/@playwright/test');
const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const h=path.resolve(__dirname,'..');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/home/dennis/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell'});
 const page=await browser.newPage({viewport:{width:1440,height:1050},acceptDownloads:true});const errors=[],network=[];page.on('pageerror',e=>errors.push(String(e)));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
 await page.goto('file://'+h+'/evidence.html');await page.waitForSelector('#grid tbody tr');
 assert.equal(await page.locator('#world option').count(),24);assert.equal(await page.locator('#context option').count(),3);assert.equal(await page.locator('#menu option').count(),2);assert.equal(await page.locator('#branch option').count(),16);
 const coverage=await page.evaluate(()=>{
  const p=JSON.parse(document.getElementById('data').textContent);let cases=0;const branches=p.branches;
  if(document.getElementById('world').value!==p.worlds[0].world_id||document.getElementById('context').value!=='blocked_harmful'||document.getElementById('menu').value!=='0'||document.getElementById('branch').value!=='team_intact')throw Error('Noncanonical default');
  for(const w of p.worlds)for(const c of ['blocked_harmful','unstated_harmful','authorized_harmless'])for(const m of ['0','1']){
   el('world').value=w.world_id;el('context').value=c;el('menu').value=m;render();const rows=[...document.querySelectorAll('#grid tbody tr')];if(rows.length!==16)throw Error('Missing branch rows');
   rows.forEach((row,i)=>{const r=p.records[p.index[[w.world_id,c,m,branches[i]].join('|')]];if(row.dataset.branch!==branches[i]||!row.textContent.includes(r.score.action_code+' → '+r.score.semantic_action))throw Error('Wrong rendered decision');cases++});
  }
  el('world').selectedIndex=0;el('context').value='blocked_harmful';el('menu').value='0';render();
  for(const b of branches){el('branch').value=b;detail();const r=selected();const cards=[...document.querySelectorAll('#path .card')];const ids=[...r.assignment.upstream_ids,r.id];if(cards.length!==ids.length)throw Error('Wrong path length');cards.forEach((card,i)=>{if(!card.textContent.includes(ids[i])||card.querySelector('.advice').textContent!==p.records[ids[i]].raw)throw Error('Wrong recorded advice')})}
  el('branch').value='team_intact';detail();return {worldContextMenuViews:144,renderedDecisions:cases,branchPathsChecked:branches.length,storedRecords:Object.keys(p.records).length,activationVectorsAbsent:!Object.values(p.records).some(r=>'activations'in r||'early_activations'in r)};
 });
 const before=await page.locator('#world').inputValue();await page.locator('#next').click();assert.notEqual(await page.locator('#world').inputValue(),before);await page.locator('#prev').click();assert.equal(await page.locator('#world').inputValue(),before);
 await page.locator('#grid tbody tr[data-branch="team_reviewer_a"]').click();assert.equal(await page.locator('#branch').inputValue(),'team_reviewer_a');assert((await page.locator('#path').textContent()).includes('Refusal ablation'));
 const downloadPromise=page.waitForEvent('download');await page.locator('#download').click();const download=await downloadPromise;const saved=JSON.parse(fs.readFileSync(await download.path(),'utf8'));assert.equal(saved.path.length,3);assert(saved.path.every(r=>r.messages&&r.raw_line_sha256&&!r.activations));
 await page.locator('#branch').selectOption('team_intact');await page.screenshot({path:h+'/figures/evidence-viewer-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:h+'/figures/evidence-viewer-mobile.png',fullPage:true});await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:h+'/figures/evidence-viewer-mobile-top.png',fullPage:false});const mobileWidth=await page.evaluate(()=>({content:document.documentElement.scrollWidth,viewport:innerWidth}));assert(mobileWidth.content<=mobileWidth.viewport+1,JSON.stringify(mobileWidth));assert.deepEqual(errors,[]);assert.deepEqual(network,[]);await browser.close();
 const result={passed:true,created:new Date().toISOString(),html_sha256:crypto.createHash('sha256').update(fs.readFileSync(h+'/evidence.html')).digest('hex'),...coverage,previousNextAndTableSelection:true,evidenceDownloadVerified:true,mobileWidth,pageErrors:errors,externalNetworkRequests:network,scope:'Browser behavior and visual artifacts only. Scientific results remain bound to original frozen data and independent reviews.'};fs.writeFileSync(h+'/records/evidence-viewer-browser-review.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
})().catch(e=>{console.error(e);process.exit(1)});
