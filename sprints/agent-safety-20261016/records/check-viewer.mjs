import {spawn} from 'node:child_process';
import {mkdtemp,rm,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const root=resolve(import.meta.dirname,'..');
const profile=await mkdtemp(join(tmpdir(),'safety-evidence-'));
const chrome=spawn('/home/dennis/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome',[
 '--headless','--no-sandbox','--disable-gpu','--no-first-run','--disable-background-networking',
 '--remote-debugging-port=19386',`--user-data-dir=${profile}`,'about:blank'],{stdio:'ignore'});
let socket;
try{
 let tabs;
 for(let i=0;i<80;i++){try{tabs=await(await fetch('http://127.0.0.1:19386/json/list')).json();break;}catch{await new Promise(r=>setTimeout(r,100));}}
 if(!tabs)throw Error('Browser startup failed');
 socket=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
 await new Promise((ok,no)=>{socket.onopen=ok;socket.onerror=no});
 let sequence=0;const pending=new Map();const errors=[];
 socket.onmessage=e=>{const r=JSON.parse(e.data);if(r.id){const p=pending.get(r.id);pending.delete(r.id);r.error?p.reject(r.error):p.resolve(r.result);}else if(r.method==='Runtime.exceptionThrown')errors.push(r.params);};
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await send('Runtime.enable');await send('Page.enable');
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:pathToFileURL(join(root,'index.html')).href});
 for(let i=0;i<80;i++){if(await evaluate('document.readyState === "complete" && document.querySelectorAll(".decision").length===3'))break;await new Promise(r=>setTimeout(r,100));}
 const results=await evaluate(`(()=>{
 const w=document.querySelector('#world'),c=document.querySelector('#condition');let pairs=0;
 for(const wo of w.options)for(const co of c.options){w.value=wo.value;c.value=co.value;w.dispatchEvent(new Event('change'));
 const cards=[...document.querySelectorAll('.decision')];if(cards.length!==3||cards.some(x=>!x.querySelector('pre').textContent.includes(wo.value)))throw Error('Invalid paired rendering');pairs++;}
 const tabs=[];for(const b of document.querySelectorAll('nav button')){b.click();const shown=[...document.querySelectorAll('section.panel')].filter(p=>!p.hidden);if(shown.length!==1||shown[0].id!==b.dataset.tab)throw Error('Tab failure');tabs.push(b.dataset.tab);}
 document.querySelector('[data-tab="exp6"]').click();w.selectedIndex=0;c.selectedIndex=0;w.dispatchEvent(new Event('change'));
 const images=[...document.images].map(i=>({src:i.getAttribute('src'),loaded:i.complete&&i.naturalWidth>0}));
 return {pairs,tabs,images,worlds:w.options.length,conditions:c.options.length,desktopOverflow:document.documentElement.scrollWidth>innerWidth};})()`);
 const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 await writeFile(join(root,'records/evidence-viewer-desktop.png'),Buffer.from(shot.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 results.mobileOverflow=await evaluate('document.documentElement.scrollWidth > innerWidth');results.runtimeErrors=errors;
 const mobile=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 await writeFile(join(root,'records/evidence-viewer-mobile.png'),Buffer.from(mobile.data,'base64'));
 results.passed=results.pairs===384&&results.images.every(i=>i.loaded)&&!results.desktopOverflow&&!results.mobileOverflow&&!errors.length;
 await writeFile(join(root,'records/viewer-browser-validation.json'),JSON.stringify(results,null,2)+'\n');
 console.log(JSON.stringify(results));if(!results.passed)process.exitCode=1;
}finally{socket?.close();chrome.kill('SIGTERM');await new Promise(r=>setTimeout(r,500));await rm(profile,{recursive:true,force:true});}
