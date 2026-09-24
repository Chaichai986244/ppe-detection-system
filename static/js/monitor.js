/**
 * PPE Monitor — Single Page Control Room
 * Left: video + controls (full height). Right: charts + data.
 */
(function(){
'use strict';
const $=s=>document.querySelector(s);

// ── DOM: Control Panel ───────────────────────────────────────────
const videoViewport=$('#video-viewport'),video=$('#webcam-video'),displayCanvas=$('#display-canvas'),overlayCanvas=$('#overlay-canvas');
const displayCtx=displayCanvas.getContext('2d'),overlayCtx=overlayCanvas.getContext('2d');
const noSignal=$('#no-signal'),alarmPopup=$('#alarm-popup'),alarmPopupText=$('#alarm-popup-text');
const btnStart=$('#btn-start'),btnStop=$('#btn-stop'),sourceSelect=$('#source-select'),videoFileInput=$('#video-file-input');
const sceneSelect=$('#scene-select'),locationSelect=$('#location-select'),confidenceSlider=$('#confidence-slider'),confValue=$('#conf-value'),rulesHint=$('#scene-rules-hint');
const connectionDot=$('#connection-dot'),connectionText=$('#connection-text'),deviceBadge=$('#device-badge');
const valDetections=$('#val-detections'),valViolations=$('#val-violations'),valCompliance=$('#val-compliance'),valUptime=$('#val-uptime');
const fpsValue=$('#fps-value'),inferenceFps=$('#inference-fps');
const lightGreen=$('#light-green'),lightYellow=$('#light-yellow'),lightRed=$('#light-red'),alarmLevelText=$('#alarm-level-text');
const alarmTableBody=$('#alarm-table-body');

// ── State ────────────────────────────────────────────────────────
let webcamStream=null,detectionRunning=false,animId=null,sendId=null,lastAnnotated=null,alarmHistory=[],startTime=null,frameTS=[],uptimeTimer=null,popupTimer=null;

// ── Socket.IO ────────────────────────────────────────────────────
const socket=io({transports:['websocket','polling'],maxHttpBufferSize:10*1024*1024,reconnection:true,reconnectionDelay:1e3,reconnectionDelayMax:1e4});
socket.on('connect',()=>{connectionDot.className='status-dot live';connectionText.textContent='已连接';socket.emit('request_stats')});
socket.on('disconnect',()=>{connectionDot.className='status-dot dead';connectionText.textContent='断开'});
socket.on('server_info',i=>{const g=i.device==='cuda'||i.device==='0';deviceBadge.textContent=g?'GPU':'CPU';deviceBadge.className=g?'badge badge-info':'badge'});
socket.on('server_error',e=>console.error(e.message));

socket.on('stats_update',s=>{
  if(s.persons!==undefined)valDetections.textContent=s.persons;
  if(s.violations!==undefined)valViolations.textContent=s.violations;
  if(s.fps!==undefined)inferenceFps.textContent=s.fps;
  updateLights(s.alarm_level||0);
  updateEntryExit(s.entry_exit);
});

socket.on('detection_frame',d=>{
  lastAnnotated=d.image_b64||null;
  valDetections.textContent=d.persons||0;valViolations.textContent=d.violations||0;inferenceFps.textContent=d.fps||0;
  const p=d.persons||0,v=d.violations||0;
  valCompliance.textContent=p>0?(v===0?'100%':Math.max(0,Math.round((1-v/p)*100))+'%'):'--';
  updateLights(d.alarm_level||0);drawOverlay(d.image_b64);
  updateEntryExit(d.entry_exit);
});

socket.on('alarm_alert',a=>{showAlarm(a);addAlarmRow(a)});
socket.on('detection_status',s=>{
  if(s.running){detectionRunning=true;btnStart.classList.add('hidden');btnStop.classList.remove('hidden');noSignal.classList.add('hidden')}
  else{detectionRunning=false;btnStart.classList.remove('hidden');btnStop.classList.add('hidden')}
});

socket.on('scene_changed',d=>{updateLocations(d.presets,d.location);updateRules(d.rules)});
socket.on('classes_changed',d=>{activeClasses=d.active||[];renderChips()});socket.on('model_changed',d=>{if(d.classes){activeClasses=d.classes;renderChips();}fetch('/api/models').then(r=>r.json()).then(m=>{if(m.active)document.querySelector('#model-select')?.value!==m.active&&(document.querySelector('#model-select').value=m.active)})});
socket.on('confidence_changed',d=>{confValue.textContent=d.confidence.toFixed(2);confidenceSlider.value=Math.round(d.confidence*100)});

// ── Uptime ───────────────────────────────────────────────────────
function startUptime(){startTime=Date.now();uptimeTimer=setInterval(()=>{const e=Math.floor((Date.now()-startTime)/1e3);valUptime.textContent=String(Math.floor(e/60)).padStart(2,'0')+':'+String(e%60).padStart(2,'0')},1e3)}
function stopUptime(){clearInterval(uptimeTimer);uptimeTimer=null;valUptime.textContent='00:00'}

// ── Webcam ───────────────────────────────────────────────────────
async function startWebcam(){if(webcamStream)return;try{webcamStream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'environment'},audio:false});video.srcObject=webcamStream;await video.play();const w=video.videoWidth||640,h=video.videoHeight||480;displayCanvas.width=w;displayCanvas.height=h;overlayCanvas.width=w;overlayCanvas.height=h;renderLoop();sendId=setInterval(sendFrame,200);noSignal.classList.add('hidden')}catch(e){noSignal.innerHTML='<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg><p>摄像头权限被拒绝</p><small>请在浏览器中允许摄像头权限</small>'}}
function stopWebcam(){if(webcamStream){webcamStream.getTracks().forEach(t=>t.stop());webcamStream=null}video.srcObject=null;cancelAnimationFrame(animId);animId=null;clearInterval(sendId);sendId=null;displayCtx.clearRect(0,0,displayCanvas.width,displayCanvas.height);overlayCtx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);noSignal.classList.remove('hidden');noSignal.innerHTML='<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg><p>监控已停止</p><small>点击「开始检测」</small>'}
function renderLoop(){if(!webcamStream)return;const n=performance.now();frameTS.push(n);frameTS=frameTS.filter(t=>n-t<5e3);if(frameTS.length>=2){const e=(frameTS[frameTS.length-1]-frameTS[0])/1e3;fpsValue.textContent=(frameTS.length/e).toFixed(1)}displayCtx.save();displayCtx.translate(displayCanvas.width,0);displayCtx.scale(-1,1);displayCtx.drawImage(video,0,0,displayCanvas.width,displayCanvas.height);displayCtx.restore();animId=requestAnimationFrame(renderLoop)}
function sendFrame(){if(!webcamStream||!detectionRunning)return;const c=document.createElement('canvas');c.width=video.videoWidth||640;c.height=video.videoHeight||480;c.getContext('2d').drawImage(video,0,0);socket.emit('webcam_frame',{image_b64:c.toDataURL('image/jpeg',.5),width:c.width,height:c.height})}
function drawOverlay(b64){if(!b64)return;const i=new Image();i.onload=()=>{overlayCtx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);overlayCtx.save();overlayCtx.translate(overlayCanvas.width,0);overlayCtx.scale(-1,1);overlayCtx.drawImage(i,0,0,overlayCanvas.width,overlayCanvas.height);overlayCtx.restore()};i.src=b64}

// ── Alarms ───────────────────────────────────────────────────────
function showAlarm(a){const l=a.level>=2?'【二级告警】':'【一级预警】';const n=(a.display_name||'').replace('NO-Helmet','未戴安全帽').replace('NO-Safety Vest','未穿反光衣');alarmPopupText.textContent=l+n;alarmPopup.classList.remove('hidden');clearTimeout(popupTimer);popupTimer=setTimeout(()=>alarmPopup.classList.add('hidden'),4e3)}
function addAlarmRow(a){alarmHistory.unshift(a);if(alarmHistory.length>50)alarmHistory.pop();const rows=alarmHistory.slice(0,20).map(a=>{const t=a.timestamp?a.timestamp.split(' ')[1]||a.timestamp:'--';const n=(a.display_name||a.type||'').replace('NO-Helmet','未戴安全帽').replace('NO-Safety Vest','未穿反光衣');const b=a.level>=2?'<span class=\"badge badge-alarm\">二级</span>':'<span class=\"badge badge-warn\">一级</span>';return`<tr><td>${t}</td><td>${n}</td><td>${b}</td></tr>`}).join('');alarmTableBody.innerHTML=rows||'<tr><td colspan="3" class="empty-cell">暂无记录</td></tr>'}
function updateLights(l){lightGreen.className=l===0?'alarm-light on-green':'alarm-light';lightYellow.className=l===1?'alarm-light on-amber':'alarm-light';lightRed.className=l>=2?'alarm-light on-red':'alarm-light';if(l===0){alarmLevelText.textContent='正常';alarmLevelText.style.color='var(--green)'}else if(l===1){alarmLevelText.textContent='一级预警';alarmLevelText.style.color='var(--amber)'}else{alarmLevelText.textContent='二级告警';alarmLevelText.style.color='var(--red)'}}
function updateEntryExit(ee){if(!ee)return;$('#val-in').textContent=ee.in||0;$('#val-out').textContent=ee.out||0}

// ── Class Filter Chips ──────────────────────────────────────────
const chipNames={Helmet:'安全帽','Safety Vest':'反光衣'};
const chipColors={Helmet:'#4caf50','Safety Vest':'#ff9800'};
let activeClasses=['Helmet','Safety Vest'];

function renderChips(){const g=$('#class-toggle-group');g.innerHTML=['Helmet','Safety Vest'].map(c=>`<label class=\"chip ${activeClasses.includes(c)?'active':''}\" onclick=\"toggleChip('${c}')\"><span class=\"dot\" style=\"background:${activeClasses.includes(c)?chipColors[c]:'var(--border-lit)'}\"></span>${chipNames[c]||c}</label>`).join('')}
window.toggleChip=function(c){if(activeClasses.includes(c))activeClasses=activeClasses.filter(x=>x!==c);else activeClasses.push(c);socket.emit('set_classes',{classes:activeClasses});renderChips()}

// ── Scene Controls ───────────────────────────────────────────────
function updateLocations(presets,active){locationSelect.innerHTML=(presets||[]).map(p=>`<option value="${p}" ${p===active?'selected':''}>${p}</option>`).join('')}
function updateRules(rules){if(!rules)return;const n={Helmet:'安全帽','Safety Vest':'反光衣'};const m=(rules.mandatory||[]).map(p=>n[p]||p).join('、');const r=(rules.recommended||[]).map(p=>n[p]||p).join('、');rulesHint.innerHTML=`<span class="rule-tag mand">强制:${m||'无'}</span><span class="rule-tag rec">建议:${r||'无'}</span>`}
sceneSelect.addEventListener('change',()=>{socket.emit('switch_scene',{scene:sceneSelect.value})});
locationSelect.addEventListener('change',()=>socket.emit('set_location',{location:locationSelect.value}));
confidenceSlider.addEventListener('input',()=>confValue.textContent=(parseInt(confidenceSlider.value)/100).toFixed(2));
confidenceSlider.addEventListener('change',()=>socket.emit('set_confidence',{value:parseInt(confidenceSlider.value)/100}));window.switchModel=function(k){socket.emit('switch_model',{model:k})};

// ── DingTalk Settings ────────────────────────────────────────────
window.openSettings=function(){$('#settings-overlay').classList.remove('hidden')}
window.closeSettings=function(){$('#settings-overlay').classList.add('hidden')}
window.saveDt=function(){
  const e=$('#dingtalk-enabled')?.checked??true,i=parseInt($('#dingtalk-interval')?.value??300),v=$('#dingtalk-video-end')?.checked??true;
  fetch('/api/dingtalk/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:e,interval_sec:i,on_video_end:v})}).catch(err=>console.error(err));
  const b=$('#dingtalk-status');b.textContent=e?'推送开':'推送关';b.className=e?'badge badge-info':'badge'
}
window.testDingtalk=async function(){try{const r=await fetch('/api/dingtalk/test',{method:'POST'});const d=await r.json();alert(d.ok?'测试消息发送成功！':'发送失败：'+(d.error||''))}catch(e){alert('请求失败：'+e.message)}}

// ── Controls ─────────────────────────────────────────────────────
window.toggleDetection=function(){if(detectionRunning){socket.emit('stop_detection');stopWebcam();stopUptime();detectionRunning=false;btnStart.classList.remove('hidden');btnStop.classList.add('hidden');lastAnnotated=null;overlayCtx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);updateLights(0)}else{const s=sourceSelect.value;if(s==='webcam'){startWebcam().then(()=>{socket.emit('start_detection',{source:'webcam'});startUptime()})}else{videoFileInput.click()}}}
sourceSelect.addEventListener('change',()=>videoFileInput.classList.toggle('hidden',sourceSelect.value!=='video'));
videoFileInput.addEventListener('change',e=>{const f=e.target.files[0];if(!f)return;if(detectionRunning){socket.emit('stop_detection');detectionRunning=false}stopWebcam();stopUptime();lastAnnotated=null;overlayCtx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);updateLights(0);const fd=new FormData();fd.append('video',f);fetch('/api/upload_video',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{if(d.path){socket.emit('start_detection',{source:'video',video_path:d.path});startUptime();detectionRunning=true;btnStart.classList.add('hidden');btnStop.classList.remove('hidden');noSignal.classList.add('hidden')}}).catch(e=>console.error(e))});
window.captureScreenshot=function(){const m=document.createElement('canvas');m.width=overlayCanvas.width;m.height=overlayCanvas.height;const mc=m.getContext('2d');mc.drawImage(displayCanvas,0,0);if(lastAnnotated){const i=new Image();i.onload=()=>{mc.save();mc.translate(m.width,0);mc.scale(-1,1);mc.drawImage(i,0,0);mc.restore();dl(m)};i.src=lastAnnotated}else{dl(m)}function dl(c){const a=document.createElement('a');a.download='screenshot_'+new Date().toISOString().replace(/[:.]/g,'-')+'.png';a.href=c.toDataURL('image/png');a.click()}}

// ═══════════════════════════════════════════════════════════════════
// RIGHT PANEL: ECharts + KPI (replaces Plotly)
// ═══════════════════════════════════════════════════════════════════
const chartInstances={};
function getChart(id){if(!chartInstances[id]){const d=document.getElementById(id);chartInstances[id]=echarts.init(d,null,{renderer:'canvas'})}return chartInstances[id]}
window.addEventListener('resize',()=>Object.values(chartInstances).forEach(c=>c.resize()));

async function fetchJSON(url){const r=await fetch(url);if(!r.ok)throw Error('HTTP '+r.status);return r.json()}

async function refreshViz(){
  try{
    // KPI
    const ov=await fetchJSON('/api/stats/overview');
    $('#kpi-detections').textContent=(ov.total_detections||0).toLocaleString();
    $('#kpi-violations').textContent=(ov.total_violations||0).toLocaleString();
    $('#kpi-compliance').textContent=(ov.compliance_rate||100)+'%';
    $('#kpi-today').textContent=(ov.today_alarms||0).toLocaleString();

    // Bar chart — from pyecharts API
    const barOpt=await fetchJSON('/api/charts/bar');
    getChart('chart-bar').setOption({...barOpt,
      grid:{left:40,right:20,top:10,bottom:30},
      backgroundColor:'transparent',
    });

    // Line chart
    const lineOpt=await fetchJSON('/api/charts/line');
    getChart('chart-line').setOption({...lineOpt,
      grid:{left:40,right:20,top:30,bottom:30},
      backgroundColor:'transparent',
    });

    // Pie chart
    const pieOpt=await fetchJSON('/api/charts/pie');
    getChart('chart-pie').setOption({...pieOpt,
      backgroundColor:'transparent',
    });

    $('#viz-updated').textContent=new Date().toLocaleTimeString('zh-CN');
  }catch(e){console.error('Viz refresh:',e)}
}

// ── Init ─────────────────────────────────────────────────────────
fetch('/api/classes').then(r=>r.json()).then(d=>{activeClasses=d.active||[];renderChips()});
fetch('/api/scene/current').then(r=>r.json()).then(d=>{updateLocations(d.presets,d.location);updateRules({mandatory:d.rules?.mandatory||[],recommended:d.rules?.recommended||[]})});
fetch('/api/dingtalk/settings').then(r=>r.json()).then(s=>{const iv=$('#dingtalk-interval');if(iv)iv.value=s.interval_sec;$('#dingtalk-status').textContent=s.enabled?'推送开':'推送关';$('#dingtalk-status').className=s.enabled?'badge badge-info':'badge'});

refreshViz();
setInterval(refreshViz,30000);
console.log('CCTV Control Room ready');
})();
