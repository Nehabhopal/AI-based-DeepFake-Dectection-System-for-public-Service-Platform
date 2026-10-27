const $ = (s) => document.querySelector(s);
let currentUser = null;
let signupMode = false;
let latestResult = null;
const titles = {dashboard:"Dashboard",auth:"Account",check:"Check an Image",history:"My Results",awareness:"Public Awareness",feedback:"Report / Feedback",admin:"Admin Dashboard"};

function toast(message){const el=$("#toast");el.textContent=message;el.classList.add("show");setTimeout(()=>el.classList.remove("show"),3500)}
async function api(url, options={}) {
  const opts={credentials:"same-origin",...options,headers:{...(options.headers||{})}};
  if(opts.body && !(opts.body instanceof FormData) && typeof opts.body!=="string"){opts.headers["Content-Type"]="application/json";opts.body=JSON.stringify(opts.body)}
  const response=await fetch(url,opts);
  let data={};try{data=await response.json()}catch(_){}
  if(!response.ok || data.success===false) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function showView(view){
  if(view==="history"&&!currentUser){view="auth";signupMode=false}
  if(view==="check"&&!currentUser){view="auth";signupMode=false}
  if(view==="feedback"&&!currentUser){view="auth";signupMode=false}
  document.querySelectorAll(".view").forEach(el=>el.classList.toggle("hidden",el.id!==view));
  document.querySelectorAll(".nav").forEach(el=>el.classList.toggle("active",el.dataset.view===view));
  $("#pageTitle").textContent=titles[view]||"DeepSafe";
  if(view==="history")loadHistory();
  if(view==="admin")loadAdmin();
  if(view==="auth")renderAuthMode();
}
document.querySelectorAll("[data-view]").forEach(b=>b.addEventListener("click",()=>showView(b.dataset.view)));
document.querySelectorAll("[data-go]").forEach(b=>b.addEventListener("click",()=>showView(b.dataset.go)));
function renderAuthMode(){
  $("#authTitle").textContent=signupMode?"Create your account":"Welcome back";
  $("#authSubtitle").textContent=signupMode?"Register to save and review your results.":"Log in to your DeepSafe account.";
  $("#nameLabel").classList.toggle("hidden",!signupMode);
  $("#name").required=signupMode;
  $("#authSubmit").textContent=signupMode?"Create account":"Log in";
  $("#toggleAuth").textContent=signupMode?"Already have an account? Log in":"Create an account";
}
$("#toggleAuth").addEventListener("click",()=>{signupMode=!signupMode;renderAuthMode();$("#authError").textContent=""});
$("#authButton").addEventListener("click",async()=>{
  if(currentUser){try{await api("/api/auth/logout",{method:"POST"});currentUser=null;refreshAuth();toast("Logged out successfully.");showView("dashboard")}catch(e){toast(e.message)}}
  else showView("auth");
});
$("#authForm").addEventListener("submit",async e=>{
  e.preventDefault();$("#authError").textContent="";
  const body=signupMode?{name:$("#name").value,email:$("#email").value,password:$("#password").value}:{email:$("#email").value,password:$("#password").value};
  try{const data=await api(signupMode?"/api/auth/signup":"/api/auth/login",{method:"POST",body});currentUser=data.user;refreshAuth();toast(`Welcome${currentUser.name?" "+currentUser.name:""}!`);showView("dashboard")}
  catch(err){$("#authError").textContent=err.message}
});
function refreshAuth(){ $("#authButton").textContent=currentUser?"Log out":"Log in";$("#adminNav").classList.toggle("hidden",!currentUser||currentUser.role!=="admin"); }
async function init(){
  try{const h=await api("/api/health");$("#connection").textContent=`API ${h.database==="connected"?"online":"database offline"} · model ${h.model}`;$("#connection").title=h.model_message||""}
  catch(e){$("#connection").textContent="API unavailable"}
  try{const data=await api("/api/auth/me");currentUser=data.user;refreshAuth()}catch(_){}
}
const fileInput=$("#imageFile");
fileInput.addEventListener("change",()=>{
  const f=fileInput.files[0];if(!f)return;
  $("#fileLabel").textContent=f.name;
  const url=URL.createObjectURL(f);$("#preview").src=url;$("#previewWrap").classList.remove("hidden");
});
$("#clearImage").addEventListener("click",()=>{fileInput.value="";$("#preview").removeAttribute("src");$("#previewWrap").classList.add("hidden");$("#fileLabel").textContent="Choose an image to analyze"});
$("#uploadForm").addEventListener("submit",async e=>{
  e.preventDefault();if(!currentUser){showView("auth");return}
  const file=fileInput.files[0];if(!file){$("#uploadMessage").textContent="Choose an image first.";return}
  const form=new FormData();form.append("image",file);
  $("#uploadMessage").textContent="Uploading and analyzing with the trained model…";
  const submit=e.submitter;submit.disabled=true;submit.textContent="Analyzing…";
  $("#analysisResult").classList.add("hidden");
  try{const data=await api("/api/analyze/image",{method:"POST",body:form});latestResult=data;renderResult(data);$("#analysisResult").classList.remove("hidden");$("#uploadMessage").textContent="Analysis completed.";loadHistory()}
  catch(err){$("#uploadMessage").textContent=err.message}
  finally{submit.disabled=false;submit.textContent="Analyze image"}
});
function renderResult(r){
 const label=r.prediction==="REAL"?"Likely Real":"Likely DeepFake";
 const badge=r.prediction==="REAL"?"real":"deepfake";
 const original=$("#preview").src;
 $("#analysisResult").innerHTML=`<div class="section-heading"><div><p class="eyebrow">ANALYSIS RESULT</p><h2>${label} <span class="badge ${badge}">${r.prediction}</span></h2><p>Model version ${escapeHtml(r.model_version)} · ${new Date(r.created_at).toLocaleString()}</p></div></div>
 <div class="result-grid"><div class="metric"><small>Model confidence</small><strong>${Number(r.confidence).toFixed(2)}%</strong></div><div class="metric"><small>REAL probability</small><strong>${Number(r.probabilities.REAL).toFixed(2)}%</strong></div><div class="metric"><small>DEEPFAKE probability</small><strong>${Number(r.probabilities.DEEPFAKE).toFixed(2)}%</strong></div><div class="metric"><small>Image quality</small><strong>${escapeHtml(r.image_quality)}</strong></div><div class="metric"><small>Face detected</small><strong>${r.face_detected?"Yes":"No"}</strong></div><div class="metric"><small>Face count</small><strong>${Number(r.face_count)}</strong></div></div>
 <p>The model output indicates patterns associated more strongly with the <b>${r.prediction}</b> class. This is not verified forensic evidence.</p>
 <div class="heatmaps"><figure><img src="${escapeAttr(r.image_url)}" alt="Uploaded original image"><figcaption>Original image</figcaption></figure><figure><img src="${escapeAttr(r.gradcam_url)}" alt="Grad-CAM heatmap generated by the model"><figcaption>Grad-CAM heatmap</figcaption></figure><figure><img src="${escapeAttr(r.overlay_url)}" alt="Grad-CAM heatmap overlaid on the image"><figcaption>Grad-CAM overlay</figcaption></figure></div>
 <div class="notice">${escapeHtml(r.disclaimer)} Heatmaps show model activation, not proof of manipulation.</div><div class="result-actions"><button class="button secondary" id="copyResult">Copy result summary</button><button class="button primary" data-go="history">View history</button><button class="button secondary" data-go="check">Analyze another</button></div>`;
 $("#copyResult").addEventListener("click",async()=>{const summary=`DeepSafe result: ${r.prediction}; confidence ${Number(r.confidence).toFixed(2)}%; REAL ${Number(r.probabilities.REAL).toFixed(2)}%; DEEPFAKE ${Number(r.probabilities.DEEPFAKE).toFixed(2)}%. AI detection is probabilistic, not absolute proof.`;try{await navigator.clipboard.writeText(summary);toast("Result summary copied.")}catch(_){toast(summary)}});
 $("#analysisResult").querySelectorAll("[data-go]").forEach(b=>b.addEventListener("click",()=>showView(b.dataset.go)));
}
function escapeHtml(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function escapeAttr(s){return escapeHtml(s)}
async function loadHistory(){
 const el=$("#historyContent");
 try{const data=await api("/api/history");if(!data.results.length){el.innerHTML="<p>No saved analyses yet. Your successful model analyses will appear here.</p>";return}
 el.innerHTML=`<table><thead><tr><th>Result</th><th>Confidence</th><th>Quality</th><th>Date</th><th>Action</th></tr></thead><tbody>${data.results.map(r=>`<tr><td><span class="badge ${r.prediction==="REAL"?"real":"deepfake"}">${r.prediction}</span></td><td>${r.confidence.toFixed(2)}%</td><td>${escapeHtml(r.image_quality)}</td><td>${new Date(r.created_at).toLocaleString()}</td><td><button class="text-button" data-delete="${r.id}">Delete</button></td></tr>`).join("")}</tbody></table>`;
 el.querySelectorAll("[data-delete]").forEach(b=>b.addEventListener("click",async()=>{if(!confirm("Delete this analysis record?"))return;try{await api(`/api/history/${b.dataset.delete}`,{method:"DELETE"});loadHistory();toast("Analysis deleted.")}catch(e){toast(e.message)}}));
 }catch(e){el.textContent=e.message}
}
$("#feedbackForm").addEventListener("submit",async e=>{e.preventDefault();try{const data=await api("/api/feedback",{method:"POST",body:{category:$("#feedbackCategory").value,message:$("#feedbackMessage").value}});$("#feedbackMessageStatus").textContent=data.message;$("#feedbackMessage").value=""}catch(err){$("#feedbackMessageStatus").textContent=err.message}});
async function loadAdmin(){const el=$("#adminContent");try{const d=await api("/api/admin/summary");el.innerHTML=`<div class="result-grid">${Object.entries(d.totals).map(([k,v])=>`<div class="metric"><small>${escapeHtml(k.replace("_"," "))}</small><strong>${v}</strong></div>`).join("")}</div><h3>Model metadata</h3><pre>${escapeHtml(JSON.stringify(d.model,null,2))}</pre><h3>Analysis count by date</h3><table><thead><tr><th>Date</th><th>Count</th></tr></thead><tbody>${d.analyses_over_time.map(x=>`<tr><td>${escapeHtml(x.date)}</td><td>${x.count}</td></tr>`).join("")||"<tr><td colspan='2'>No analysis records yet.</td></tr>"}</tbody></table>`}catch(e){el.textContent=e.message}}
init();
