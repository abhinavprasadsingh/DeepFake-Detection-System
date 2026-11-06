const pages=["home","image","video"];
function showPage(p){pages.forEach(x=>document.getElementById(x).classList.toggle("active",x===p));document.querySelectorAll(".nav").forEach(x=>x.classList.toggle("active",x.dataset.page===p));document.getElementById("crumb").textContent=p[0].toUpperCase()+p.slice(1)}
document.querySelectorAll("[data-page]").forEach(x=>x.onclick=()=>showPage(x.dataset.page));
document.querySelectorAll("[data-go]").forEach(x=>x.onclick=()=>showPage(x.dataset.go));
function stats(values){return '<div class="stats">'+values.map(x=>'<span>'+x+'</span>').join("")+'</div>'}
async function analyze(kind){
 const input=document.getElementById(kind+"File"),result=document.getElementById(kind+"Result"),btn=document.getElementById(kind+"Run"),file=input.files[0];
 if(!file){alert("Please select a file first.");return}
 if(file.size>40*1024*1024){alert("Maximum file size is 40 MB.");return}
 btn.disabled=true;btn.textContent="Processing on Render...";
 const body=new FormData();body.append("file",file);
 try{
  const res=await fetch("/api/analyze/"+kind,{method:"POST",body}),data=await res.json();
  if(!res.ok)throw new Error(data.error||"Analysis failed");
  if(kind==="image"){
   result.className="";result.innerHTML='<img class="preview" src="'+data.preview+'" alt="Processed image">'+stats(["Faces located: "+data.faces,"Width: "+data.width,"Height: "+data.height])+"<p>"+data.notice+"</p>";
  }else{
   result.className="";result.innerHTML=stats(["Frames: "+data.sampled_frames,"Frames with faces: "+data.frames_with_faces,"Face locations: "+data.face_detections])+'<div class="frames">'+data.samples.map(x=>'<div><img src="'+x.preview+'" alt="Video frame"><small>Frame '+x.frame+' · '+x.faces+' face(s)</small></div>').join("")+'</div><p>'+data.notice+'</p>';
  }
 }catch(e){result.textContent=e.message;result.className="empty"}
 finally{btn.disabled=false;btn.textContent="Run inspection ↗"}
}
document.getElementById("imageRun").onclick=()=>analyze("image");
document.getElementById("videoRun").onclick=()=>analyze("video");
