"""FastAPI HTTP transport over ReuseDetectorService (live demo).

Thin wrapper: POST /enroll and POST /check accept an image upload and delegate
to the service -- no detection logic lives here. The interactive console at
/docs lets you upload an image and run enroll/check from the browser.

The service is held on app.state (constructed via create_app from an injected
config), not as a module global.
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .config import DetectorConfig
from .service import ReuseDetectorService


class MatchOut(BaseModel):
    record_id: int
    user_id: str
    attempt_id: str
    inliers: int
    residual: float
    hash_distance: int


class AlertOut(BaseModel):
    triggered: bool
    distinct_users: int
    severity: float


class EnrollResponse(BaseModel):
    record_id: int
    user_id: str
    attempt_id: str
    corpus_size: int


class CheckResponse(BaseModel):
    matches: list[MatchOut]
    alert: AlertOut


class SubmitResponse(BaseModel):
    record_id: int
    user_id: str
    attempt_id: str
    corpus_size: int
    matches: list[MatchOut]
    alert: AlertOut


def _decode(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    return arr


_DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>image-movement — reuse detection demo</title>
<style>
 body{font-family:system-ui,-apple-system,sans-serif;max-width:780px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
 h1{margin-bottom:.1rem} .sub{color:#666;margin-top:.2rem}
 fieldset{border:1px solid #ddd;border-radius:10px;margin:1.2rem 0;padding:1rem 1.2rem}
 legend{font-weight:600;padding:0 .4rem}
 label{display:block;margin:.6rem 0 .2rem;font-size:.85rem;color:#444}
 input[type=text]{padding:.45rem;border:1px solid #ccc;border-radius:6px;width:220px}
 input[type=file]{font-size:.9rem}
 button{margin-top:.9rem;padding:.55rem 1.1rem;border:0;border-radius:7px;background:#2563eb;color:#fff;font-size:.95rem;cursor:pointer}
 button:hover{background:#1d4ed8}
 .result{margin-top:1rem;padding:.8rem .9rem;border-radius:7px;font-size:.9rem;white-space:pre-wrap;line-height:1.4}
 .ok{background:#ecfdf5;border:1px solid #6ee7b7}
 .alert{background:#fef2f2;border:1px solid #fca5a5}
 .muted{background:#f8fafc;border:1px solid #e2e8f0;color:#475569}
 code{background:#f1f5f9;padding:.1rem .3rem;border-radius:4px}
 a{color:#2563eb}
</style>
</head>
<body>
<h1>image-movement</h1>
<p class="sub">Detect reuse of the <b>same core image</b> &mdash; the same picture appearing more than once &mdash; even after zoom, pixel shift, and re-compression.</p>

<fieldset style="border-color:#2563eb">
 <legend>Submit a new image &nbsp;(check &rarr; enroll)</legend>
 <p class="sub" style="margin-top:0">The typical workflow: check the image against the corpus, alert on reuse, then enroll it for future checks.</p>
 <label>user_id</label><input id="su" type="text" value="bob">
 <label>attempt_id</label><input id="sa" type="text" value="b1">
 <label>image</label><input id="sf" type="file" accept="image/*">
 <br><button onclick="submitImg()">Submit</button>
 <div id="sr" class="result muted" style="display:none"></div>
</fieldset>

<details>
<summary style="cursor:pointer;color:#666;margin:1rem 0">&#9656; raw primitives (enroll-only / check-only)</summary>

<fieldset>
 <legend>Enroll only</legend>
 <label>user_id</label><input id="eu" type="text" value="alice">
 <label>attempt_id</label><input id="ea" type="text" value="a1">
 <label>image</label><input id="ef" type="file" accept="image/*">
 <br><button onclick="enroll()">Enroll</button>
 <div id="er" class="result muted" style="display:none"></div>
</fieldset>

<fieldset>
 <legend>Check only</legend>
 <label>image</label><input id="cf" type="file" accept="image/*">
 <br><button onclick="check()">Check for reuse</button>
 <div id="cr" class="result muted" style="display:none"></div>
</fieldset>
</details>

<p class="sub">Raw API console: <a href="/docs">/docs</a></p>

<script>
async function post(url, fd, box){
  box.style.display='block'; box.className='result muted'; box.textContent='working\\u2026';
  try{
    const res = await fetch(url, {method:'POST', body:fd});
    const j = await res.json();
    if(!res.ok){ box.className='result alert'; box.textContent='error: '+(j.detail||res.status); return null; }
    return j;
  }catch(e){ box.className='result alert'; box.textContent='error: '+e; return null; }
}
async function submitImg(){
  const f=document.getElementById('sf').files[0];
  const box=document.getElementById('sr');
  if(!f){ box.style.display='block'; box.className='result alert'; box.textContent='pick an image first'; return; }
  const fd=new FormData();
  fd.append('user_id', document.getElementById('su').value||'-');
  fd.append('attempt_id', document.getElementById('sa').value||'-');
  fd.append('file', f);
  const j=await post('/submit', fd, box); if(!j) return;
  let s=`enrolled record ${j.record_id} (user=${j.user_id}). corpus size = ${j.corpus_size}\\n`;
  if(j.matches.length===0){ box.className='result ok'; box.textContent='\\u2713 '+s+'no prior reuse \\u2014 clean.'; return; }
  const a=j.alert;
  s += (a.triggered ? '\\u26a0 REUSE DETECTED ON SUBMIT \\u2014 ALERT TRIGGERED' : 'reuse found (below alert threshold)')
     + `\\n${j.matches.length} match(es) across ${a.distinct_users} distinct user(s), severity ${a.severity}`;
  for(const m of j.matches){ s += `\\n  \\u2022 record ${m.record_id}  user=${m.user_id}  attempt=${m.attempt_id}  inliers=${m.inliers}  residual=${m.residual}`; }
  box.className = a.triggered ? 'result alert' : 'result muted';
  box.textContent = s;
}
async function enroll(){
  const f=document.getElementById('ef').files[0];
  const box=document.getElementById('er');
  if(!f){ box.style.display='block'; box.className='result alert'; box.textContent='pick an image first'; return; }
  const fd=new FormData();
  fd.append('user_id', document.getElementById('eu').value||'-');
  fd.append('attempt_id', document.getElementById('ea').value||'-');
  fd.append('file', f);
  const j=await post('/enroll', fd, box); if(!j) return;
  box.className='result ok';
  box.textContent=`enrolled record ${j.record_id} (user=${j.user_id}, attempt=${j.attempt_id}). corpus size = ${j.corpus_size}`;
}
async function check(){
  const f=document.getElementById('cf').files[0];
  const box=document.getElementById('cr');
  if(!f){ box.style.display='block'; box.className='result alert'; box.textContent='pick an image first'; return; }
  const fd=new FormData(); fd.append('file', f);
  const j=await post('/check', fd, box); if(!j) return;
  if(j.matches.length===0){ box.className='result ok'; box.textContent='\\u2713 no reuse detected (0 matches)'; return; }
  const a=j.alert;
  let s=(a.triggered ? '\\u26a0 REUSE DETECTED \\u2014 ALERT TRIGGERED' : 'reuse found (below alert threshold)')
      + `\\n${j.matches.length} match(es) across ${a.distinct_users} distinct user(s), severity ${a.severity}`;
  for(const m of j.matches){
    s += `\\n  \\u2022 record ${m.record_id}  user=${m.user_id}  attempt=${m.attempt_id}  inliers=${m.inliers}  residual=${m.residual}  hash_dist=${m.hash_distance}`;
  }
  box.className = a.triggered ? 'result alert' : 'result muted';
  box.textContent = s;
}
</script>
</body>
</html>
"""


def create_app(config: DetectorConfig | None = None) -> FastAPI:
    app = FastAPI(
        title="image-movement",
        description="Detect reuse of the same core image across a collection of images.",
    )
    app.state.service = ReuseDetectorService(config or DetectorConfig())

    @app.post("/enroll", response_model=EnrollResponse)
    async def enroll(
        user_id: str = Form(...),
        attempt_id: str = Form("-"),
        file: UploadFile = File(...),
    ) -> EnrollResponse:
        svc: ReuseDetectorService = app.state.service
        rec = svc.enroll(_decode(await file.read()), user_id, attempt_id)
        return EnrollResponse(
            record_id=rec.id, user_id=rec.user_id, attempt_id=rec.attempt_id, corpus_size=len(svc.corpus)
        )

    @app.post("/check", response_model=CheckResponse)
    async def check(file: UploadFile = File(...)) -> CheckResponse:
        svc: ReuseDetectorService = app.state.service
        matches = svc.check(_decode(await file.read()))
        alert = svc.alert(matches)
        return CheckResponse(
            matches=[
                MatchOut(
                    record_id=m.record_id, user_id=m.user_id, attempt_id=m.attempt_id,
                    inliers=m.evidence.inliers, residual=round(m.evidence.residual, 2),
                    hash_distance=m.hash_distance,
                )
                for m in matches
            ],
            alert=AlertOut(
                triggered=alert.triggered, distinct_users=alert.distinct_users,
                severity=round(alert.severity, 3),
            ),
        )

    @app.post("/submit", response_model=SubmitResponse)
    async def submit(
        user_id: str = Form(...),
        attempt_id: str = Form("-"),
        file: UploadFile = File(...),
    ) -> SubmitResponse:
        svc: ReuseDetectorService = app.state.service
        alert, rec = svc.submit(_decode(await file.read()), user_id, attempt_id)
        return SubmitResponse(
            record_id=rec.id, user_id=rec.user_id, attempt_id=rec.attempt_id,
            corpus_size=len(svc.corpus),
            matches=[
                MatchOut(
                    record_id=m.record_id, user_id=m.user_id, attempt_id=m.attempt_id,
                    inliers=m.evidence.inliers, residual=round(m.evidence.residual, 2),
                    hash_distance=m.hash_distance,
                )
                for m in alert.matches
            ],
            alert=AlertOut(
                triggered=alert.triggered, distinct_users=alert.distinct_users,
                severity=round(alert.severity, 3),
            ),
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _DEMO_HTML

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, config: DetectorConfig | None = None) -> None:
    """Run the API with uvicorn (imported lazily so the library has no hard server dep at import)."""
    import uvicorn

    uvicorn.run(create_app(config), host=host, port=port)
