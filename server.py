#!/usr/bin/env python3
"""VoiceCut — 音频裁剪 / 视频提取 / 音频转字幕"""

import os, sys, json, re, uuid, tempfile, threading, subprocess, glob, time
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler

HOST = '127.0.0.1'
PORT = 8768
MODEL_PATH = r'D:\Develop\Workspace\voicecut\models\whisper-turbo-ct2'
FFMPEG = r'D:\Develop\ffmpeg\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe'
TEMP_DIR = os.path.join(tempfile.gettempdir(), 'voicecut')
os.makedirs(TEMP_DIR, exist_ok=True)
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voicecut.log")

# Add CUDA DLL paths for ctranslate2
_cuda_dll_paths = [
    r'D:\Develop\MyPythonLibs\nvidia\cublas\bin',
]
for _p in _cuda_dll_paths:
    if os.path.isdir(_p):
        try: os.add_dll_directory(_p)
        except Exception: pass
# Proxy settings for translate
os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:10808')
os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:10808')


# ==================== Whisper model ====================
_whisper = None
_wlock = threading.Lock()

def get_whisper():
    global _whisper
    if _whisper is None:
        with _wlock:
            if _whisper is None:
                from faster_whisper import WhisperModel
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    _whisper = WhisperModel(MODEL_PATH, device='cuda', compute_type='float16')
                    print(f'[Whisper] Using GPU (CUDA) with float16')
                else:
                    _whisper = WhisperModel(MODEL_PATH, device='cpu', compute_type='int8',
                                           cpu_threads=max(1, (os.cpu_count() or 8) - 1), num_workers=2)
                    print(f'[Whisper] Using CPU with int8')
    return _whisper


# ==================== SRT helpers ====================
def fmt_srt(sec):
    h=int(sec//3600); m=int((sec%3600)//60); s=int(sec%60); ms=int((sec%1)*1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

def to_srt(segs):
    out=[]
    for i, sg in enumerate(segs, 1):
        t=sg.text.strip()
        if t:
            out.extend([str(i), f"{fmt_srt(sg.start)} --> {fmt_srt(sg.end)}", t, ''])
    return '\n'.join(out)
# ==================== Transcribe ====================
def transcribe_audio(path, sse, lang=None, beam_size=5):
    model = get_whisper()
    if lang == 'auto': lang = None

    def log(msg):
        sse('log', {'text': msg})
        print(msg, end='', flush=True)
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as lf:
                lf.write(f'[{time.strftime("%H:%M:%S")}] {msg}')
        except: pass

    src_dir = os.path.dirname(os.path.abspath(path)) or '.'
    src_stem = os.path.splitext(os.path.basename(path))[0]
    seg_pattern = os.path.join(src_dir, f'{src_stem}_seg_%03d.mp3')

    # Clean up old segment files from previous runs
    for oldf in glob.glob(os.path.join(src_dir, f'{src_stem}_seg_*.mp3')):
        try: os.remove(oldf)
        except: pass

    t0 = time.time()
    log(f'[1/4] Splitting audio into 60s segments via FFmpeg...\n')
    log(f'  Source: {path}\n  Pattern: {seg_pattern}\n')
    subprocess.run([FFMPEG, '-y', '-i', path, '-f', 'segment', '-segment_time', '60',
                    '-c:a', 'libmp3lame', '-b:a', '64k', '-reset_timestamps', '1', seg_pattern],
                   capture_output=True, check=True)
    log(f'  Split done ({time.time()-t0:.1f}s)\n')

    seg_files = sorted(glob.glob(os.path.join(src_dir, f'{src_stem}_seg_*.mp3')))
    if not seg_files:
        log(f'ERROR: No segment files found matching {src_stem}_seg_*.mp3 in {src_dir}\n')
        sse('error', {'text': f'No segment files found'})
        return

    sizes = [os.path.getsize(sp) for sp in seg_files]
    total_segs = len(seg_files)
    log(f'[2/4] Found {total_segs} segments ({sum(sizes)/1024/1024:.1f} MB total)\n')
    for i, (sp, sz) in enumerate(zip(seg_files, sizes)):
        log(f'  [{i+1}] {os.path.basename(sp)} ({sz/1024:.0f} KB)\n')

    import ctranslate2
    _is_cuda = ctranslate2.get_cuda_device_count() > 0
    log(f'[3/4] Starting transcription on {"GPU" if _is_cuda else "CPU"}...\n')

    all_segs = []

    def do_one_segment(idx, sp):
        sz = os.path.getsize(sp)
        log(f'  [{idx+1}/{total_segs}] Transcribing {os.path.basename(sp)} ({sz/1024:.0f} KB)...\n')
        t1 = time.time()
        segs_iter, info = model.transcribe(sp, beam_size=beam_size, vad_filter=True, language=lang)
        seg_duration = info.duration or 60.0
        local = []
        offset = idx * 60.0
        last_pct = -1
        phrase_count = 0
        for sg in segs_iter:
            sg.start += offset
            sg.end += offset
            local.append(sg)
            phrase_count += 1
            intra = min(1.0, max(0, (sg.end - offset) / max(seg_duration, 1.0)))
            pct = int(intra * 100)
            if pct - last_pct >= 20:
                sse('progress', {'percent': int(((idx + intra) / total_segs) * 100),
                    'text': f'Segment {idx+1}/{total_segs} ({pct}%)'})
                last_pct = pct
        elapsed = time.time() - t1
        log(f'  [{idx+1}/{total_segs}] Done: {phrase_count} phrases in {elapsed:.1f}s\n')
        return local, idx

    if _is_cuda:
        sse('status', f'Transcribing {total_segs} segments (GPU sequential)...')
        for idx, sp in enumerate(seg_files):
            try:
                local, seg_idx = do_one_segment(idx, sp)
                all_segs.extend(local)
                pct = int(((idx + 1) / total_segs) * 100)
                sse('progress', {'percent': pct, 'text': f'Segment {idx+1}/{total_segs} done'})
            except Exception as e:
                log(f'  [{idx+1}/{total_segs}] FAILED: {e}\n')
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        MAX_WORKERS = min(2, total_segs)
        sse('status', f'Transcribing {total_segs} segments ({MAX_WORKERS} CPU workers)...')
        done = 0; lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(do_one_segment, i, p): i for i, p in enumerate(seg_files)}
            for fut in as_completed(futures):
                try:
                    local, seg_idx = fut.result()
                    with lock:
                        all_segs.extend(local)
                        done += 1
                        pct = int((done / total_segs) * 100)
                        sse('progress', {'percent': pct, 'text': f'Segment {done}/{total_segs} done'})
                except Exception as e:
                    log(f'  Segment FAILED: {e}\n')

    log(f'[4/4] Merging {len(all_segs)} phrases...\n')
    all_segs.sort(key=lambda s: s.start)

    for sp in seg_files:
        try: os.remove(sp)
        except: pass

    log(f'[Done] {len(all_segs)} total phrases in {time.time()-t0:.1f}s\n')
    srt = to_srt(all_segs)
    sse('result', {'srt': srt, 'count': len(all_segs)})

def extract_audio_ffmpeg(path, fmt, out):
    cmd = [FFMPEG, '-y', '-i', path, '-vn',
           '-c:a', 'pcm_s16le' if fmt == 'wav' else 'libmp3lame',
           *([] if fmt == 'wav' else ['-b:a', '192k']),
           '-rf64', 'auto', '-threads', '0', out]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    lines = []
    for l in proc.stdout: lines.append(l)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(''.join(lines[-5:])[:300])
    return ''.join(lines)


# ==================== HTML ====================
HTML = r'''<!DOCTYPE html>

<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VoiceCut</title>
<script src="https://unpkg.com/lucide@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/lamejs@1.2.1/lame.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0c0c16;--surface:#16162a;--border:#2a2a48;--text:#e2e2ee;--accent:#5b8def;--accent-hover:#7aa5ff;--radius:6px}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
body{display:flex;justify-content:center;align-items:center}
.app{width:min(860px,95vw);height:min(740px,92vh);background:var(--surface);border-radius:12px;border:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;box-shadow:0 8px 48px rgba(0,0,0,.4)}
.header{display:flex;align-items:center;padding:12px 20px;gap:12px;border-bottom:1px solid var(--border);flex-shrink:0}
.header h1{font-size:17px;font-weight:600;display:flex;align-items:center;gap:8px}
.header h1 svg{width:22px;height:22px;stroke:var(--accent)}
.tabs{display:flex;gap:4px;margin-left:auto}
.tabs button{padding:7px 14px;border:1px solid transparent;border-radius:6px;background:transparent;color:#7a7a9a;font-size:13px;cursor:pointer;display:flex;align-items:center;gap:5px}
.tabs button:hover{color:var(--text)}
.tabs button.active{color:var(--accent);border-color:var(--border);background:rgba(91,141,239,.08)}
.tabs button svg{width:15px;height:15px;stroke:currentColor}
.panel{display:none;flex-direction:column;flex:1;min-height:0;padding:16px 20px;gap:10px}
.panel.active{display:flex}
.bar{height:4px;background:var(--border);border-radius:2px;overflow:hidden;flex-shrink:0}
.bar-fill{height:100%;width:0;background:var(--accent);border-radius:2px;transition:width .3s}
.row{display:flex;gap:8px;flex-shrink:0}
.row input{flex:1;padding:8px 10px;border:1px solid var(--border);border-radius:6px;background:#0c0c18;color:var(--text);font-size:13px;outline:none}
.row input:focus{border-color:var(--accent)}
.row input::placeholder{color:#7a7a9a;opacity:.4}
.row label.btn{padding:8px 12px;border:1px solid var(--border);border-radius:6px;cursor:pointer;display:inline-flex;align-items:center;gap:4px;font-size:13px}
.row label.btn:hover{border-color:var(--accent)}
.row label.btn svg{width:16px;height:16px;stroke:var(--text)}
.fmt-row{display:flex;gap:12px;align-items:center;flex-shrink:0}
.fmt-row select{padding:6px 10px;border:1px solid var(--border);border-radius:6px;background:transparent;color:var(--text);font-size:13px;cursor:pointer}
.fmt-row label{font-size:12px;color:#7a7a9a}
.actions{display:flex;gap:8px;flex-shrink:0}
.actions button{display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:8px 14px;border-radius:6px;font-size:13px;cursor:pointer;border:none;background:var(--accent);color:#fff}
.actions button:hover{background:var(--accent-hover)}
.actions button:disabled{opacity:.4;cursor:not-allowed}
.actions button svg{width:16px;height:16px;stroke:#fff}
.actions .sec{background:transparent;border:1px solid var(--border);color:var(--text);display:inline-flex;align-items:center;justify-content:center;gap:6px}
.actions .sec:hover{background:rgba(30,30,58,.6);border-color:var(--accent)}
.actions .sec svg{stroke:var(--text)}
.actions .spacer{flex:1}
.status{font-size:12px;color:#7a7a9a;flex-shrink:0;min-height:18px}
.status.done{color:#3dce7a}
.status.err{color:#ef6b3a}
.area{flex:1;min-height:0;display:flex;flex-direction:column;gap:4px}
.area label{font-size:11px;color:#7a7a9a;display:flex;align-items:center;gap:4px}
.area label svg{width:13px;height:13px}
.area textarea{flex:1;background:#0c0c18;border:1px solid var(--border);border-radius:6px;color:var(--text);font-family:'Cascadia Code','JetBrains Mono','Consolas',monospace;font-size:13px;padding:10px;resize:none;outline:none;width:100%;min-height:40px}
.area textarea::placeholder{color:#7a7a9a;opacity:.3}
.wf-wrap{flex:1;min-height:0;position:relative;user-select:none}
.wf-wrap canvas{display:block;width:100%;height:100%}
.wf-bar{position:absolute;bottom:0;left:0;right:0;height:20px;background:rgba(0,0,0,.3);display:flex;align-items:center;padding:0 8px;gap:6px;font-size:10px;color:#7a7a9a}
.wf-bar select,.wf-bar button{padding:2px 6px;border:1px solid var(--border);border-radius:4px;background:transparent;color:#7a7a9a;font-size:10px;cursor:pointer}
.wf-bar button:hover{border-color:var(--accent);color:var(--text)}
.wf-bar button svg{width:12px;height:12px}
.wf-bar .spacer{flex:1}
@media(max-width:600px){.app{width:100vw;height:100vh;border-radius:0}.tabs button{padding:5px 8px;font-size:12px}}
.main-container{display:flex;gap:0;width:min(1160px,98vw);height:min(760px,94vh);background:var(--surface);border-radius:12px;border:1px solid var(--border);overflow:hidden;box-shadow:0 8px 48px rgba(0,0,0,.4)}
.main-container .app{width:auto;height:100%;border-radius:0;border:none;box-shadow:none;flex:1;min-width:0}
.sidebar{width:280px;flex-shrink:0;background:rgba(0,0,0,.2);border-right:1px solid var(--border);display:flex;flex-direction:column;gap:0}
.sidebar-header{padding:12px 14px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;flex-shrink:0}
.sidebar-header h2{font-size:13px;font-weight:600;display:flex;align-items:center;gap:6px}
.sidebar-header h2 svg{width:16px;height:16px;stroke:var(--accent)}
.sidebar-actions{padding:8px 10px;display:flex;gap:6px;flex-shrink:0;border-bottom:1px solid var(--border)}
.sidebar-actions button,.queue-clear button{display:flex;align-items:center;justify-content:center;gap:4px;padding:6px 10px;border:1px solid var(--border);border-radius:6px;background:transparent;color:var(--text);font-size:11px;cursor:pointer}
.sidebar-actions input{flex:1;min-width:0;padding:6px 8px;border:1px solid var(--border);border-radius:6px;background:#0c0c18;color:var(--text);font-size:11px;outline:none}
.sidebar-actions input:focus{border-color:var(--accent)}
.sidebar-actions button{flex:0;padding:6px 10px;border:1px solid var(--border);border-radius:6px;background:transparent;color:var(--text);font-size:11px;cursor:pointer;display:flex;align-items:center;justify-content:center}
.sidebar-actions button:hover{background:rgba(91,141,239,.08);border-color:var(--accent)}
.sidebar-actions button svg{width:14px;height:14px;stroke:var(--text)}
.queue-list{flex:1;overflow-y:auto;padding:6px 0;min-height:0}
.queue-item{padding:8px 10px;margin:2px 8px;border-radius:6px;background:rgba(0,0,0,.15);border:1px solid transparent;display:flex;flex-direction:column;gap:4px;font-size:11px;transition:background .15s}
.queue-item.active{background:rgba(91,141,239,.1);border-color:var(--accent)}
.queue-item .qi-name{font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--text)}
.queue-item .qi-step{font-size:10px;color:#7a7a9a}
.queue-item .qi-bar{height:3px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:2px}
.queue-item .qi-bar-fill{height:100%;width:0;background:var(--accent);border-radius:2px;transition:width .3s}
.queue-item .qi-bar-fill.done{background:var(--accent)}
.queue-item .qi-bar-fill.err{background:#ef6b3a}
.queue-item .qi-status{font-size:10px;display:flex;align-items:center;gap:3px}
.queue-item .qi-status svg{width:12px;height:12px}
.queue-item .qi-status.done{color:#3dce7a}
.queue-item .qi-status.err{color:#ef6b3a}
.queue-item .qi-status.wait{color:#7a7a9a}
.queue-clear{padding:6px 10px;border-top:1px solid var(--border);flex-shrink:0}
.queue-clear button{flex:1;display:flex;align-items:center;justify-content:center;gap:4px;padding:7px 10px;border:1px solid var(--border);border-radius:6px;font-size:11px;cursor:pointer}
.queue-clear button:hover{color:var(--text);border-color:var(--accent)}
@media(max-width:900px){.main-container{flex-direction:column}.sidebar{width:100%;max-height:200px;border-right:none;border-bottom:1px solid var(--border)}}

</style>
</head>
<body>
<div class="main-container">
<div class="sidebar">
  <div class="sidebar-header">
    <h2><svg data-lucide="list-ordered" stroke-width="1.5"></svg>队列</h2>
    <span style="font-size:10px;color:#7a7a9a;margin-left:auto" id="q-count">0 个文件</span>
  </div>
  <div class="sidebar-actions">
    <input type="text" id="q-path-input" placeholder="文件路径，多个用换行或分号分隔" style="flex:1;padding:6px 8px;border:1px solid var(--border);border-radius:6px;background:#0c0c18;color:var(--text);font-size:11px;outline:none">
    <button id="q-add-path" style="flex:0;width:auto"><svg data-lucide="plus" stroke-width="1.5"></svg></button>
  </div>
  <div class="queue-list" id="q-list">
    <div style="text-align:center;padding:20px;color:#7a7a9a;font-size:11px">添加文件路径后点击开始</div>
  </div>
  <div class="queue-clear" style="display:flex;gap:6px">
    <button id="q-clear" disabled style="flex:1;display:flex;align-items:center;justify-content:center;gap:4px">清空队列</button>
    <button id="q-start" disabled style="flex:1;display:flex;align-items:center;justify-content:center;gap:4px;background:var(--accent);color:#fff;border:none;border-radius:6px;font-size:12px;font-weight:500">开始处理</button>
  </div>
</div>
<div class="app">
<div class="header">
<h1><svg data-lucide="scissors" stroke-width="1.5"></svg>VoiceCut</h1>
<div class="tabs">
<button class="active" data-tab="cutter"><svg data-lucide="audio-lines" stroke-width="1.5"></svg>裁剪</button>
<button data-tab="extract"><svg data-lucide="video" stroke-width="1.5"></svg>提取</button>
<button data-tab="subtitle"><svg data-lucide="subtitles" stroke-width="1.5"></svg>字幕</button>
</div>
</div>

<!-- ===== Cutter ===== -->
<div class="panel active" id="p-cutter">
<div class="row">
<input type="text" id="cp-path" placeholder="粘贴音频路径，或点击选择文件">

</div>
<div class="wf-wrap">
<canvas id="cp-canvas"></canvas>
<div class="wf-bar">
<span class="spacer"></span>
<span id="cp-sel" style="color:#7a7a9a">在波形上拖拽选择区域</span>
<span class="spacer"></span>
<button id="cp-play"><svg data-lucide="play" stroke-width="2"></svg></button>
<select id="cp-fmt"><option value="mp3">MP3</option><option value="wav">WAV</option></select>
<button id="cp-export">导出选区</button>
</div>
</div>
</div>

<!-- ===== Extract ===== -->
<div class="panel" id="p-extract">
<div class="row">
<input type="text" id="ex-path" placeholder="选择视频文件，选择后自动上传处理">

</div>
<div class="fmt-row">
<label>输出格式</label>
<select id="ex-fmt"><option value="mp3">MP3</option><option value="wav">WAV</option></select>
</div>
<div class="bar"><div class="bar-fill" id="ex-bar"></div></div>
<div class="status" id="ex-st">就绪</div>
<div class="area" style="flex:1">
<label><svg data-lucide="terminal" stroke-width="1.5"></svg>FFmpeg 日志</label>
<textarea id="ex-log" readonly placeholder="日志输出..." wrap="off"></textarea>
</div>
<div class="actions">
<button id="ex-start"><svg data-lucide="wand" stroke-width="1.5"></svg>提取音频</button>
</div>
</div>

<!-- ===== Subtitle ===== -->
<div class="panel" id="p-subtitle">
<div class="row">
<input type="text" id="sb-path" placeholder="选择音频/视频文件">

</div>
<div class="fmt-row">
<label>识别语言</label>
<select id="sb-lang">
<option value="auto">自动检测</option>
<option value="zh">中文</option>
<option value="ja">日文</option>
<option value="en">英文</option>
<option value="ko">韩文</option>
</select>
<label style="margin-left:16px">速度</label>
<select id="sb-speed">
<option value="fast" selected>快速</option>
<option value="default">标准</option>
</select>
</div>
<div class="bar"><div class="bar-fill" id="sb-bar"></div></div>
<div class="status" id="sb-st">就绪</div>
<div class="area" style="flex:1">
<label><svg data-lucide="file-text" stroke-width="1.5"></svg>SRT 字幕</label>
<textarea id="sb-out" placeholder="转录完成后的字幕将显示在此处..."></textarea>
</div>
<div class="actions">
 <button id="sb-start"><svg data-lucide="wand" stroke-width="1.5"></svg>开始转录</button>
 <button class="sec" id="sb-trans"><svg data-lucide="languages" stroke-width="1.5"></svg>翻译中文</button>
<div class="spacer"></div>
<button class="sec" id="sb-copy" disabled><svg data-lucide="copy" stroke-width="1.5"></svg>复制</button>
<button class="sec" id="sb-dl" disabled><svg data-lucide="download" stroke-width="1.5"></svg>下载 SRT</button>
</div>
</div>

<script>
const $=id=>document.getElementById(id)

// Tabs
document.querySelectorAll('.tabs button').forEach(b=>b.addEventListener('click',()=>{
document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('active'))
document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'))
b.classList.add('active');$('p-'+b.dataset.tab).classList.add('active')
}))



// SSE reader helper
async function readSSE(response, handlers){
const rd=response.body.getReader(),dc=new TextDecoder();let buf=''
while(true){
const{done,value}=await rd.read();if(done)break
buf+=dc.decode(value,{stream:true})
const ps=buf.split('\n\n');buf=ps.pop()||''
for(const ev of ps){
let ty='',da=''
for(const ln of ev.split('\n')){if(ln.startsWith('event: '))ty=ln.slice(7);else if(ln.startsWith('data: '))da=ln.slice(6)}
if(!da)continue
try{const j=JSON.parse(da);(handlers[ty]||(()=>{}))(j)}catch(e){}
}
}
}

// Upload helper - returns fetch promise using FormData if file available, else JSON with path
function sendFileOrPath(fid, pathId, url, extraFields){
const f=$(fid)?.files[0]
const p=$(pathId).value.trim()
if(f){const fd=new FormData();fd.append('file',f);if(extraFields)Object.entries(extraFields).forEach(([k,v])=>fd.append(k,v));return fetch(url,{method:'POST',body:fd})}
if(p){return fetch(url.replace('-upload',''),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:p,...extraFields||{}})})}
return Promise.reject(new Error('No file or path'))
}

// ========== Cutter ==========
;(function(){
const cv=$('cp-canvas'),ctx=cv.getContext('2d')
const play=$('cp-play'),exportBtn=$('cp-export'),fmt=$('cp-fmt'),sel=$('cp-sel')
let buf=null,ch=0,sr=0,len=0,ss=0,se=0,hasSel=false,playing=false,srcNode=null

function resize(){
const p=cv.parentElement,pr=window.devicePixelRatio||1
cv.width=p.clientWidth*pr;cv.height=p.clientHeight*pr
cv.style.width=p.clientWidth+'px';cv.style.height=p.clientHeight+'px'
draw()
}
function draw(){
if(!buf)return;const w=cv.width,h=cv.height;ctx.clearRect(0,0,w,h)
const yd=4,wr=w,wh=h-yd*2,cy=yd+wh/2,amp=wh*.44
const dur=len/sr,pps=wr/dur
for(let px=0;px<wr;px++){
const t=px/pps,si=Math.floor(t*sr),ei=Math.floor((t+1/pps)*sr)
let mx=0;for(let c=0;c<ch;c++){const d=buf[c];for(let s=Math.max(0,si);s<Math.min(ei,d.length);s++)mx=Math.max(mx,Math.abs(d[s]))}
const x=px,hv=Math.max(mx*amp,.5),inSel=hasSel&&t>=ss&&t<=se
if(inSel){ctx.fillStyle='rgba(239,107,58,.07)';ctx.fillRect(x,yd,1,wh)}
ctx.fillStyle=inSel?'#ef6b3a':'#5b8def';ctx.fillRect(x,cy-hv,1,hv*2)
}
if(hasSel){
const l=ss*pps,r=se*pps;ctx.strokeStyle='rgba(239,107,58,.5)';ctx.lineWidth=1;ctx.setLineDash([4,4])
ctx.beginPath();ctx.moveTo(l,yd);ctx.lineTo(l,yd+wh);ctx.stroke()
ctx.beginPath();ctx.moveTo(r,yd);ctx.lineTo(r,yd+wh);ctx.stroke();ctx.setLineDash([])
ctx.fillStyle='rgba(239,107,58,.7)';ctx.font='11px sans-serif';ctx.textAlign='center'
const d=se-ss;ctx.fillText(Math.round(d/60)+':'+String(Math.round(d%60)).padStart(2,'0'),(l+r)/2,yd+12)
}
}

// Cutter: load audio from path input (paste path, then click waveform area)

let mdown=false,mx0=0
cv.addEventListener('mousedown',e=>{if(!buf)return;const r=cv.getBoundingClientRect();mx0=(e.clientX-r.left)*(window.devicePixelRatio||1);mdown=true
const t=mx0/cv.width*(len/sr);ss=t;se=t;hasSel=true;draw()})
window.addEventListener('mousemove',e=>{if(!mdown||!buf)return
const r=cv.getBoundingClientRect();const x=(e.clientX-r.left)*(window.devicePixelRatio||1)
const t=x/cv.width*(len/sr);const t0=mx0/cv.width*(len/sr);ss=Math.min(t0,t);se=Math.max(t0,t);draw()})
window.addEventListener('mouseup',()=>{if(!mdown)return;mdown=false;if(Math.abs(se-ss)<.005)hasSel=false;draw()})

play.addEventListener('click',()=>{
if(playing){stop();return}
if(!buf||!hasSel)return
const ac=new AudioContext();if(ac.state==='suspended')ac.resume()
const ab=ac.createBuffer(ch,len/ac.sampleRate,ac.sampleRate)
for(let i=0;i<ch;i++)ab.copyToChannel(buf[i],i)
srcNode=ac.createBufferSource();srcNode.buffer=ab;srcNode.connect(ac.destination)
srcNode.start(0,ss,se-ss);playing=true;srcNode.onended=()=>{playing=false;draw()}
play.innerHTML='<svg data-lucide="pause" stroke-width="2"></svg>';lucide.createIcons()
})
function stop(){playing=false;if(srcNode)try{srcNode.stop()}catch(e){}srcNode=null
play.innerHTML='<svg data-lucide="play" stroke-width="2"></svg>';lucide.createIcons()}

exportBtn.addEventListener('click',()=>{
if(!buf||!hasSel)return
const st=Math.round(ss*sr),ed=Math.round(se*sr),ln=ed-st,ch2=ch
const inter=new Float32Array(ln*ch2)
for(let c=0;c<ch2;c++){const d=buf[c];for(let i=0;i<ln;i++)inter[i*ch2+c]=d[st+i]}
const stem=$('cp-path').value.replace(/^.*[/\\\\]/,'').replace(/[.][^.]+$/,'')||'audio'
const ext=fmt.value
if(ext==='wav'){exportWav(inter,sr,ch2,stem)}else{exportMp3(inter,sr,ch2,stem,buf,st,ln)}
})
window.addEventListener('resize',()=>{if(buf)resize()})

function exportWav(inter,sr,ch2,sn){
const bps=16,br=sr*ch2*bps/8,ba=ch2*bps/8,ds=inter.length*bps/8,bs=44+ds
const w=new ArrayBuffer(bs),v=new DataView(w)
v.setUint32(0,0x46464952,true);v.setUint32(4,bs-8,true)
v.setUint32(8,0x45564157,true);v.setUint32(12,0x20746d66,true)
v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,ch2,true)
v.setUint32(24,sr,true);v.setUint32(28,br,true);v.setUint16(32,ba,true);v.setUint16(34,bps,true)
v.setUint32(36,0x61746164,true);v.setUint32(40,ds,true)
let o=44;for(let i=0;i<inter.length;i++){let s=Math.max(-1,Math.min(1,inter[i]));s=s<0?s*0x8000:s*0x7FFF;v.setInt16(o,s,true);o+=2}
const blob=new Blob([w],{type:'audio/wav'}),url=URL.createObjectURL(blob)
const a=document.createElement('a');a.href=url;a.download=sn+'_cut.wav'
document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url)
}
function exportMp3(inter,sr,ch2,sn,buf2,st2,ln2){
if(typeof lamejs==='undefined'){alert('MP3 encoder not loaded');return}
const enc=new lamejs.Mp3Encoder(ch2,sr,192),mp3=[],cs=1152
if(ch2===1){const s=new Int16Array(ln2);for(let i=0;i<ln2;i++){let v=Math.max(-1,Math.min(1,buf2[0][st2+i]));s[i]=v<0?v*0x8000:v*0x7FFF}
for(let i=0;i<ln2;i+=cs){const m=enc.encodeBuffer(s.subarray(i,Math.min(i+cs,ln2)));if(m.length)mp3.push(m)}}
else{const l=new Int16Array(ln2),r2=new Int16Array(ln2)
for(let i=0;i<ln2;i++){let v=Math.max(-1,Math.min(1,buf2[0][st2+i]));l[i]=v<0?v*0x8000:v*0x7FFF;v=Math.max(-1,Math.min(1,buf2[1][st2+i]));r2[i]=v<0?v*0x8000:v*0x7FFF}
for(let i=0;i<ln2;i+=cs){const e=Math.min(i+cs,ln2);const m=enc.encodeBuffer(l.subarray(i,e),r2.subarray(i,e));if(m.length)mp3.push(m)}}
const fl=enc.flush();if(fl.length)mp3.push(fl)
const blob=new Blob(mp3,{type:'audio/mpeg'}),url=URL.createObjectURL(blob)
const a=document.createElement('a');a.href=url;a.download=sn+'_cut.mp3'
document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url)
}
})()

// ========== Extract ==========
;(function(){
const s=$('ex-start'),p=$('ex-path'),fm=$('ex-fmt'),st=$('ex-st'),bf=$('ex-bar'),lg=$('ex-log')
function set(m,c,pc){st.textContent=m;st.className='status'+(c?' '+c:'');if(pc!=null)bf.style.width=Math.min(100,pc)+'%'}
function log(t){lg.value+=t;lg.scrollTop=lg.scrollHeight}
s.addEventListener('click',async()=>{
const fp=p.value.trim()
if(!fp){set('请输入视频文件路径','err');return}
s.disabled=true;lg.value='';set('正在提取...','',0)
try{
let r=await fetch('/extract',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:fp,format:fm.value})})
if(!r.ok){const t=await r.text();throw new Error(t)}
await readSSE(r,{log:j=>log(j.text),done:j=>{set('提取完成!','done',100);log('\n[完成] '+j.text+'\n')},error:j=>{set('错误: '+j.text,'err');log('[错误] '+j.text+'\n')}})
}catch(e){set('连接错误: '+e.message,'err')}
s.disabled=false
})
})()

// ========== Subtitle ==========
;(function(){
const btn=$('sb-start'),p=$('sb-path'),out=$('sb-out'),st=$('sb-st'),bf=$('sb-bar'),cp=$('sb-copy'),dl=$('sb-dl')
function set(m,c,pc){st.textContent=m;st.className='status'+(c?' '+c:'');if(pc!=null)bf.style.width=Math.min(100,pc)+'%'}
btn.addEventListener('click',async()=>{
const fp=p.value.trim()
_srtOriginal=out.value;_srtTranslated='';_isTranslated=false
const lang=$('sb-lang')?.value||'auto',speed=$('sb-speed')?.value||'fast'
if(!fp){set('请输入文件路径','err');return}
btn.disabled=true;cp.disabled=true;dl.disabled=true;out.value='';set('正在处理...','',0)
try{
let r=await fetch('/transcribe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:fp,language:lang,speed:speed})});if(!r.ok){const t=await r.text();throw new Error(t)}
await readSSE(r,{log:j=>{const ta=document.getElementById('sb-out');if(ta)ta.value+=j.text;ta.scrollTop=ta.scrollHeight},status:j=>{set(j,'',0)},progress:j=>{set(j.text,'',j.percent)},
result:j=>{out.value=j.srt;set('完成! '+j.count+' 段','done',100);cp.disabled=false;dl.disabled=false},
error:j=>{set('错误: '+j.text,'err')}})
}catch(e){set('连接错误: '+e.message,'err')}
btn.disabled=false
})
let _srtOriginal='',_srtTranslated='',_isTranslated=false
const transBtn=$('sb-trans')
transBtn.addEventListener('click',async()=>{
  if(_isTranslated){
    out.value=_srtOriginal
    transBtn.innerHTML='<svg data-lucide="languages" stroke-width="1.5"></svg>翻译中文'
    _isTranslated=false;lucide.createIcons()
    return
  }
  if(!_srtTranslated){
    transBtn.disabled=true
    try{
      const r=await fetch('/translate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({srt:out.value})})
      if(!r.ok)throw new Error(await r.text())
      const j=await r.json()
      _srtOriginal=out.value;_srtTranslated=j.srt
    }catch(e){transBtn.disabled=false;return}
  }
  out.value=_srtTranslated
  transBtn.innerHTML='<svg data-lucide="undo-2" stroke-width="1.5"></svg>恢复原文'
  _isTranslated=true;transBtn.disabled=false;lucide.createIcons()
})
out.addEventListener('input',()=>{_srtTranslated='';_isTranslated=false;transBtn.innerHTML='<svg data-lucide="languages" stroke-width="1.5"></svg>翻译中文';lucide.createIcons()})
cp.addEventListener('click',()=>{out.select();navigator.clipboard.writeText(out.value).catch(()=>{})})
dl.addEventListener('click',()=>{
const s2=p.value.replace(/^.*[/\\\\]/,'').replace(/[.][^.]+$/,'')||'字幕'
const blob=new Blob([out.value],{type:'text/plain;charset=utf-8'}),url=URL.createObjectURL(blob)
const a=document.createElement('a');a.href=url;a.download=s2+'.srt'
document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url)
})
})()
lucide.createIcons()

// ========== Queue Panel ==========
;(function(){
const list=document.getElementById("q-list"),countEl=document.getElementById("q-count"),clearBtn=document.getElementById("q-clear")
const pathInput=document.getElementById("q-path-input"),addBtn=document.getElementById("q-add-path"),startBtn=document.getElementById("q-start")

let queue=[],running=false,currentIdx=-1

function updateUI(){
  countEl.textContent=queue.length+" 个文件"
  clearBtn.disabled=queue.length===0||running
  startBtn.disabled=queue.length===0||running
  list.innerHTML=""
  if(queue.length===0){
    list.innerHTML='<div style="text-align:center;padding:20px;color:#7a7a9a;font-size:11px">添加文件路径后点击开始</div>'
    return
  }
  queue.forEach((f,i)=>{
    const d=document.createElement("div")
    d.className="queue-item"+(i===currentIdx?" active":"")
    const stepLabels={extract:"提取音频",transcribe:"转录字幕",translate:"翻译字幕"}
    const stepN=f.step?stepLabels[f.step]||"":""
    const pc=f.percent||0
    const s=f.status||"wait"
    let icon="circle",color="wait",statusText="等待中"
    if(s==="done"){icon="check-circle";color="done";statusText="完成"}
    else if(s==="error"){icon="x-circle";color="err";statusText="失败"}
    else if(s==="running"){icon="loader";color="wait";statusText=pc+"%"}
    d.innerHTML='<div class="qi-name" title="'+f.name+'">'+f.name+'</div>'+
      (stepN?'<div class="qi-step">'+stepN+' ('+f.step_n+'/3)</div>':'')+
      '<div class="qi-bar"><div class="qi-bar-fill '+(s==="error"?"err":"")+'" style="width:'+Math.max(0,Math.min(100,pc))+'%"></div></div>'+
      '<div class="qi-status '+color+'"><svg data-lucide="'+icon+'" stroke-width="2"></svg>'+statusText+'</div>'
    list.appendChild(d)
  })
  lucide.createIcons()
}

async function processQueue(){
  if(running||queue.length===0)return
  running=true;clearBtn.disabled=true;startBtn.disabled=true;updateUI()
  for(let i=0;i<queue.length;i++){
    if(!running)break
    if(queue[i].status==="done"||queue[i].status==="error")continue
    currentIdx=i
    queue[i].status="running"
    queue[i].step="extract";queue[i].step_n=1;queue[i].step_total=3;queue[i].percent=0
    updateUI()
    try{
      const r=await fetch("/pipeline",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({files:[queue[i].path],output_dir:"D:\\Downloads"})})
      if(!r.ok)throw new Error(await r.text())
      const reader=r.body.getReader();const dec=new TextDecoder();let buf=""
      while(true){
        const{done,value}=await reader.read()
        if(done)break
        buf+=dec.decode(value,{stream:true})
        const ps=buf.split("\n\n");buf=ps.pop()||""
        for(const ev of ps){
          let ty="",da=""
          for(const ln of ev.split("\n")){if(ln.startsWith("event: "))ty=ln.slice(7);else if(ln.startsWith("data: "))da=ln.slice(6)}
          if(!da)continue
          try{const j=JSON.parse(da)
          if(ty==="file_progress"&&j.file_index===0){
            queue[i].step=j.step;queue[i].percent=Math.max(0,j.percent)
            queue[i].step_n=j.step_n;queue[i].step_total=j.step_total
            updateUI()
          }else if(ty==="file_done"&&j.file_index===0){
            queue[i].status="done";queue[i].percent=100;updateUI()
          }else if(ty==="file_error"&&j.file_index===0){
            queue[i].status="error";updateUI()
          }}catch(e){}
        }
      }
    }catch(e){
      queue[i].status="error";updateUI()
    }
  }
  currentIdx=-1;running=false;clearBtn.disabled=false;startBtn.disabled=false
  updateUI()
}

function addPath(){
  const raw=pathInput.value.trim();if(!raw)return
  const paths=raw.split(/[\n;]+/)
  for(const p of paths){
    const pt=p.trim();if(!pt)continue
    const name=pt.replace(/^.*[\\\/]/,"")
    if(!queue.find(function(q){return q.path===pt}))queue.push({name:name,path:pt,status:"wait",percent:0})
  }
  pathInput.value="";updateUI()
}

addBtn.addEventListener("click",addPath)
pathInput.addEventListener("keydown",function(e){if(e.key==="Enter")addPath()})

clearBtn.addEventListener("click",function(){if(!running){queue=[];currentIdx=-1;updateUI()}})
startBtn.addEventListener("click",function(){if(!running)processQueue()})

document.addEventListener("dragover",function(e){e.preventDefault()})
document.addEventListener("drop",function(e){
  e.preventDefault()
  for(var fi=0;fi<e.dataTransfer.files.length;fi++){
    var f=e.dataTransfer.files[fi]
    var p=f.path||f.name
    if(p&&!queue.find(function(q){return q.path===p}))queue.push({name:f.name,path:p,status:"wait",percent:0})
  }
  updateUI()
})
})()

lucide.createIcons()
</script>
</div>
</div>
</body>
</html>'''


# ==================== HTTP Handler ====================
class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        p = urlparse(self.path).path
        if p == '/': self._serve_html()
        else: self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path).path
        if p == '/transcribe': self._handle_transcribe()
        elif p == '/transcribe-upload': self._handle_upload('transcribe')
        elif p == '/extract': self._handle_extract()
        elif p == '/extract-upload': self._handle_upload('extract')
        elif p == '/pipeline': self._handle_pipeline()
        elif p == '/translate': self._handle_translate()
        else: self.send_error(404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML.encode('utf-8'))

    def _sse(self, type_, data):
        try:
            self.wfile.write(f'event: {type_}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'.encode('utf-8'))
            self.wfile.flush()
        except: pass

    def _start_sse(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()

    def _handle_transcribe(self):
        cl = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(cl).decode('utf-8'))
        path = body.get('path', '').strip()
        lang = body.get('language', 'auto') or 'auto'
        speed = body.get('speed', 'fast')
        beam_size = 1 if speed == 'fast' else 5
        if not path or not os.path.isfile(path):
            self._err('File not found'); return
        self._start_sse(); sse = self._sse
        try: transcribe_audio(path, sse, lang, beam_size)
        except Exception as e: sse('error', {'text': str(e)[:300]})

    def _extract_ffmpeg_cmd(self, path, fmt, out):
        """Build FFmpeg cmd with GPU decode if available."""
        hw = []
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                hw = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
        except: pass
        return [FFMPEG, '-y', *hw, '-i', path, '-vn',
                '-c:a', 'pcm_s16le' if fmt=='wav' else 'libmp3lame',
                *([] if fmt=='wav' else ['-b:a','192k']), '-rf64','auto','-threads','0', out]

    def _handle_extract(self):
        cl = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(cl).decode('utf-8'))
        path = body.get('path', '').strip(); fmt = body.get('format', 'mp3')
        if not path or not os.path.isfile(path):
            self._err('File not found'); return
        d = os.path.dirname(path) or '.'; s2 = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(d, f'{s2}_audio.{fmt}')
        self._start_sse(); sse = self._sse; sse('log', {'text': f'$ ffmpeg -i {path} ...\n'})
        try:
            cmd = self._extract_ffmpeg_cmd(path, fmt, out)
            proc = subprocess.Popen(cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
            for l in proc.stdout: sse('log', {'text': l})
            proc.wait()
            if proc.returncode == 0: sse('done', {'text': f'Saved: {out}'})
            else: sse('error', {'text': f'FFmpeg error {proc.returncode}'})
        except Exception as e: sse('error', {'text': str(e)[:200]})

    def _handle_upload(self, mode):
        """Generic multipart upload handler: saves file, then calls transcribe or extract."""
        cl = int(self.headers.get('Content-Length', 0))
        ct = self.headers.get('Content-Type', '')
        boundary = None
        if 'boundary=' in ct:
            boundary = ct.split('boundary=')[1].strip()
            if boundary.startswith('"') and boundary.endswith('"'): boundary = boundary[1:-1]
        body = self.rfile.read(cl)
        file_data = None; file_name = 'input.bin'; extra = {}
        if boundary:
            parts = body.split(('--'+boundary).encode())
            for part in parts:
                hp = part.split(b'\r\n\r\n', 1)
                if len(hp) < 2: continue
                hd, bd = hp[0], hp[1]
                if bd.endswith(b'\r\n'): bd = bd[:-2]
                disp = hd.decode('utf-8', errors='replace')
                if 'filename=' in disp:
                    m = re.search(r'filename="([^"]*)"', disp)
                    if m: file_name = m.group(1); file_data = bd
                elif 'name="format"' in disp:
                    extra['format'] = bd.decode('utf-8', errors='replace').strip()
                elif 'name="language"' in disp:
                    extra['language'] = bd.decode('utf-8', errors='replace').strip()
        if not file_data:
            self._err('No file uploaded'); return
        path = os.path.join(TEMP_DIR, f'{uuid.uuid4().hex}_{file_name}')
        with open(path, 'wb') as f: f.write(file_data)
        lang = extra.get('language', 'auto') or 'auto'
        beam_size = 1 if extra.get('speed', 'fast') == 'fast' else 5
        self._start_sse(); sse = self._sse
        try:
            if mode == 'transcribe':
                transcribe_audio(path, sse, lang, beam_size)
            else:
                fmt = extra.get('format', 'mp3')
                d = os.path.dirname(path) or '.'; s2 = os.path.splitext(os.path.basename(path))[0]
                out = os.path.join(d, f'{s2}_audio.{fmt}')
                sse('log', {'text': f'Uploaded: {file_name}\n'})
                cmd = self._extract_ffmpeg_cmd(path, fmt, out)
                proc = subprocess.Popen(cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
                for l in proc.stdout: sse('log', {'text': l})
                proc.wait()
                if proc.returncode == 0: sse('done', {'text': f'Saved: {out}'})
                else: sse('error', {'text': f'FFmpeg error {proc.returncode}'})
        except Exception as e: sse('error', {'text': str(e)[:300]})

    def _handle_pipeline(self):
        cl = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(cl).decode('utf-8'))
        files = body.get('files', [])
        output_dir_override = body.get('output_dir', '')
        fmt = body.get('format', 'mp3')
        lang = body.get('language', 'auto') or None
        self._start_sse(); sse = self._sse
        total = len(files)
        for fi, file_path in enumerate(files):
            if not file_path or not os.path.isfile(file_path):
                sse('file_error', {'file': os.path.basename(file_path or ''), 'file_index': fi, 'total': total, 'error': 'File not found'})
                continue
            # Use source file's directory unless output_dir is explicitly provided
            src_dir = os.path.dirname(os.path.abspath(file_path)) or '.'
            output_dir = output_dir_override if output_dir_override else src_dir
            os.makedirs(output_dir, exist_ok=True)
            name = os.path.splitext(os.path.basename(file_path))[0]
            audio_path = os.path.join(output_dir, f'{name}_audio.{fmt}')
            srt_path = os.path.join(output_dir, f'{name}.srt')
            srt_zh_path = os.path.join(output_dir, f'{name}_zh.srt')
            sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'extract', 'percent': 0, 'step_n': 1, 'step_total': 3})
            try:
                sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'extract', 'percent': 10, 'step_n': 1, 'step_total': 3, 'substep': 'Extracting audio via FFmpeg...'})
                subprocess.run([FFMPEG, '-y', '-i', file_path, '-vn', '-c:a', 'pcm_s16le' if fmt == 'wav' else 'libmp3lame', *([] if fmt == 'wav' else ['-b:a', '192k']), '-threads', '0', audio_path], capture_output=True, check=True)
                sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'extract', 'percent': 100, 'step_n': 1, 'step_total': 3})
            except Exception as e:
                sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'extract', 'percent': -1, 'step_n': 1, 'step_total': 3, 'error': str(e)[:200]})
                continue
            sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'transcribe', 'percent': 0, 'step_n': 2, 'step_total': 3})
            try:
                srt_text = ""
                t0 = time.time()
                seg_pattern = os.path.join(output_dir, f'{name}_seg_%03d.mp3')
                # Clean up old segment files from previous runs
                for oldf in glob.glob(os.path.join(output_dir, f'{name}_seg_*.mp3')):
                    try: os.remove(oldf)
                    except: pass
                sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'transcribe', 'percent': 5, 'step_n': 2, 'step_total': 3, 'substep': 'Splitting audio into 60s segments...'})
                subprocess.run([FFMPEG, '-y', '-i', audio_path, '-f', 'segment', '-segment_time', '60',
                                '-c:a', 'libmp3lame', '-b:a', '64k', '-reset_timestamps', '1', seg_pattern],
                               capture_output=True, check=True)
                seg_files = sorted(glob.glob(os.path.join(output_dir, f'{name}_seg_*.mp3')))
                sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'transcribe', 'percent': 10, 'step_n': 2, 'step_total': 3, 'substep': f'{len(seg_files)} segments created, starting transcription...'})
                model = get_whisper()
                all_segs = []
                if seg_files:
                    import ctranslate2
                    _is_cu = ctranslate2.get_cuda_device_count() > 0
                    def transcribe_seg(idx_sp):
                        idx, sp = idx_sp
                        segs_iter, info = model.transcribe(sp, beam_size=1, vad_filter=True, language=lang)
                        local = []
                        offset = idx * 60.0
                        for sg in segs_iter:
                            sg.start += offset; sg.end += offset
                            local.append(sg)
                        return local, idx
                    if _is_cu:
                        for i, sp in enumerate(seg_files):
                            local, _ = transcribe_seg((i, sp))
                            all_segs.extend(local)
                            pct = int(((i + 1) / len(seg_files)) * 90)
                            sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'transcribe', 'percent': pct, 'step_n': 2, 'step_total': 3})
                    else:
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        MAX_W = min(2, len(seg_files))
                        done = 0; lock = threading.Lock()
                        with ThreadPoolExecutor(max_workers=MAX_W) as pool:
                            fut_map = {pool.submit(transcribe_seg, (i, p)): i for i, p in enumerate(seg_files)}
                            for fut in as_completed(fut_map):
                                segs_local, seg_idx = fut.result()
                                with lock:
                                    all_segs.extend(segs_local)
                                    done += 1
                                    pct = 10 + int((done / len(seg_files)) * 85)
                                    sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'transcribe', 'percent': pct, 'step_n': 2, 'step_total': 3})
                    all_segs.sort(key=lambda s: s.start)
                    for sp in seg_files:
                        try: os.remove(sp)
                        except: pass
                sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'transcribe', 'percent': 95, 'step_n': 2, 'step_total': 3, 'substep': 'Writing SRT file...'})
                srt_text = to_srt(all_segs)
                with open(srt_path, 'w', encoding='utf-8') as f: f.write(srt_text)
                sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'transcribe', 'percent': 100, 'step_n': 2, 'step_total': 3})
            except Exception as e:
                sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'transcribe', 'percent': -1, 'step_n': 2, 'step_total': 3, 'error': str(e)[:200]})
                continue
            sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'translate', 'percent': 0, 'step_n': 3, 'step_total': 3})
            try:
                from deep_translator import GoogleTranslator
                blocks = srt_text.strip().split('\n\n')
                translated_blocks = []
                for bi, block in enumerate(blocks):
                    blines = block.split('\n')
                    new_blines = []
                    for li, bline in enumerate(blines):
                        if li == 0 or '-->' in bline or bline.strip() == '':
                            new_blines.append(bline)
                        else:
                            try:
                                translated = GoogleTranslator(source='auto', target='zh-CN').translate(bline)
                                new_blines.append(translated)
                            except:
                                new_blines.append(bline)
                    translated_blocks.append('\n'.join(new_blines))
                    pct = int((bi / max(1, len(blocks))) * 95)
                    sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'translate', 'percent': pct, 'step_n': 3, 'step_total': 3})
                zh_text = '\n\n'.join(translated_blocks)
                with open(srt_zh_path, 'w', encoding='utf-8') as f: f.write(zh_text)
                sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'translate', 'percent': 100, 'step_n': 3, 'step_total': 3})
            except Exception as e:
                sse('file_progress', {'file': name, 'file_index': fi, 'total': total, 'step': 'translate', 'percent': -1, 'step_n': 3, 'step_total': 3, 'error': str(e)[:200]})
                continue
            sse('file_done', {'file': name, 'file_index': fi, 'total': total, 'audio': audio_path, 'srt': srt_path, 'srt_zh': srt_zh_path})
        sse('all_done', {'total': total})

    def _handle_translate(self):
        cl = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(cl).decode('utf-8'))
        srt_text = body.get('srt', '')
        target = body.get('target', 'zh-CN')
        if not srt_text.strip():
            self._err('Empty SRT'); return
        try:
            from deep_translator import GoogleTranslator
            blocks = srt_text.strip().split('\n\n')
            translated_blocks = []
            for block in blocks:
                lines_b = block.split('\n')
                new_lines = []
                for i, line in enumerate(lines_b):
                    if i == 0 or '-->' in line or line.strip() == '':
                        new_lines.append(line)
                    else:
                        try:
                            translated = GoogleTranslator(source='auto', target=target).translate(line)
                            new_lines.append(translated)
                        except:
                            new_lines.append(line)
                translated_blocks.append('\n'.join(new_lines))
            result = '\n\n'.join(translated_blocks)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'srt': result, 'original': srt_text}, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            self._err(str(e))

    def _err(self, msg):
        self.send_response(500)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(msg.encode('utf-8'))

    def log_message(self, fmt, *args):
        from datetime import datetime
        ts = time.strftime("%H:%M:%S")
        msg = fmt % args if args else fmt
        print(f'[{ts}] {self.client_address[0]} {msg}', flush=True)


# ==================== Main ====================
def main():
    print('Loading Whisper Turbo...')
    try: get_whisper(); print('Model loaded.')
    except Exception as e: print(f'Warning: {e}')
    srv = HTTPServer((HOST, PORT), Handler)
    print(f'\n  VoiceCut  http://{HOST}:{PORT}\n')
    try: srv.serve_forever()
    except KeyboardInterrupt: srv.server_close()

if __name__ == '__main__':
    main()
