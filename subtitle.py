#!/usr/bin/env python3
"""VoxSub Subtitle — 长音频转字幕 (faster-whisper + FFmpeg)"""

import os, sys, json, re, tempfile, struct, threading, subprocess
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler

HOST = '127.0.0.1'
PORT = 8765
MODEL_PATH = r'D:\Develop\Workspace\voicecut\models\whisper-turbo-ct2'
FFMPEG = r'D:\Develop\ffmpeg\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe'
# Processing happens in the source file directory, no global temp dir needed
CHUNK_SEC = 60


# ==================== Model ====================
_model = None
_model_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel(MODEL_PATH, device='cpu', compute_type='int8', cpu_threads=4, num_workers=1)
    return _model


# ==================== SRT ====================
def fmt_srt(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

def segments_to_srt(segments):
    lines = []
    for i, seg in enumerate(segments, 1):
        text = seg.text.strip()
        if text:
            lines.append(str(i))
            lines.append(f"{fmt_srt(seg.start)} --> {fmt_srt(seg.end)}")
            lines.append(text)
            lines.append('')
    return '\n'.join(lines)


# ==================== SRT Normalize & Convert ====================
def normalize_srt(text):
    """Normalize SRT: remove empty cues, fix spacing, clean formatting."""
    if not text.strip():
        return ''
    blocks = re.split(r'\n\s*\n', text.strip())
    cleaned = []
    for blk in blocks:
        ln = [l.strip() for l in blk.strip().split('\n') if l.strip()]
        if len(ln) < 3:
            continue
        idx_line = ln[0]
        ts_line = ln[1]
        txt_lines = ln[2:]
        if '-->' not in ts_line or not idx_line.isdigit():
            continue
        txt = ' '.join(txt_lines).strip()
        if not txt:
            continue
        ts_line = re.sub(r'\s+', ' ', ts_line)
        cleaned.append(f'{idx_line}\n{ts_line}\n{txt}')
    return '\n\n'.join(cleaned) + '\n'


def srt_to_vtt(srt_text):
    """Convert SRT to WebVTT format."""
    normalized = normalize_srt(srt_text)
    if not normalized.strip():
        return 'WEBVTT\n\n'
    vtt_lines = ['WEBVTT']
    for blk in re.split(r'\n\s*\n', normalized.strip()):
        ln = blk.strip().split('\n')
        if len(ln) < 2:
            continue
        ts_line = ln[1] if len(ln) > 1 and '-->' in ln[1] else ln[0]
        ts_line = ts_line.replace(',', '.')
        txt = '\n'.join(ln[2:]) if len(ln) > 2 else ''
        vtt_lines.append('')
        vtt_lines.append(ts_line)
        if txt:
            vtt_lines.append(txt)
    return '\n'.join(vtt_lines) + '\n'


# ==================== FFmpeg helpers ====================
def convert_to_wav(path, output_dir):
    """Convert to 16kHz mono WAV in output_dir."""
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(output_dir, f'{stem}_temp.wav')
    # Clean up previous temp WAVs for this stem
    for oldf in glob.glob(os.path.join(output_dir, f'{stem}_temp*.wav')):
        try: os.remove(oldf)
        except: pass
    subprocess.run([FFMPEG, '-y', '-i', path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', out],
                   capture_output=True, check=True)
    return out

def split_audio(wav_path, chunk_sec=CHUNK_SEC):
    """Split WAV into chunks alongside the WAV file."""
    import re
    out_dir = os.path.dirname(os.path.abspath(wav_path)) or '.'
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    # Clean up any previous chunks for this stem
    for oldf in glob.glob(os.path.join(out_dir, f'{stem}_chunk_*.wav')):
        try: os.remove(oldf)
        except: pass

    dur_str = subprocess.run([FFMPEG, '-i', wav_path, '-f', 'null', '-'],
                              capture_output=True, text=True).stderr
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.?\d*)', dur_str)
    if not m:
        raise RuntimeError('Could not parse audio duration from FFmpeg output')
    total_sec = float(m.group(1)) * 3600 + float(m.group(2)) * 60 + float(m.group(3))

    chunks = []
    for start in range(0, int(total_sec), chunk_sec):
        end = min(start + chunk_sec, total_sec)
        out = os.path.join(out_dir, f'{stem}_chunk_{start // chunk_sec:03d}.wav')
        dur = end - start
        subprocess.run([FFMPEG, '-y', '-ss', str(start), '-i', wav_path, '-t', str(dur),
                        '-c', 'copy' if start > 0 else 'pcm_s16le', out],
                       capture_output=True)
        chunks.append((out, start, dur))
    return chunks, total_sec


# ==================== Transcription ====================
def transcribe_with_progress(audio_path, sse_func):
    model = get_model()
    import re

    # Use source file's directory for all intermediate files
    src_dir = os.path.dirname(os.path.abspath(audio_path)) or '.'
    sse_func('log', {'text': 'Converting audio to 16kHz WAV (in source dir)...\n'})
    wav = convert_to_wav(audio_path, src_dir)
    sse_func('progress', {'percent': 2, 'status': 'WAV conversion done'})

    dur_str = subprocess.run([FFMPEG, '-i', wav, '-f', 'null', '-'],
                              capture_output=True, text=True).stderr
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.?\d*)', dur_str)
    if m:
        total = float(m.group(1)) * 3600 + float(m.group(2)) * 60 + float(m.group(3))
    else:
        total = 0
    sse_func('log', {'text': f'Audio duration: {total:.0f}s\n'})

    if total <= CHUNK_SEC * 2:
        sse_func('log', {'text': 'Short file, transcribing directly...\n'})
        segments, info = model.transcribe(wav, beam_size=5, vad_filter=True, language='zh')
        results = list(segments)
        # Clean up temp WAV
        try: os.remove(wav)
        except: pass
        srt = segments_to_srt(results)
        sse_func('result', {'srt': srt, 'count': len(results)})
        return

    sse_func('log', {'text': f'Splitting into {CHUNK_SEC}s chunks...\n'})
    chunks, _ = split_audio(wav)
    total_chunks = len(chunks)
    sse_func('log', {'text': f'{total_chunks} chunks in {src_dir}\n'})

    all_segments = []
    for idx, (chunk_path, offset, _) in enumerate(chunks):
        pct = int((idx / total_chunks) * 95)
        sse_func('progress', {'percent': pct, 'status': f'Transcribing chunk {idx+1}/{total_chunks}'})
        sse_func('log', {'text': f'[{idx+1}/{total_chunks}] Transcribing...\n'})

        segments, _ = model.transcribe(chunk_path, beam_size=5, vad_filter=True, language='zh')
        for seg in segments:
            seg.start += offset
            seg.end += offset
            all_segments.append(seg)

        try: os.remove(chunk_path)
        except: pass

    # Clean up temp WAV
    try: os.remove(wav)
    except: pass

    all_segments.sort(key=lambda s: s.start)
    srt = segments_to_srt(all_segments)
    sse_func('result', {'srt': srt, 'count': len(all_segments)})
# ==================== HTML ====================
HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VoxSub — 音频转字幕</title>
<script src="https://unpkg.com/lucide@latest"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0c0c16;--surface:#16162a;--surface-hover:#1e1e3a;--border:#2a2a48;--text:#e2e2ee;--text-dim:#7a7a9a;--accent:#5b8def;--accent-hover:#7aa5ff;--success:#3dce7a;--radius:6px}
html,body{height:100%}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);display:flex;justify-content:center;align-items:center}
.app{width:min(820px,95vw);height:min(720px,90vh);background:var(--surface);border-radius:12px;border:1px solid var(--border);display:flex;flex-direction:column;padding:20px 24px 16px;gap:10px;box-shadow:0 8px 48px rgba(0,0,0,.4)}
h1{font-size:16px;font-weight:500;display:flex;align-items:center;gap:8px;flex-shrink:0}
h1 svg{width:20px;height:20px;stroke:var(--accent)}
.desc{font-size:13px;color:var(--text-dim);flex-shrink:0}
.row{display:flex;gap:8px;flex-shrink:0}
.row input{flex:1;padding:8px 10px;border:1px solid var(--border);border-radius:6px;background:#0c0c18;color:var(--text);font-size:13px;outline:none}
.row input:focus{border-color:var(--accent)}
.row button{padding:8px 12px;border:1px solid var(--border);border-radius:6px;background:transparent;color:var(--text);font-size:13px;cursor:pointer;display:flex;align-items:center;gap:4px}
.row button:hover{background:var(--surface-hover);border-color:var(--accent)}
.row button svg{width:16px;height:16px;stroke:var(--text)}
.bar{height:4px;background:var(--border);border-radius:2px;overflow:hidden;flex-shrink:0}
.bar-fill{height:100%;width:0;background:var(--accent);border-radius:2px;transition:width .2s}
.status{font-size:12px;color:var(--text-dim);flex-shrink:0}
.status.done{color:var(--success)}
.status.err{color:#ef6b3a}
.action-row{display:flex;gap:8px;flex-shrink:0}
.action-row button{display:inline-flex;align-items:center;gap:6px;padding:8px 14px;border-radius:6px;font-size:13px;cursor:pointer;border:none;background:var(--accent);color:#fff}
.action-row button:hover{background:var(--accent-hover)}
.action-row button:disabled{opacity:.4;cursor:not-allowed}
.action-row button svg{width:16px;height:16px;stroke:#fff}
.action-row .spacer{flex:1}
.action-row .sec{background:transparent;border:1px solid var(--border);color:var(--text)}
.action-row .sec:hover{background:var(--surface-hover);border-color:var(--accent)}
.action-row .sec svg{stroke:var(--text)}
.result-area{flex:1;min-height:0;display:flex;flex-direction:column;gap:4px}
.result-area label{font-size:11px;color:var(--text-dim);display:flex;align-items:center;gap:4px;flex-shrink:0}
.result-area label svg{width:13px;height:13px;stroke:var(--text-dim)}
.result-area textarea{flex:1;background:#0c0c18;border:1px solid var(--border);border-radius:6px;color:var(--text);font-family:'Cascadia Code','JetBrains Mono','Consolas',monospace;font-size:13px;padding:10px;resize:none;outline:none;width:100%;min-height:60px}
.result-area textarea::placeholder{color:var(--text-dim);opacity:.3}
.browse-btn{display:inline-flex;align-items:center;gap:4px;padding:8px 12px;border:1px solid var(--border);border-radius:6px;background:transparent;color:var(--text);font-size:13px;cursor:pointer;transition:background .15s,border-color .15s}
.browse-btn:hover{background:var(--surface-hover);border-color:var(--accent)}
.browse-btn svg{width:16px;height:16px;stroke:var(--text)}
@media(max-width:600px){.app{padding:14px;height:100vh;width:100vw;border-radius:0}}
</style>
</head>
<body>
<div class="app">
  <h1><svg data-lucide="subtitles" stroke-width="1.5"></svg>音频转字幕 <span style="font-size:12px;font-weight:400;color:var(--text-dim)">Whisper Turbo</span></h1>
  <p class="desc">选择音频或视频文件，Whisper Turbo 将自动转录并生成 SRT 字幕。</p>

  <div class="row">
    <input type="text" id="pathInput" placeholder="点击 Browse 选择，或粘贴路径">
    <button id="btnBrowse"><svg data-lucide="folder-open" stroke-width="1.5"></svg> Browse</button>
    <input type="file" id="fileInput" accept="audio/*,video/*" style="position:absolute;opacity:0;width:0;height:0">
  </div>

  <div class="bar"><div class="bar-fill" id="barFill"></div></div>
  <div class="status" id="statusMsg">就绪</div>

  <div class="result-area">
    <label><svg data-lucide="file-text" stroke-width="1.5"></svg>字幕预览 (SRT 格式)</label>
    <textarea id="srtOutput" readonly placeholder="转录完成后字幕将显示在此处…" spellcheck="false"></textarea>
  </div>

  <div class="action-row">
    <button id="btnStart"><svg data-lucide="wand" stroke-width="1.5"></svg>开始转录</button>
    <div class="spacer"></div>
    <button class="sec" id="btn复制" disabled><svg data-lucide="copy" stroke-width="1.5"></svg>复制</button>
    <button class="sec" id="btnNorm" disabled><svg data-lucide="align-left" stroke-width="1.5"></svg>标准化</button>
    <button class="sec" id="btnDlVtt" disabled><svg data-lucide="file-down" stroke-width="1.5"></svg>下载 VTT</button>
    <button class="sec" id="btnDl" disabled><svg data-lucide="download" stroke-width="1.5"></svg>下载 SRT</button>
  </div>
</div>

<script>
const $=id=>document.getElementById(id)
const p=$("pathInput"),b=$("btnBrowse"),s=$("btnStart"),m=$("statusMsg"),bf=$("barFill"),sr=$("srtOutput"),cp=$("btn复制"),dl=$("btnDl"),nn=$("btnNorm"),dv=$("btnDlVtt")

function st(msg,cls,pct){m.textContent=msg;m.className="status"+(cls?" "+cls:"");if(pct!=null)bf.style.width=Math.min(100,pct)+"%"}
function log(t){sr.value+=t;sr.scrollTop=sr.scrollHeight}

b.addEventListener("click",()=>{document.getElementById("fileInput").click()})$("fileInput").addEventListener("change",()=>{
  const f=$("fileInput").files[0]
  if(f){
    p.value=f.path||f.name
    st("???: "+f.name,"",0)
  }
})
document.addEventListener("dragover",e=>e.preventDefault())
document.addEventListener("drop",e=>{e.preventDefault();const f=e.dataTransfer.files[0];if(f&&f.path)p.value=f.path})

s.addEventListener("click",async()=>{
  const path=p.value.trim();if(!path){st("请先选择文件","err");return}
  s.disabled=true;cp.disabled=true;dl.disabled=true;nn.disabled=true;dv.disabled=true;sr.value="";st("正在处理...","",0)
  try{
    const r=await fetch("/transcribe",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({path})})
    if(!r.ok){const t=await r.text();throw new Error(t)}
    const reader=r.body.getReader();const dec=new TextDecoder();let buf=""
    while(true){
      const{done,value}=await reader.read();if(done)break
      buf+=dec.decode(value,{stream:true})
      const parts=buf.split("\n\n");buf=parts.pop()||""
      for(const ev of parts){
        let type="",data=""
        for(const ln of ev.split("\n")){
          if(ln.startsWith("event: "))type=ln.slice(7)
          else if(ln.startsWith("data: "))data=ln.slice(6)
        }
        if(!data)continue
        try{
          const jd=JSON.parse(data)
          if(type==="log"){log(jd.text)}
          else if(type==="progress"){st(jd.status,"",jd.percent)}
          else if(type==="result"){
            sr.value=jd.srt;st("完成! "+jd.count+" 段","done",100)
            cp.disabled=false;dl.disabled=false;nn.disabled=false;dv.disabled=false
          }
          else if(type==="error"){st("错误: "+jd.text,"err");log("[Error] "+jd.text+"\n")}
        }catch(e){}
      }
    }
  }catch(e){st("错误: "+e.message,"err")}
  s.disabled=false
})

cp.addEventListener("click",()=>{sr.select();navigator.clipboard.writeText(sr.value).catch(()=>{})})
dl.addEventListener("click",()=>{
  const stem=p.value.replace(/^.*[/\\\\]/,"").replace(/[.][^.]+$/,"")||"字幕"
  const blob=new Blob([sr.value],{type:"text/plain;charset=utf-8"})
  const url=URL.createObjectURL(blob);const a=document.createElement("a")
  a.href=url;a.download=stem+".srt";document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url)
})
lucide.createIcons()
// Normalize SRT via server
nn.addEventListener("click",async()=>{
  if(!sr.value.trim()) return
  nn.disabled=true;st("标准化中...","",0)
  try{
    const r=await fetch("/convert",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({srt:sr.value,format:"normalized"})})
    if(!r.ok){const t=await r.text();throw new Error(t)}
    sr.value=await r.text();st("标准化完成","done",100)
  }catch(e){st("标准化失败: "+e.message,"err")}
  nn.disabled=false
})

// SRT→VTT client-side conversion + download
dv.addEventListener("click",()=>{
  const stem=p.value.replace(/^.*[/\\]/,"").replace(/[.][^.]+$/,"")||"字幕"
  const lines=sr.value.split("\n");const out=["WEBVTT",""]
  let i=0
  while(i<lines.length){
    const l=lines[i].trim()
    if(l===""||/^\d+$/.test(l)){i++;continue}
    if(l.includes("-->")){
      out.push(l.replace(/,/g,"."))
      i++;let txt=[]
      while(i<lines.length&&lines[i].trim()!==""&&!/^\d+$/.test(lines[i].trim())&&!lines[i].includes("-->")){
        txt.push(lines[i]);i++
      }
      if(txt.length)out.push(txt.join("\n"))
      out.push("")
    }else{i++}
  }
  const vtt=out.join("\n")
  const blob=new Blob([vtt],{type:"text/vtt;charset=utf-8"})
  const url=URL.createObjectURL(blob);const a=document.createElement("a")
  a.href=url;a.download=stem+".vtt";document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url)
})
// Normalize SRT via server
nn.addEventListener("click",async()=>{
  if(!sr.value.trim()) return
  nn.disabled=true;st("标准化中...","",0)
  try{
    const r=await fetch("/convert",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({srt:sr.value,format:"normalized"})})
    if(!r.ok){const t=await r.text();throw new Error(t)}
    sr.value=await r.text();st("标准化完成","done",100)
  }catch(e){st("标准化失败: "+e.message,"err")}
  nn.disabled=false
})

// SRT→VTT client-side conversion + download
dv.addEventListener("click",()=>{
  const stem=p.value.replace(/^.*[/\\]/,"").replace(/[.][^.]+$/,"")||"字幕"
  const lines=sr.value.split("\n");const out=["WEBVTT",""]
  let i=0
  while(i<lines.length){
    const l=lines[i].trim()
    if(l===""||/^\d+$/.test(l)){i++;continue}
    if(l.includes("-->")){
      out.push(l.replace(/,/g,"."))
      i++;let txt=[]
      while(i<lines.length&&lines[i].trim()!==""&&!/^\d+$/.test(lines[i].trim())&&!lines[i].includes("-->")){
        txt.push(lines[i]);i++
      }
      if(txt.length)out.push(txt.join("\n"))
      out.push("")
    }else{i++}
  }
  const vtt=out.join("\n")
  const blob=new Blob([vtt],{type:"text/vtt;charset=utf-8"})
  const url=URL.createObjectURL(blob);const a=document.createElement("a")
  a.href=url;a.download=stem+".vtt";document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url)
})
</script>
</body>
</html>'''


# ==================== HTTP ====================
class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        p = urlparse(self.path).path
        if p == '/': self._html()
        else: self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path == '/transcribe': self._transcribe()
        elif urlparse(self.path).path == '/convert': self._convert()
        else: self.send_error(404)

    def _html(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML.encode('utf-8'))

    def _transcribe(self):
        cl = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(cl).decode('utf-8'))
        audio_path = body.get('path', '').strip()

        if not audio_path or not os.path.isfile(audio_path):
            self._sse_error('File not found')
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        def sse(type_, data):
            msg = f'event: {type_}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'
            try:
                self.wfile.write(msg.encode('utf-8'))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        try:
            transcribe_with_progress(audio_path, sse)
        except Exception as e:
            sse('error', {'text': str(e)[:300]})

    def _convert(self):
        cl = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(cl).decode('utf-8'))
        srt_text = body.get('srt', '').strip()
        to_format = body.get('format', 'vtt').strip().lower()

        if not srt_text:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'No SRT content provided'}).encode('utf-8'))
            return

        self.send_response(200)
        ct = 'text/vtt; charset=utf-8' if to_format == 'vtt' else 'text/plain; charset=utf-8'
        self.send_header('Content-Type', ct)
        self.send_header('Content-Disposition', 'attachment' if to_format == 'vtt' else 'inline')
        self.end_headers()

        try:
            if to_format == 'vtt':
                result = srt_to_vtt(srt_text)
            elif to_format == 'normalized':
                result = normalize_srt(srt_text)
            else:
                result = srt_text
            self.wfile.write(result.encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))


    def _sse_error(self, msg):
        self.send_response(500)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.end_headers()
        self.wfile.write(f'event: error\ndata: {json.dumps({"text": msg})}\n\n'.encode('utf-8'))

    def log_message(self, fmt, *args):
        print(f'  {args[0]} {args[1]} {args[2]}')


def main():
    try:
        # Preload model on startup
        print('Loading Whisper Turbo...')
        get_model()
        print('Model loaded.')
    except Exception as e:
        print(f'Model load failed: {e}')
    srv = HTTPServer((HOST, PORT), Handler)
    print(f'\n  VoxSub Subtitle Server')
    print(f'  http://{HOST}:{PORT}')
    print(f'  Ctrl+C to stop\n')
    try: srv.serve_forever()
    except KeyboardInterrupt: srv.server_close()


if __name__ == '__main__':
    main()



