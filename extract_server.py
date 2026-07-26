#!/usr/bin/env python3
"""VoxSub Extract — 本地视频提取音频 (FFmpeg)"""

import os, sys, json, re, uuid, tempfile, struct, threading, subprocess, shutil
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler

HOST = '127.0.0.1'
PORT = 8767
FFMPEG = r'D:\Develop\ffmpeg\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe'
TEMP_DIR = os.path.join(tempfile.gettempdir(), 'voicecut_extract')
os.makedirs(TEMP_DIR, exist_ok=True)


# ==================== Audio extraction ====================
def extract_audio(video_path, fmt, output_path):
    if not os.path.isfile(FFMPEG):
        raise FileNotFoundError(f'FFmpeg not found: {FFMPEG}')
    if fmt == 'wav':
        cmd = [FFMPEG, '-y', '-i', video_path, '-vn', '-c:a', 'pcm_s16le', '-rf64', 'auto', '-threads', '0', output_path]
    else:
        cmd = [FFMPEG, '-y', '-i', video_path, '-vn', '-c:a', 'libmp3lame', '-b:a', '192k', '-threads', '0', output_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    lines = []
    for line in proc.stdout:
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        err = ''.join(lines[-10:]).strip() or 'FFmpeg failed'
        if 'Invalid data found' in err:
            raise ValueError('Not a valid video file')
        raise RuntimeError(err[:400])
    return ''.join(lines), os.path.getsize(output_path)


def _browse_file_dialog():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title='Select a video file', filetypes=[('Video files', '*.mp4 *.mov *.avi *.mkv *.webm'), ('All files', '*.*')])
    root.destroy()
    return path


# ==================== HTML ====================
HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VoxSub - 视频转音频</title>
<script src="https://unpkg.com/lucide@latest"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0c0c16;--surface:#16162a;--surface-hover:#1e1e3a;--border:#2a2a48;--text:#e2e2ee;--text-dim:#7a7a9a;--accent:#5b8def;--accent-hover:#7aa5ff;--radius:6px}
html,body{height:100%}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);display:flex;justify-content:center;align-items:center}
.app{width:min(740px,95vw);height:min(700px,90vh);background:var(--surface);border-radius:12px;border:1px solid var(--border);display:flex;flex-direction:column;padding:20px 24px 16px;gap:10px;box-shadow:0 8px 48px rgba(0,0,0,.4)}
h1{font-size:16px;font-weight:500;display:flex;align-items:center;gap:8px;flex-shrink:0}
h1 svg{width:20px;height:20px;stroke:var(--accent)}
.desc{font-size:13px;color:var(--text-dim);flex-shrink:0;line-height:1.5}
.row{display:flex;gap:8px;flex-shrink:0}
.row input{flex:1;padding:8px 10px;border:1px solid var(--border);border-radius:6px;background:#0c0c18;color:var(--text);font-size:13px;outline:none}
.row input:focus{border-color:var(--accent)}
.row input::placeholder{color:var(--text-dim);opacity:.4}
.row button{padding:8px 12px;border:1px solid var(--border);border-radius:6px;background:transparent;color:var(--text);font-size:13px;cursor:pointer;display:flex;align-items:center;gap:4px}
.row button:hover{background:var(--surface-hover);border-color:var(--accent)}
.row button svg{width:16px;height:16px;stroke:var(--text)}
.fmt-row{display:flex;gap:12px;align-items:center;flex-shrink:0}
.fmt-row label{font-size:12px;color:var(--text-dim)}
.fmt-row select{padding:8px 10px;border:1px solid var(--border);border-radius:6px;background:transparent;color:var(--text);font-size:13px;cursor:pointer}
.fmt-row select:hover{border-color:var(--accent)}

.extract-btn{display:inline-flex;align-items:center;gap:6px;padding:8px 14px;border-radius:4px;font-size:13px;cursor:pointer;border:none;background:var(--accent);color:#fff}
.extract-btn:hover{background:var(--accent-hover)}
.extract-btn:disabled{opacity:.4;cursor:not-allowed}
.extract-btn svg{width:16px;height:16px;stroke:#fff}
.extract-btn.processing svg{animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

.bar{height:4px;background:var(--border);border-radius:2px;overflow:hidden;flex-shrink:0}
.bar-fill{height:100%;width:0;background:var(--accent);border-radius:2px;transition:width .2s}
.status{font-size:12px;color:var(--text-dim);flex-shrink:0}
.status.done{color:#3dce7a}
.status.err{color:#ef6b3a}
.log-area{flex:1;min-height:0;display:flex;flex-direction:column;gap:4px}
.log-area label{font-size:11px;color:var(--text-dim);display:flex;align-items:center;gap:4px;flex-shrink:0}
.log-area label svg{width:13px;height:13px;stroke:var(--text-dim)}
.log-area textarea{flex:1;background:#080812;border:1px solid var(--border);border-radius:6px;color:#b0b0c8;font-family:'Cascadia Code','JetBrains Mono','Consolas',monospace;font-size:12px;padding:10px;resize:none;outline:none;width:100%;min-height:60px}
.log-area textarea::placeholder{color:var(--text-dim);opacity:.3}
@media(max-width:600px){.app{padding:14px;height:100vh;width:100vw;border-radius:0}}
</style>
</head>
<body>
<div class="app">
  <h1><svg data-lucide="video" stroke-width="1.5"></svg>视频 	o 音频  <span style="font-size:12px;font-weight:400;color:var(--text-dim)">FFmpeg</span></h1>
  <p class="desc">选择视频文件，提取的音频将自动保存到源文件所在目录。</p>

  <div class="row">
    <input type="text" id="pathInput" placeholder="点击 Browse 选择，或粘贴路径">
    <button id="btnBrowse"><svg data-lucide="folder-open" stroke-width="1.5"></svg> Browse</button>
    <input type="file" id="fileInput" accept="video/*" style="display:none">
  </div>

  <div class="fmt-row">
    <label>输出格式</label>
    <select id="fmtSelect">
      <option value="wav">WAV (无损 PCM)</option>
      <option value="mp3" selected>MP3 (192kbps)</option>
    </select>
    <div style="flex:1"></div>
    <button class="extract-btn" id="btnStart"><svg data-lucide="wand" stroke-width="1.5"></svg><span id="btnStartText">提取音频</span></button>
  </div>

  <div class="bar"><div class="bar-fill" id="barFill"></div></div>
  <div class="status" id="statusMsg">就绪</div>

  <div class="log-area">
    <label><svg data-lucide="terminal" stroke-width="1.5"></svg>FFmpeg 命令行输出</label>
    <textarea id="logArea" readonly placeholder="点击「提取音频」后，FFmpeg 的实时输出将显示在此处…" spellcheck="false" wrap="off"></textarea>
  </div>
</div>

<script>
const $=id=>document.getElementById(id)
const p=$("pathInput"),b=$("btnBrowse"),f=$("fmtSelect"),s=$("btnStart"),m=$("statusMsg"),bf=$("barFill"),la=$("logArea")

function st(msg,cls,pct){m.textContent=msg;m.className="status"+(cls?" "+cls:"");if(pct!=null)bf.style.width=Math.min(100,pct)+"%"}
function log(t){la.value+=t;la.scrollTop=la.scrollHeight}

b.addEventListener("click",async()=>{
  try{const r=await fetch("/browse");const j=await r.json();if(j.path)p.value=j.path}catch(e){st("Browse: "+e.message,"err")}
})
document.addEventListener("dragover",e=>e.preventDefault())
document.addEventListener("drop",e=>{e.preventDefault();const f=e.dataTransfer.files[0];if(f&&f.path)p.value=f.path})

s.addEventListener("click",async()=>{
  const path=p.value.trim();if(!path){st("请先选择视频文件","err");return}
  s.disabled=true;s.classList.add("processing");
  const ic=s.querySelector("svg");
  const tx=document.getElementById("btnStartText");
  ic.setAttribute("data-lucide","loader");ic.setAttribute("stroke-width","2.5");
  tx.textContent="提取中...";lucide.createIcons();la.value="";st("正在提取...","",0)
  try{
    const fmt=f.value
    const r=await fetch("/extract-local",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({path,format:fmt})})
    if(!r.ok){const t=await r.text();throw new Error(t)}
    const reader=r.body.getReader();const dec=new TextDecoder();let buf=""
    while(true){
      const{done,value}=await reader.read();if(done)break
      buf+=dec.decode(value,{stream:true})
      const parts=buf.split("

");buf=parts.pop()||""
      for(const ev of parts){
        let type="",data=""
        for(const ln of ev.split("
")){
          if(ln.startsWith("event: "))type=ln.slice(7)
          else if(ln.startsWith("data: "))data=ln.slice(6)
        }
        if(!data)continue
        try{
          const j=JSON.parse(data)
          if(type==="log"){log(j.text)}
          else if(type==="done"){
            st("提取完成!","done",100);s.disabled=false;s.classList.remove("processing");
            ic.setAttribute("data-lucide","check-circle");ic.setAttribute("stroke-width","2");
            tx.textContent="提取完成";lucide.createIcons();
            setTimeout(()=>{ic.setAttribute("data-lucide","wand-2");tx.textContent="提取音频";lucide.createIcons()},4000);
            log("
[完成] "+j.text+"
")
          }
          else if(type==="error"){st("失败: "+j.text,"err");log("[错误] "+j.text+"
")}
        }catch(e){}
      }
    }
  }catch(e){st("连接错误: "+e.message,"err");log("[连接错误] "+e.message+"
")}
  s.disabled=false;s.classList.remove("processing");
  ic.setAttribute("data-lucide","wand-2");ic.setAttribute("stroke-width","2");
  tx.textContent="提取音频";lucide.createIcons()
})
lucide.createIcons()
</script>
</body>
</html>'''


# ==================== HTTP server ====================
class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        p = urlparse(self.path).path
        if p == '/':
            self._html()
        elif p == '/browse':
            self._browse()
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path == '/extract-local':
            self._extract_local()
        else:
            self.send_error(404)

    def _html(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML.encode('utf-8'))

    def _browse(self):
        path = _browse_file_dialog()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'path': path}).encode('utf-8'))

    def _extract_local(self):
        cl = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(cl).decode('utf-8'))
        video_path = body.get('path', '').strip()
        fmt = body.get('format', 'mp3')

        if not video_path or not os.path.isfile(video_path):
            self._error('File not found')
            return

        vid_dir = os.path.dirname(video_path) or '.'
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(vid_dir, f'{stem}_audio.{fmt}')

        if fmt == 'wav':
            cmd = [FFMPEG, '-y', '-i', video_path, '-vn', '-c:a', 'pcm_s16le', '-rf64', 'auto', '-threads', '0', out_path]
        else:
            cmd = [FFMPEG, '-y', '-i', video_path, '-vn', '-c:a', 'libmp3lame', '-b:a', '192k', '-threads', '0', out_path]

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        def sse(type_, data_dict):
            msg = f'event: {type_}\ndata: {json.dumps(data_dict, ensure_ascii=False)}\n\n'
            try:
                self.wfile.write(msg.encode('utf-8'))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        sse('log', {'text': f'$ {subprocess.list2cmdline(cmd)}\n'})

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf-8', errors='replace', bufsize=1)
            for line in proc.stdout:
                sse('log', {'text': line})
            proc.wait()

            if proc.returncode == 0:
                sse('done', {'text': f'已保存到: {out_path}'})
            else:
                sse('error', {'text': f'FFmpeg returned error code {proc.returncode}'})

        except FileNotFoundError:
            sse('error', {'text': 'FFmpeg not found'})
        except Exception as e:
            sse('error', {'text': str(e)[:300]})

    def _error(self, msg):
        self.send_response(500)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.end_headers()
        self.wfile.write(f'event: error\ndata: {json.dumps({"text": msg})}\n\n'.encode('utf-8'))

    def log_message(self, fmt, *args):
        print(f'  {args[0]} {args[1]} {args[2]}')


# ==================== Entry ====================
def main_cli():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage:\n  python extract_server.py                     start web UI')
        print('  python extract_server.py <video>             extract WAV')
        print('  python extract_server.py <video> -f mp3      extract MP3')
        return
    video = sys.argv[1]
    if not os.path.exists(video):
        print(f'File not found: {video}'); sys.exit(1)
    fmt = 'mp3' if '-f' in sys.argv and len(sys.argv) > sys.argv.index('-f')+1 and sys.argv[sys.argv.index('-f')+1]=='mp3' else 'wav'
    stem = os.path.splitext(video)[0]
    out = f'{stem}_audio.{fmt}'
    print(f'Extracting {os.path.basename(video)} -> {out} ({fmt.upper()})')
    txt, sz = extract_audio(video, fmt, out)
    print(txt)
    print(f'Done! {sz/1024/1024:.1f} MB')


def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        main_cli()
    else:
        srv = HTTPServer((HOST, PORT), Handler)
        print(f'\n  VoxSub -- local video to audio (FFmpeg)')
        print(f'  http://{HOST}:{PORT}')
        print(f'  Ctrl+C to stop\n')
        try: srv.serve_forever()
        except KeyboardInterrupt: srv.server_close()


if __name__ == '__main__':
    main()



