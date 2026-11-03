"""Small experimental local Dory chat. Requires the separately built Dory runtime."""
import argparse, json, os, pathlib, queue, socket, subprocess, tempfile, threading, time, urllib.error, urllib.request, uuid
import tkinter as tk
from tkinter import ttk

ROOT=pathlib.Path(__file__).resolve().parent
SERVER=pathlib.Path(os.environ.get('DORY_SERVER',str(ROOT/'runtime/build/bin/llama-server.exe')))
MODEL=pathlib.Path(os.environ.get('DORY_MODEL',str(ROOT/'models/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf')))
MODES={'Light (1 loop)':1,'Medium (2 loops)':2,'Intelligent (3 loops)':3,'Genius (4 loops)':4}

class Worker:
    def __init__(self,device='CUDA1',partition=(16,40,16)):
        if not SERVER.is_file() or not MODEL.is_file():raise FileNotFoundError('Set --server and --model to the patched llama-server and Dory GGUF paths; see README.md')
        if len(partition)!=3 or min(partition)<1 or sum(partition)!=72:raise ValueError('Layer partition must be three positive numbers summing to 72')
        self.token=uuid.uuid4().hex;self.process=None
        with socket.socket() as s:s.bind(('127.0.0.1',0));self.port=s.getsockname()[1]
        self.log=tempfile.TemporaryFile(mode='w+t',encoding='utf8')
        a,b,c=partition
        override=f'dory.num_loops=int:4,dory.recurrent_kv_cache_mode=str:last_loop,dory.input_layer_count=int:{a},dory.recurrent_layer_count=int:{b},dory.output_layer_count=int:{c}'
        args=[str(SERVER),'-m',str(MODEL),'--alias','dory','--host','127.0.0.1','--port',str(self.port),'--api-key',self.token,'--fit','off','-ngl','0' if device=='cpu' else '999','-c','4096','-np','1','-b','256','-ub','128','--flash-attn','off','--no-context-shift','--cache-ram','0','--jinja','--no-webui','--override-kv',override]
        if device!='cpu':args+=['--device',device]
        self.process=subprocess.Popen(args,stdout=self.log,stderr=self.log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    def request(self,path,body=None,timeout=180):
        req=urllib.request.Request(f'http://127.0.0.1:{self.port}{path}',data=None if body is None else json.dumps(body).encode(),headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
    def wait(self):
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if self.process.poll() is not None:
                self.log.seek(0);raise RuntimeError('Model server stopped: '+self.log.read()[-1600:])
            try:
                if self.request('/health',timeout=2).get('status')=='ok':return
            except (OSError,urllib.error.URLError):time.sleep(.25)
        raise TimeoutError('Model server did not become ready')
    def chat(self,messages,loops,thinking=False):
        return self.request('/v1/chat/completions',{'model':'dory','messages':messages,'stream':False,'temperature':0.35,'max_tokens':1024 if thinking else 512,'dory_loops':loops,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':thinking}})
    def chat_stream(self,messages,loops,thinking=False,max_tokens=2048):
        if type(max_tokens) is not int or not 128<=max_tokens<=3072:raise ValueError('Output tokens must be between 128 and 3072')
        body={'model':'dory','messages':messages,'stream':True,'temperature':0.35,'max_tokens':max_tokens,'dory_loops':loops,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':thinking}}
        req=urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as response:
            for raw in response:
                line=raw.decode('utf8').strip()
                if not line.startswith('data:'):continue
                data=line[5:].strip()
                if data=='[DONE]':return
                obj=json.loads(data)
                if obj.get('error'):raise RuntimeError(str(obj['error']))
                for choice in obj.get('choices',[]):
                    delta=choice.get('delta') or {}
                    yield {'thinking':delta.get('reasoning_content') or delta.get('reasoning') or '', 'answer':delta.get('content') or '', 'finish_reason':choice.get('finish_reason')}
            raise RuntimeError('Model stream disconnected before completion')
    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.log.close()

def main():
    global SERVER,MODEL
    parser=argparse.ArgumentParser(description='Experimental Dory layer-loop chat')
    parser.add_argument('--device',default='CUDA0',help='llama.cpp device, e.g. CUDA0, or cpu')
    parser.add_argument('--layers',default='16,40,16',help='input,recurrent,output layers; sum must be 72')
    parser.add_argument('--server',default=str(SERVER),help='Path to the patched llama-server executable')
    parser.add_argument('--model',default=str(MODEL),help='Path to the Dory GGUF model')
    args=parser.parse_args();partition=tuple(int(x) for x in args.layers.split(','))
    SERVER=pathlib.Path(args.server);MODEL=pathlib.Path(args.model)
    root=tk.Tk();root.title('Dory Loop Chat');root.geometry('800x620')
    frame=ttk.Frame(root,padding=12);frame.pack(fill='both',expand=True)
    mode=tk.StringVar(value='Medium (2 loops)');thinking=tk.BooleanVar(value=True);token_limit=tk.StringVar(value='2048');status=tk.StringVar(value='Starting model…');messages=[];events=queue.Queue();holder={'worker':None,'busy':True,'closing':False,'stream_part':None,'answer':''}
    header=ttk.Frame(frame);header.pack(fill='x');ttk.Label(header,text='Dory Loop Chat',font=('Segoe UI',16,'bold')).pack(side='left')
    ttk.Combobox(header,textvariable=mode,values=list(MODES),state='readonly',width=23).pack(side='right')
    ttk.Label(frame,text='Loop changes apply to the next reply. Alternate layer ranges are experimental.').pack(anchor='w',pady=5)
    options=ttk.Frame(frame);options.pack(fill='x');ttk.Checkbutton(options,text='Show model thinking',variable=thinking).pack(side='left');ttk.Label(options,text='Max output tokens').pack(side='left',padx=(20,4));ttk.Spinbox(options,from_=128,to=3072,increment=128,textvariable=token_limit,width=7).pack(side='left')
    output=tk.Text(frame,wrap='word',state='disabled');output.pack(fill='both',expand=True)
    entry=ttk.Entry(frame);entry.pack(fill='x',pady=(8,4));bar=ttk.Frame(frame);bar.pack(fill='x')
    ttk.Label(bar,textvariable=status).pack(side='left');send=ttk.Button(bar,text='Send');send.pack(side='right')
    def show(text):output.configure(state='normal');output.insert('end',text+'\n\n');output.see('end');output.configure(state='disabled')
    def append(text):output.configure(state='normal');output.insert('end',text);output.see('end');output.configure(state='disabled')
    def start():
        try:
            worker=Worker(args.device,partition);holder['worker']=worker
            if holder['closing']:worker.close();return
            worker.wait();events.put(('ready',None))
        except Exception as exc:events.put(('error',str(exc)))
    def submit(event=None):
        text=entry.get().strip()
        if not text or holder['busy']:return
        try:
            limit=int(token_limit.get())
            if not 128<=limit<=3072:raise ValueError()
        except ValueError:status.set('Output tokens must be 128–3072');return
        entry.delete(0,'end');show('You: '+text);messages.append({'role':'user','content':text});holder['busy']=True;holder['stream_part']=None;holder['answer']='';status.set('Thinking…');selected=MODES[mode.get()];show_thinking=thinking.get();snapshot=list(messages)
        def run():
            try:
                finish=None
                for chunk in holder['worker'].chat_stream(snapshot,selected,show_thinking,limit):
                    if chunk['finish_reason']:finish=chunk['finish_reason']
                    events.put(('chunk',chunk))
                events.put(('answer',finish))
            except Exception as exc:events.put(('failed',str(exc)))
        threading.Thread(target=run,daemon=True).start()
    def pump():
        if holder['closing']:return
        try:
            while True:
                kind,value=events.get_nowait()
                if kind=='ready':holder['busy']=False;status.set('Ready');entry.focus_set()
                elif kind=='chunk':
                    for part,label in (('thinking','Dory thinking:\n'),('answer','Dory: ')):
                        piece=value.get(part) or ''
                        if piece:
                            if holder['stream_part']!=part:append(('\n\n' if holder['stream_part'] else '')+label);holder['stream_part']=part
                            append(piece)
                            if part=='answer':holder['answer']+=piece
                elif kind=='answer':
                    answer=holder['answer']
                    if not answer.strip():append('\n\n[No final text returned; the response may have reached its output limit.]')
                    else:messages.append({'role':'assistant','content':answer})
                    append('\n\n');holder['busy']=False;status.set('Ready · '+str(value or 'completed'))
                elif kind=='failed':messages.pop();show('Error: '+value);holder['busy']=False;status.set('Request failed')
                else:show('Startup error: '+value);status.set('Unable to start')
        except queue.Empty:pass
        root.after(100,pump)
    def close():
        holder['closing']=True
        if holder['worker']:holder['worker'].close()
        root.destroy()
    send.configure(command=submit);entry.bind('<Return>',submit);root.protocol('WM_DELETE_WINDOW',close)
    threading.Thread(target=start,daemon=True).start();root.after(100,pump);root.mainloop()
if __name__=='__main__':main()
