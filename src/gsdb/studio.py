"""Offline Windows Gaussian training studio. Run with the gsplat environment."""
from __future__ import annotations

import argparse
import math
import os
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .studio_data import discover_models, read_image
from .studio_edit import project_points
from .studio_runtime import Runtime

BG='#111720'
PANEL='#1a2330'
FG='#e4ecf5'
MUTED='#92a4b8'
ACCENT='#51d5c2'


class Studio:
    def __init__(self,root):
        self.root=root
        root.title('GS Studio · 本地高斯训练')
        root.geometry('1480x940')
        root.minsize(1160,780)
        root.configure(bg=BG)
        self.runtime=Runtime()
        self.dataset=None
        self.editor=None
        self.row=None
        self.render_row=None
        self.center=np.zeros(3)
        self.distance=5.
        self.yaw=0.
        self.pitch=0.
        self.camera_mode=False
        self.image_zoom=1.
        self.image_pan=np.zeros(2)
        self.drag=None
        self.render_image=None
        self.source_image=None
        self.samples=[]
        self.busy=False
        self.bookmark=None
        self.closing=False
        self._photos=[]
        style=ttk.Style(root)
        style.theme_use('clam')
        style.configure('.',background=PANEL,foreground=FG,font=('Microsoft YaHei UI',9))
        style.configure('TFrame',background=PANEL)
        style.configure('TLabel',background=PANEL)
        style.configure('TButton',padding=(7,5),background='#28374b',foreground=FG)
        style.map('TButton',background=[('active','#36516a')],foreground=[('disabled','#607086')])
        style.configure('TEntry',fieldbackground='#101923',foreground=FG,insertcolor=FG)
        style.configure('TCombobox',fieldbackground='#101923',foreground=FG,arrowsize=14)
        style.map('TCombobox',fieldbackground=[('readonly','#101923')],selectbackground=[('readonly','#101923')])
        style.configure('TCheckbutton',background=PANEL,foreground=FG)
        style.configure('TLabelframe',background=PANEL,foreground=FG)
        style.configure('TLabelframe.Label',background=PANEL,foreground=ACCENT)
        style.configure('TNotebook',background=PANEL)
        style.configure('TNotebook.Tab',padding=(9,6))
        style.configure('Horizontal.TProgressbar',background=ACCENT,troughcolor='#28374b')
        header=tk.Frame(root,bg=BG,height=60)
        header.pack(fill='x',padx=18,pady=(12,8))
        tk.Label(header,text='GS / STUDIO',font=('Segoe UI',20,'bold'),fg=ACCENT,bg=BG).pack(side='left')
        tk.Label(header,text='RealityScan → 本地训练 → 模型清理',fg=MUTED,bg=BG,font=('Microsoft YaHei UI',10)).pack(side='left',padx=24)
        for text,fn in [('打开项目',self.open_project),('保存项目',self.save_project),('另存项目',self.save_as),('打开 PLY',self.open_ply)]:
            ttk.Button(header,text=text,command=fn).pack(side='right',padx=3)
        body=tk.PanedWindow(root,orient='horizontal',bg=BG,sashwidth=8)
        body.pack(fill='both',expand=True,padx=14)
        left=ttk.Frame(body,padding=12)
        middle=ttk.Frame(body,padding=8)
        right=ttk.Frame(body,padding=12)
        body.add(left,width=285,minsize=250,stretch='never')
        body.add(middle,minsize=510,stretch='always')
        body.add(right,width=305,minsize=285,stretch='never')
        self._import_panel(left)
        self._viewport_panel(middle)
        self._right_panel(right)
        self.notice=tk.StringVar(value='拖入数据集文件夹，或选择 images + COLMAP。所有处理均在本机。')
        tk.Label(root,textvariable=self.notice,bg=BG,fg=MUTED,anchor='w').pack(fill='x',padx=18,pady=8)
        root.bind('<KeyPress>',self.key)
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.after(100,self.poll)
        root.after(200,self.enable_drop)

    def heading(self,parent,text):
        ttk.Label(parent,text=text,font=('Microsoft YaHei UI',11,'bold'),foreground=ACCENT).pack(anchor='w',pady=(3,8))

    def button(self,parent,text,fn):
        b=ttk.Button(parent,text=text,command=lambda:self.guard(fn))
        b.pack(fill='x',pady=3)
        return b

    def guard(self,fn):
        try:
            fn()
        except Exception as e:
            messagebox.showerror('GS Studio',str(e),parent=self.root)

    def path_field(self,parent,label,var,kind='directory'):
        ttk.Label(parent,text=label).pack(anchor='w',pady=(5,1))
        line=ttk.Frame(parent)
        line.pack(fill='x')
        ttk.Entry(line,textvariable=var).pack(side='left',fill='x',expand=True)
        def pick():
            path=filedialog.askdirectory(parent=self.root)
            if path:
                var.set(path)
                if var is self.root_path:
                    self.discover(path)
        ttk.Button(line,text='…',width=3,command=pick).pack(side='right')

    def _import_panel(self,p):
        self.heading(p,'01  数据集')
        self.root_path=tk.StringVar()
        self.images_path=tk.StringVar()
        self.model_path=tk.StringVar()
        self.masks_path=tk.StringVar()
        self.white_ignore=tk.BooleanVar()
        self.path_field(p,'数据集根目录 / 共享包',self.root_path)
        self.path_field(p,'images 目录（可选）',self.images_path)
        self.path_field(p,'COLMAP 模型目录（可选）',self.model_path)
        self.models=ttk.Combobox(p,textvariable=self.model_path,state='readonly')
        self.models.pack(fill='x',pady=3)
        self.path_field(p,'masks 目录（可选）',self.masks_path)
        ttk.Checkbutton(p,text='白色忽略（默认白色保留）',variable=self.white_ignore).pack(anchor='w',pady=5)
        self.import_button=self.button(p,'检查并导入',self.import_data)
        self.summary=tk.StringVar(value='尚未导入数据集')
        ttk.Label(p,textvariable=self.summary,wraplength=245,foreground=MUTED).pack(fill='x',pady=8)
        self.report=tk.Text(p,height=7,bg='#101923',fg=MUTED,wrap='word',font=('Microsoft YaHei UI',9),relief='flat')
        self.report.pack(fill='x')
        self.heading(p,'图片 / 相机')
        self.image_list=tk.Listbox(p,bg='#101923',fg=FG,selectbackground='#2b685f',relief='flat',exportselection=False)
        self.image_list.pack(fill='both',expand=True)
        self.image_list.bind('<<ListboxSelect>>',self.select_image)

    def _viewport_panel(self,p):
        toolbar=ttk.Frame(p)
        toolbar.pack(fill='x',pady=(0,5))
        for text,fn in [('全场景',self.fit),('保存视角',self.save_view),('恢复视角',self.restore_view)]:
            ttk.Button(toolbar,text=text,command=lambda f=fn:self.guard(f)).pack(side='left',padx=2)
        self.selection_mode=tk.BooleanVar()
        ttk.Checkbutton(toolbar,text='矩形选区',variable=self.selection_mode).pack(side='right')
        self.canvas=tk.Canvas(p,bg='#080e16',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.request_camera())
        self.canvas.bind('<ButtonPress-1>',lambda e:self.pointer_down(e,1))
        self.canvas.bind('<ButtonPress-3>',lambda e:self.pointer_down(e,3))
        self.canvas.bind('<B1-Motion>',self.pointer_move)
        self.canvas.bind('<B3-Motion>',self.pointer_move)
        self.canvas.bind('<ButtonRelease-1>',self.pointer_up)
        self.canvas.bind('<ButtonRelease-3>',self.pointer_up)
        self.canvas.bind('<MouseWheel>',self.wheel)
        self.canvas.bind('<Double-Button-1>',self.pick_camera)
        options=ttk.Frame(p)
        options.pack(fill='x',pady=5)
        self.show_cameras=tk.BooleanVar(value=True)
        self.show_points=tk.BooleanVar(value=True)
        self.show_grid=tk.BooleanVar(value=True)
        self.show_axes=tk.BooleanVar(value=True)
        self.show_mask=tk.BooleanVar(value=True)
        self.compare=tk.BooleanVar(value=False)
        for label,var in [('相机',self.show_cameras),('稀疏点',self.show_points),('网格',self.show_grid),('坐标轴',self.show_axes),('遮罩',self.show_mask),('源图对照',self.compare)]:
            ttk.Checkbutton(options,text=label,variable=var,command=self.redraw).pack(side='left')
        self.source_canvas=tk.Canvas(p,bg='#080e16',height=140,highlightthickness=0)
        self.source_canvas.pack(fill='x')
        self.source_canvas.bind('<MouseWheel>',self.wheel)
        self.source_canvas.bind('<ButtonPress-1>',lambda e:self.pointer_down(e,1))
        self.source_canvas.bind('<B1-Motion>',self.pointer_move)
        self.source_canvas.bind('<ButtonRelease-1>',self.pointer_up)
        ttk.Label(p,text='左键旋转 · 右键平移 · 滚轮缩放 · WASD/QE 漫游 · 双击相机查看图片',foreground=MUTED).pack(anchor='w',pady=4)

    def _right_panel(self,p):
        notebook=ttk.Notebook(p)
        notebook.pack(fill='both',expand=True)
        training=ttk.Frame(notebook,padding=6)
        editing=ttk.Frame(notebook,padding=6)
        notebook.add(training,text='训练控制')
        notebook.add(editing,text='模型编辑')
        p=training
        self.heading(p,'02  训练')
        self.state_label=tk.StringVar(value='等待输入')
        ttk.Label(p,textvariable=self.state_label,font=('Microsoft YaHei UI',12,'bold')).pack(anchor='w')
        self.device_label=tk.StringVar(value='GPU：开始训练时检查')
        ttk.Label(p,textvariable=self.device_label,wraplength=265,foreground=MUTED).pack(anchor='w',pady=3)
        self.preset=tk.StringVar(value='正式训练')
        presets=ttk.Combobox(p,values=['正式训练','快速预览'],textvariable=self.preset,state='readonly')
        presets.pack(fill='x',pady=5)
        presets.bind('<<ComboboxSelected>>',self.apply_preset)
        self.steps=tk.StringVar(value='30000')
        self.max_size=tk.StringVar(value='0')
        self.degree=tk.StringVar(value='3')
        self.checkpoint=tk.StringVar(value='1000')
        self.photo=tk.BooleanVar(value=True)
        for label,var in [('训练步数',self.steps),('训练图像最长边（0=原图）',self.max_size)]:
            ttk.Label(p,text=label).pack(anchor='w',pady=(3,0))
            ttk.Entry(p,textvariable=var).pack(fill='x')
        ttk.Checkbutton(p,text='训练曝光 / 白平衡补偿',variable=self.photo).pack(anchor='w',pady=5)
        self.advanced=ttk.Frame(p)
        def toggle():
            if self.advanced.winfo_manager():
                self.advanced.pack_forget()
            else:
                self.advanced.pack(after=advanced_button,fill='x')
        advanced_button=self.button(p,'高级参数 ▾',toggle)
        for label,var in [('SH 阶数（0–3，越高越能表现视角变化）',self.degree),('检查点间隔（步）',self.checkpoint)]:
            ttk.Label(self.advanced,text=label,wraplength=260).pack(anchor='w')
            ttk.Entry(self.advanced,textvariable=var).pack(fill='x')
        self.output=tk.StringVar(value=str(Path.cwd()/'studio-projects'))
        self.path_field(p,'项目输出目录',self.output)
        self.start_button=self.button(p,'开始新实验',self.start)
        controls=ttk.Frame(p)
        controls.pack(fill='x')
        ttk.Button(controls,text='暂停',command=self.runtime.pause.set).pack(side='left',expand=True,fill='x')
        ttk.Button(controls,text='继续 / 恢复',command=lambda:self.guard(self.resume)).pack(side='left',expand=True,fill='x')
        ttk.Button(controls,text='结束保留',command=self.runtime.stop.set).pack(side='left',expand=True,fill='x')
        self.button(p,'保存训练检查点',self.runtime.save_checkpoint.set)
        self.progress=ttk.Progressbar(p,maximum=100)
        self.progress.pack(fill='x',pady=7)
        self.progress_text=tk.StringVar(value='0 / 0 步 · 0 个高斯')
        ttk.Label(p,textvariable=self.progress_text,wraplength=265).pack(anchor='w')
        self.graph=tk.Canvas(p,height=55,bg='#101923',highlightthickness=0)
        self.graph.pack(fill='x',pady=5)
        self.preview_size=tk.StringVar(value='640')
        self.preview_hz=tk.StringVar(value='2')
        preview=ttk.Frame(p)
        preview.pack(fill='x')
        ttk.Label(preview,text='预览边长 / Hz').pack(side='left')
        ttk.Combobox(preview,values=['320','640','960'],textvariable=self.preview_size,width=5,state='readonly').pack(side='left',padx=3)
        hz=ttk.Combobox(preview,values=['1','2','5'],textvariable=self.preview_hz,width=3,state='readonly')
        hz.pack(side='left')
        self.preview_size.trace_add('write',lambda *a:self.request_camera())
        self.preview_hz.trace_add('write',lambda *a:setattr(self.runtime,'preview_interval',1/float(self.preview_hz.get())))
        p=editing
        self.heading(p,'03  快照编辑')
        self.selected_text=tk.StringVar(value='训练完成或打开 PLY 后编辑')
        ttk.Label(p,textvariable=self.selected_text,wraplength=265,foreground=MUTED).pack(anchor='w')
        self.through=tk.BooleanVar(value=True)
        ttk.Checkbutton(p,text='穿透选择（关闭时按像素最近中心）',variable=self.through).pack(anchor='w')
        row=ttk.Frame(p)
        row.pack(fill='x',pady=3)
        for label,fn in [('反选',lambda:self.edit('invert')),('清空',self.clear_selection),('删除',lambda:self.edit('delete')),('撤销',lambda:self.edit('undo')),('重做',lambda:self.edit('redo'))]:
            ttk.Button(row,text=label,width=4,command=lambda f=fn:self.guard(f)).pack(side='left',expand=True)
        row=ttk.Frame(p)
        row.pack(fill='x')
        ttk.Button(row,text='三维裁剪框',command=lambda:self.guard(self.crop_dialog)).pack(side='left',fill='x',expand=True)
        ttk.Button(row,text='位置 / 方向 / 缩放',command=lambda:self.guard(self.transform_dialog)).pack(side='left',fill='x',expand=True)
        self.button(p,'导出编辑结果 PLY',self.export)
        self.button(p,'导出原始训练 PLY',self.export_original)

    def discover(self,path):
        self.root_path.set(path)
        self.images_path.set('')
        self.masks_path.set('')
        values=discover_models(Path(path))
        self.models.configure(values=values)
        self.model_path.set(values[0] if len(values)==1 else '')
        if len(values)>1:
            self.notice.set('发现多个模型，请在 COLMAP 下拉框中选择一个。')

    def ensure_idle(self):
        if self.busy:
            raise ValueError('请先结束训练并保留成果，再更换数据或编辑快照。暂停期间可继续检查训练视角。')

    def import_data(self):
        self.ensure_idle()
        if not self.root_path.get():
            raise ValueError('请选择数据集根目录')
        self.busy=True
        self.runtime.submit(self.runtime.import_data,root=self.root_path.get(),images=self.images_path.get() or None,
            model=self.model_path.get() or None,masks=self.masks_path.get() or None,white_ignore=self.white_ignore.get())

    def settings(self):
        result=dict(steps=int(self.steps.get()),max_size=int(self.max_size.get()),sh_degree=int(self.degree.get()),
                    photo_comp=self.photo.get(),checkpoint_every=int(self.checkpoint.get()))
        if result['steps']<1 or result['checkpoint_every']<1 or result['max_size']<0 or result['sh_degree'] not in range(4):
            raise ValueError('参数无效：步数与检查点间隔必须为正数，SH 为 0–3')
        return result

    def apply_preset(self,event=None):
        fast=self.preset.get()=='快速预览'
        self.steps.set('1000' if fast else str(max(30000,30*sum(r['split']=='train' for r in self.dataset.rows)) if self.dataset else 30000))
        self.max_size.set('800' if fast else '0')
        self.degree.set('3')
        self.notice.set('快速预览仅用于检查链路与构图，不用于正式画质验收。' if fast else '正式训练：原始分辨率、SH3；可在开始前调整预算。')

    def start(self):
        self.ensure_idle()
        settings=self.settings()
        if not self.dataset or self.dataset.errors:
            raise ValueError('请先通过数据检查')
        self.busy=True
        self.samples=[]
        self.runtime.submit(self.runtime.start,self.output.get(),settings)

    def resume(self):
        if self.busy:
            self.runtime.pause.clear()
        else:
            settings=self.settings()
            if self.runtime.settings!=settings:
                raise ValueError('恢复时必须保留保存的配置；改配置请开始新实验。')
            if not self.runtime.experiment or not (self.runtime.experiment/'checkpoint.pt').is_file():
                raise ValueError('当前项目没有可恢复检查点')
            self.busy=True
            self.runtime.submit(self.runtime.start,self.output.get(),settings,True)

    def fit(self):
        xyz=self.editor.xyz()[self.editor.keep] if self.editor else self.dataset.xyz if self.dataset else np.array([[-1,-1,-1],[1,1,1]])
        low,high=xyz.min(axis=0),xyz.max(axis=0)
        self.center=(low+high)/2
        self.distance=max(float(np.linalg.norm(high-low))*1.2,0.1)
        self.camera_mode=False
        self.render_image=None
        self.request_camera()

    def camera(self):
        max_size=int(self.preview_size.get()) if hasattr(self,'preview_size') else 640
        if self.camera_mode and self.row:
            row=dict(self.row)
            factor=min(1,max_size/max(row['width'],row['height']))
            w,h=max(1,round(row['width']*factor)),max(1,round(row['height']*factor))
            K=np.asarray(row['K'],float).copy()
            K[0]*=w/row['width']
            K[1]*=h/row['height']
            K[0,0]*=self.image_zoom
            K[1,1]*=self.image_zoom
            K[0,2]=(K[0,2]-w/2-self.image_pan[0]*w)*self.image_zoom+w/2
            K[1,2]=(K[1,2]-h/2-self.image_pan[1]*h)*self.image_zoom+h/2
            return dict(row,width=w,height=h,K=K.tolist())
        aspect=max(1,self.canvas.winfo_width())/max(1,self.canvas.winfo_height())
        w,h=(max_size,max(1,round(max_size/aspect))) if aspect>=1 else (max(1,round(max_size*aspect)),max_size)
        forward=np.array([math.sin(self.yaw)*math.cos(self.pitch),math.sin(self.pitch),math.cos(self.yaw)*math.cos(self.pitch)])
        right=np.cross([0,1,0],forward)
        right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        eye=self.center-forward*self.distance
        R=np.array([right,down,forward])
        matrix=np.eye(4)
        matrix[:3,:3]=R
        matrix[:3,3]=-R@eye
        return dict(world_to_camera=matrix.tolist(),K=[[w*.9,0,w/2],[0,w*.9,h/2],[0,0,1]],width=w,height=h)

    def request_camera(self):
        if not hasattr(self,'canvas'):
            return
        self.runtime.set_camera(self.camera())
        self.redraw()

    def select_image(self,event=None):
        if self.dataset and self.image_list.curselection():
            self.row=self.dataset.rows[self.image_list.curselection()[0]]
            self.camera_mode=True
            self.image_zoom=1.
            self.image_pan=np.zeros(2)
            self.render_image=None
            try:
                self.source_image=read_image(self.row['source_path'])[...,::-1].copy()
            except Exception as e:
                self.notice.set(str(e))
            self.compare.set(True)
            self.request_camera()

    def view_state(self):
        return dict(center=self.center.tolist(),distance=self.distance,yaw=self.yaw,pitch=self.pitch,
                    camera_mode=self.camera_mode,image_index=self.image_list.curselection()[0] if self.image_list.curselection() else None,
                    bookmark=self.bookmark,image_zoom=self.image_zoom,image_pan=self.image_pan.tolist())

    def apply_view(self,v):
        if not v:
            return
        self.center=np.asarray(v['center'],float)
        self.distance=float(v['distance'])
        self.yaw=float(v['yaw'])
        self.pitch=float(v['pitch'])
        self.camera_mode=bool(v.get('camera_mode'))
        self.bookmark=v.get('bookmark')
        if self.camera_mode and self.dataset and v.get('image_index') is not None:
            self.image_list.selection_clear(0,'end')
            self.image_list.selection_set(v['image_index'])
            self.select_image()
        self.image_zoom=float(v.get('image_zoom',1))
        self.image_pan=np.asarray(v.get('image_pan',[0,0]),float)
        self.request_camera()

    def save_view(self):
        v=self.view_state()
        v.pop('bookmark',None)
        self.bookmark=v
        self.notice.set('观察位置已保存；保存项目后可跨会话恢复。')

    def restore_view(self):
        if self.bookmark:
            v=self.bookmark
            self.apply_view(v)
            self.bookmark=v

    def to_canvas(self,xy,row,canvas=None):
        canvas=canvas or self.canvas
        scale=min(canvas.winfo_width()/row['width'],canvas.winfo_height()/row['height'])
        offset=np.array([(canvas.winfo_width()-row['width']*scale)/2,(canvas.winfo_height()-row['height']*scale)/2])
        return xy*scale+offset,scale,offset

    def redraw(self):
        if not hasattr(self,'show_points'):
            return
        c=self.canvas
        c.delete('all')
        self._photos=[]
        row=self.camera()
        if self.render_image is not None and self.render_row==row:
            self.draw_image(c,self.render_image)
        if self.dataset and self.show_points.get():
            xyz=self.dataset.xyz[::max(1,len(self.dataset.xyz)//2500)]
            xy,z=project_points(xyz,row)
            coords,_,_=self.to_canvas(xy,row)
            for (x,y),depth in zip(coords,z):
                if depth>0 and 0<=x<c.winfo_width() and 0<=y<c.winfo_height():
                    c.create_oval(x-1,y-1,x+1,y+1,fill='#65788c',outline='')
        if self.show_grid.get():
            radius=self.distance
            for v in np.linspace(-radius,radius,11):
                for a,b in [([v,0,-radius],[v,0,radius]),([-radius,0,v],[radius,0,v])]:
                    self.line3d(np.array(a)+self.center,np.array(b)+self.center,row,'#283445')
        if self.show_axes.get():
            for axis,color in zip(np.eye(3),['#ec6d73','#68cb94','#6e9ee8']):
                self.line3d(self.center,self.center+axis*self.distance*.15,row,color,2)
        if self.dataset and self.show_cameras.get():
            for r in self.dataset.rows:
                inv=np.linalg.inv(r['world_to_camera'])
                origin=inv[:3,3]
                direction=inv[:3,2]*self.distance*.02
                self.line3d(origin,origin+direction,row,ACCENT,2)
                xy,z=project_points(origin[None],row)
                xy,_,_=self.to_canvas(xy,row)
                x,y=xy[0]
                if z[0]>0 and 0<=x<c.winfo_width() and 0<=y<c.winfo_height():
                    c.create_rectangle(x-3,y-3,x+3,y+3,outline=ACCENT)
        if self.editor and self.editor.selected.any():
            xyz=self.editor.xyz()[self.editor.selected]
            xyz=xyz[::max(1,len(xyz)//5000)]
            xy,z=project_points(xyz,row)
            xy,_,_=self.to_canvas(xy,row)
            for (x,y),depth in zip(xy,z):
                if depth>0 and 0<=x<c.winfo_width() and 0<=y<c.winfo_height():
                    c.create_oval(x-2,y-2,x+2,y+2,fill='#ffc963',outline='')
        if getattr(self,'crop_bounds',None):
            low,high=self.crop_bounds
            corners=np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
            for i in range(8):
                for bit in (1,2,4):
                    if i<(i^bit):
                        self.line3d(corners[i],corners[i^bit],row,'#ffc963',2)
        label='输入相机视角' if self.camera_mode else '自由视角'
        c.create_text(12,12,anchor='nw',fill=FG,text=label+'  /  '+('实时模型' if self.render_image is not None else '稀疏重建预览'))
        if not self.dataset and not self.editor:
            c.create_text(c.winfo_width()/2,c.winfo_height()/2,fill=MUTED,justify='center',font=('Microsoft YaHei UI',17),text='从已重建的数据开始\n\n导入 images + COLMAP\n或打开 Gaussian PLY')
        self.source_canvas.delete('all')
        if self.compare.get() and self.source_image is not None and self.row:
            rgb=self.source_image.copy()
            if self.show_mask.get() and self.dataset:
                keep=self.dataset.keep(self.row)
                rgb[~keep]=(rgb[~keep]*0.4+np.array([220,65,90])*0.6).astype(np.uint8)
            if self.camera_mode:
                h,w=rgb.shape[:2]
                center=np.array([w,h])*(.5+self.image_pan)
                half=np.array([w,h])/self.image_zoom/2
                box=tuple(np.rint([*(center-half),*(center+half)]).astype(int))
                rgb=np.asarray(Image.fromarray(rgb).crop(box))
            self.draw_image(self.source_canvas,rgb)
            self.source_canvas.create_text(8,8,anchor='nw',fill=FG,text=('同视角源图' if self.camera_mode else '源图参考（当前为自由视角）')+' · 红色为忽略区域')

    def draw_image(self,canvas,pixels):
        im=Image.fromarray(pixels)
        factor=min(max(1,canvas.winfo_width())/im.width,max(1,canvas.winfo_height())/im.height)
        im=im.resize((max(1,round(im.width*factor)),max(1,round(im.height*factor))),Image.Resampling.BILINEAR)
        photo=ImageTk.PhotoImage(im)
        self._photos.append(photo)
        canvas.create_image(canvas.winfo_width()/2,canvas.winfo_height()/2,image=photo)

    def line3d(self,a,b,row,color,width=1):
        xy,z=project_points(np.asarray([a,b]),row)
        if min(z)<=0:
            return
        xy,_,_=self.to_canvas(xy,row)
        self.canvas.create_line(*xy.ravel(),fill=color,width=width)

    def free_camera(self):
        if self.camera_mode and self.row:
            inv=np.linalg.inv(self.row['world_to_camera'])
            f=inv[:3,2]
            self.yaw=math.atan2(f[0],f[2])
            self.pitch=float(np.clip(math.asin(np.clip(f[1],-1,1)),-1.5,1.5))
            self.center=inv[:3,3]+f*self.distance
            self.camera_mode=False

    def pointer_down(self,event,button):
        self.canvas.focus_set()
        self.drag=(event.x,event.y,button,event.x,event.y)

    def pointer_move(self,event):
        if not self.drag:
            return
        x,y,button,x0,y0=self.drag
        if self.selection_mode.get() and button==1:
            self.canvas.delete('selection')
            self.canvas.create_rectangle(x0,y0,event.x,event.y,outline='#ffc963',tags='selection')
            return
        if self.camera_mode and self.compare.get() and button==1:
            self.image_pan-=np.array([event.x-x,event.y-y])/max(1,event.widget.winfo_width())/self.image_zoom
            self.drag=(event.x,event.y,button,x0,y0)
            self.request_camera()
            return
        self.free_camera()
        dx,dy=event.x-x,event.y-y
        if button==1:
            self.yaw+=dx*.006
            self.pitch=float(np.clip(self.pitch+dy*.006,-1.5,1.5))
        else:
            R=np.asarray(self.camera()['world_to_camera'])[:3,:3]
            self.center+=(-dx*R[0]-dy*R[1])*self.distance*.002
        self.drag=(event.x,event.y,button,x0,y0)
        self.request_camera()

    def pointer_up(self,event):
        if self.drag and self.selection_mode.get() and self.drag[2]==1:
            *_,x0,y0=self.drag
            row=self.camera()
            _,scale,offset=self.to_canvas(np.zeros((1,2)),row)
            rect=((np.array([[x0,y0],[event.x,event.y]])-offset)/scale).ravel().tolist()
            self.guard(lambda:self.edit('select',row,rect,self.through.get()))
        self.drag=None

    def wheel(self,event):
        if self.camera_mode and self.compare.get():
            self.image_zoom=float(np.clip(self.image_zoom*math.exp(event.delta/120*.12),1,32))
            self.request_camera()
            return
        self.free_camera()
        self.distance=max(1e-5,self.distance*math.exp(-event.delta/120*.12))
        self.request_camera()

    def key(self,event):
        if event.widget is not self.canvas:
            return
        key=event.keysym.lower()
        if key not in 'wasdqe' or len(key)!=1:
            return
        self.free_camera()
        R=np.asarray(self.camera()['world_to_camera'])[:3,:3]
        directions={'w':R[2],'s':-R[2],'a':-R[0],'d':R[0],'q':-R[1],'e':R[1]}
        self.center+=directions[key]*self.distance*.04
        self.request_camera()

    def pick_camera(self,event):
        if not self.dataset or self.selection_mode.get():
            return
        centers=np.array([np.linalg.inv(r['world_to_camera'])[:3,3] for r in self.dataset.rows])
        xy,z=project_points(centers,self.camera())
        xy,_,_=self.to_canvas(xy,self.camera())
        distance=np.linalg.norm(xy-[event.x,event.y],axis=1)
        distance[z<=0]=np.inf
        i=int(np.argmin(distance))
        if distance[i]<20:
            self.image_list.selection_clear(0,'end')
            self.image_list.selection_set(i)
            self.image_list.see(i)
            self.select_image()

    def edit(self,method,*args,**kwargs):
        self.ensure_idle()
        if not self.editor:
            raise ValueError('请先打开模型或结束训练以编辑快照')
        self.runtime.submit(self.runtime.edit,method,*args,**kwargs)

    def clear_selection(self):
        self.edit('select',self.camera(),[-2,-2,-1,-1],True)

    def numeric_dialog(self,title,fields,apply,extra=None):
        top=tk.Toplevel(self.root)
        top.title(title)
        top.configure(bg=PANEL)
        frame=ttk.Frame(top,padding=16)
        frame.pack(fill='both',expand=True)
        variables=[]
        for label,value in fields:
            ttk.Label(frame,text=label).pack(anchor='w')
            var=tk.StringVar(value=str(value))
            ttk.Entry(frame,textvariable=var,width=42).pack(fill='x',pady=(0,6))
            variables.append(var)
        if extra:
            extra(frame,variables)
        self.button(frame,'应用',lambda:apply([float(v.get()) for v in variables]))
        self.button(frame,'关闭',top.destroy)
        return top

    def crop_dialog(self):
        self.ensure_idle()
        if not self.editor:
            raise ValueError('请先打开 PLY')
        xyz=self.editor.xyz()[self.editor.keep]
        low,high=xyz.min(axis=0),xyz.max(axis=0)
        keep_inside=tk.BooleanVar(value=True)
        def values(v,commit):
            self.crop_bounds=(v[:3],v[3:])
            self.edit('crop',v[:3],v[3:],keep_inside.get(),commit)
        def extra(frame,vars):
            ttk.Checkbutton(frame,text='保留框内（取消则保留框外）',variable=keep_inside).pack(anchor='w')
            self.button(frame,'预览将删除的高斯',lambda:values([float(v.get()) for v in vars],False))
        top=self.numeric_dialog('三维裁剪 · 世界坐标',list(zip(['最小 X','最小 Y','最小 Z','最大 X','最大 Y','最大 Z'],[*low,*high])),lambda v:values(v,True),extra)
        def clear(e):
            if e.widget is top:
                self.crop_bounds=None
                self.redraw()
        top.bind('<Destroy>',clear)

    def transform_dialog(self):
        self.ensure_idle()
        if not self.editor:
            raise ValueError('请先打开 PLY')
        def extra(frame,vars):
            self.button(frame,'Z-up → Y-up（绕 X -90°）',lambda:vars[3].set('-90'))
            ttk.Label(frame,text='旋转顺序 X→Y→Z；围绕世界原点。\n高斯方向和 SH 光照系数同步调整。',foreground=MUTED).pack(anchor='w')
        self.numeric_dialog('场景变换',list(zip(['平移 X','平移 Y','平移 Z','旋转 X（度）','旋转 Y（度）','旋转 Z（度）','统一缩放'],[0,0,0,0,0,0,1])),
            lambda v:self.edit('transform',v[:3],v[3:6],v[6]),extra)

    def export(self):
        self.ensure_idle()
        path=filedialog.asksaveasfilename(parent=self.root,defaultextension='.ply',filetypes=[('Gaussian PLY','*.ply')])
        if path:
            self.edit('export',path)
            self.notice.set('正在导出编辑结果…')

    def export_original(self):
        self.ensure_idle()
        if not self.editor:
            raise ValueError('没有模型')
        path=filedialog.asksaveasfilename(parent=self.root,defaultextension='.ply')
        if path:
            def copy():
                import shutil
                if Path(path).exists():
                    raise FileExistsError('请选择不存在的新文件')
                shutil.copy2(self.editor.source,path)
                self.runtime.emit('notice',message=f'原始模型已导出: {path}')
            self.runtime.submit(copy)

    def open_ply(self):
        self.guard(self.ensure_idle)
        if self.busy:
            return
        path=filedialog.askopenfilename(parent=self.root,filetypes=[('Gaussian PLY','*.ply')])
        if path:
            self.runtime.submit(self.runtime.load_model,path)

    def open_project(self):
        self.guard(self.ensure_idle)
        if self.busy:
            return
        path=filedialog.askopenfilename(parent=self.root,filetypes=[('Studio 项目','project.json')])
        if path:
            self.runtime.submit(self.runtime.open_project,path)

    def save_project(self):
        if self.busy:
            self.runtime.pending_view=self.view_state()
            self.runtime.save_checkpoint.set()
            self.notice.set('训练检查点已请求；项目元数据将在训练结束时保存。')
        elif self.runtime.project:
            self.runtime.submit(self.runtime.save_project,self.view_state())
        else:
            self.save_as()

    def save_as(self):
        self.guard(self.ensure_idle)
        if self.busy:
            return
        path=filedialog.askdirectory(parent=self.root,title='选择另存项目的父目录')
        if path:
            self.runtime.submit(self.runtime.save_as,path,self.view_state())

    def poll(self):
        try:
            while True:
                kind,p=self.runtime.events.get_nowait()
                if kind=='status':
                    states=dict(empty='等待输入',preparing='准备中',ready='可训练',invalid='导入检查未通过',waiting='等待 GPU',training='训练中',paused='已暂停',completed='已完成 · 画质待验收',stopped='已停止 · 成果已保留',failed='失败 · 查看错误',viewing='模型查看 / 编辑')
                    self.state_label.set(p.get('message',states.get(p['state'],p['state'])))
                    self.busy=p['state'] in ('preparing','waiting','training','paused')
                    self.start_button.configure(state='disabled' if self.busy else 'normal')
                    self.import_button.configure(state='disabled' if self.busy else 'normal')
                elif kind=='dataset':
                    self.dataset=p['dataset']
                    self.editor=None
                    self.row=None
                    self.source_image=None
                    self.render_image=None
                    self.image_list.delete(0,'end')
                    for row in self.dataset.rows:
                        self.image_list.insert('end',('验 ' if row['split']=='validation' else '训 ')+row['source_image'])
                    n=sum(r['split']=='train' for r in self.dataset.rows)
                    self.summary.set(f'图片 {self.dataset.inventory} · 匹配 {len(self.dataset.rows)}\n训练 {n} · 验证 {len(self.dataset.rows)-n}\n稀疏点 {len(self.dataset.xyz):,}')
                    self.report.delete('1.0','end')
                    self.report.insert('end','\n'.join(['错误: '+e for e in self.dataset.errors]+self.dataset.warnings))
                    self.fit()
                elif kind=='editor':
                    self.editor=p['editor']
                    self.selected_text.set(f'{self.editor.keep.sum():,} 个高斯 · 已选 {self.editor.selected.sum():,}')
                    if p.get('fit',True):
                        self.fit()
                    else:
                        self.redraw()
                elif kind=='clear_dataset':
                    self.dataset=None
                    self.row=None
                    self.source_image=None
                    self.image_list.delete(0,'end')
                    self.summary.set('独立模型查看')
                    self.report.delete('1.0','end')
                elif kind=='preview':
                    if p['revision']==self.runtime.preview_revision:
                        self.render_image=p['pixels']
                        self.render_row=p['camera']
                        self.redraw()
                elif kind=='progress':
                    self.progress['value']=100*p['step']/p['target']
                    elapsed=p['elapsed_seconds']
                    rate=p['step']/max(elapsed,1e-6)
                    eta=(p['target']-p['step'])/rate if rate else 0
                    self.progress_text.set(f'{p["step"]:,} / {p["target"]:,} 步 · {p["splats"]:,} 高斯\n{rate:.1f} 步/秒 · 已用 {elapsed/60:.1f} 分\n剩余约 {eta/60:.1f} 分 · 显存 {p["vram_bytes"]/2**30:.2f} GiB')
                    if p.get('loss') is not None:
                        self.samples.append(p['loss'])
                        self.samples=self.samples[-500:]
                        self.graph.delete('all')
                        low,high=min(self.samples),max(self.samples)
                        coords=[v for i,y in enumerate(self.samples) for v in (i*max(1,self.graph.winfo_width()-8)/max(1,len(self.samples)-1)+4,48-(y-low)/max(high-low,1e-9)*36)]
                        if len(coords)>=4:
                            self.graph.create_line(*coords,fill=ACCENT,width=2)
                        self.graph.create_text(5,3,anchor='nw',fill=MUTED,text=f'Loss {p["loss"]:.5f}')
                elif kind=='device':
                    self.device_label.set(f'{p["name"]}\n可用 {p["free"]/2**30:.1f} / {p["total"]/2**30:.1f} GiB')
                elif kind=='project':
                    if p['settings']:
                        s=p['settings']
                        for var,key in [(self.steps,'steps'),(self.max_size,'max_size'),(self.degree,'sh_degree'),(self.checkpoint,'checkpoint_every')]:
                            var.set(str(s[key]))
                        self.photo.set(s['photo_comp'])
                    self.apply_view(p['view'])
                elif kind=='error':
                    self.notice.set(p['message'])
                    if not self.closing:
                        messagebox.showerror('GS Studio',p['message'],parent=self.root)
                elif kind=='notice':
                    self.notice.set(p['message'])
        except queue.Empty:
            pass
        if self.closing and not self.busy and not self.runtime.active_task.is_set() and self.runtime.tasks.empty():
            self.runtime.quit.set()
            self.root.destroy()
            return
        self.root.after(100,self.poll)

    def close(self):
        self.closing=True
        self.runtime.pending_view=self.view_state()
        if self.busy:
            self.runtime.stop.set()
            self.runtime.pause.clear()
            self.notice.set('正在保存训练成果，完成后关闭…')
        else:
            if self.runtime.project:
                self.runtime.submit(self.runtime.save_project,self.view_state())
            self.notice.set('正在完成保存 / 导出，完成后关闭…')

    def enable_drop(self):
        """Use the Windows file-drop message; no extra Tcl packages required."""
        if os.name!='nt':
            return
        import ctypes
        from ctypes import wintypes
        user=ctypes.windll.user32
        shell=ctypes.windll.shell32
        user.GetParent.argtypes=[wintypes.HWND]
        user.GetParent.restype=wintypes.HWND
        hwnd=user.GetParent(self.root.winfo_id())
        shell.DragAcceptFiles.argtypes=[wintypes.HWND,wintypes.BOOL]
        shell.DragQueryFileW.argtypes=[wintypes.HANDLE,wintypes.UINT,wintypes.LPWSTR,wintypes.UINT]
        shell.DragFinish.argtypes=[wintypes.HANDLE]
        callback=ctypes.WINFUNCTYPE(ctypes.c_ssize_t,wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM)
        setproc=user.SetWindowLongPtrW
        setproc.argtypes=[wintypes.HWND,ctypes.c_int,ctypes.c_void_p]
        setproc.restype=ctypes.c_void_p
        user.CallWindowProcW.argtypes=[ctypes.c_void_p,wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
        user.CallWindowProcW.restype=ctypes.c_ssize_t
        def drop(hwnd,msg,wparam,lparam):
            if msg==0x233:
                try:
                    length=shell.DragQueryFileW(wparam,0,None,0)
                    buffer=ctypes.create_unicode_buffer(length+1)
                    shell.DragQueryFileW(wparam,0,buffer,length+1)
                    path=buffer.value
                    self.root.after_idle(lambda:self.guard(lambda:self.drop_path(path)))
                finally:
                    shell.DragFinish(wparam)
                return 0
            return user.CallWindowProcW(self._old_proc,hwnd,msg,wparam,lparam)
        self._drop_callback=callback(drop)
        self._old_proc=setproc(hwnd,-4,ctypes.cast(self._drop_callback,ctypes.c_void_p))
        shell.DragAcceptFiles(hwnd,True)

    def drop_path(self,path):
        self.ensure_idle()
        if Path(path).is_dir():
            self.discover(path)
            self.notice.set('已接收文件夹；请检查目录与遮罩设置后导入。')
        elif Path(path).suffix.lower()=='.ply':
            self.runtime.submit(self.runtime.load_model,path)
        elif Path(path).name=='project.json':
            self.runtime.submit(self.runtime.open_project,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,help='预填数据集根目录，不自动训练')
    parser.add_argument('--project',type=Path,help='打开 project.json')
    args=parser.parse_args()
    root=tk.Tk()
    app=Studio(root)
    if args.dataset:
        app.discover(str(args.dataset.resolve()))
    if args.project:
        app.runtime.submit(app.runtime.open_project,args.project)
    root.mainloop()


if __name__=='__main__':
    main()
