
import os, sys, time, csv, math, sqlite3, urllib.request, traceback
from datetime import datetime
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QFrame,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QProgressBar,QMessageBox
import mediapipe as mp
from ultralytics import YOLO

BASE=os.path.dirname(os.path.abspath(__file__))
MODEL_DIR=os.path.join(BASE,"models"); EVIDENCE=os.path.join(BASE,"evidence"); REPORTS=os.path.join(BASE,"reports")
os.makedirs(MODEL_DIR,exist_ok=True); os.makedirs(EVIDENCE,exist_ok=True); os.makedirs(REPORTS,exist_ok=True)
FACE_MODEL=os.path.join(MODEL_DIR,"face_landmarker.task")
FACE_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DB=os.path.join(BASE,"exam_events.db")

def d(a,b): return math.hypot(a.x-b.x,a.y-b.y)
def ear(p,ids):
    a,b,c,e,f,g=[p[i] for i in ids]
    return (d(b,g)+d(c,f))/(2*d(a,e)+1e-9)

class App(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("PROCTOR VISION • AI EXAM MONITOR"); self.resize(1500,900)
        self.cap=None; self.face=None; self.yolo=None; self.running=True; self.monitoring=False
        self.faces=self.persons=self.phones=0; self.risk=0; self.events=0; self.head="CENTER"; self.eye="OPEN"
        self.last_face=time.time(); self.look_start=None; self.closed_start=None; self.last_event={}; self.last_ts=0
        self.frame=None; self.fps=0; self.fc=0; self.ft=time.time()
        self.conn=sqlite3.connect(DB); self.conn.execute("""CREATE TABLE IF NOT EXISTS events
        (id INTEGER PRIMARY KEY,timestamp TEXT,event TEXT,severity TEXT,risk INTEGER,details TEXT,snapshot TEXT)"""); self.conn.commit()
        self.ui(); self.load_models(); self.camera()
        self.timer=QTimer(); self.timer.timeout.connect(self.tick); self.timer.start(30)

    def ui(self):
        self.setStyleSheet("""QWidget{background:#070b10;color:#edf2f5;font-family:'Segoe UI';}
        QFrame#card{background:#101820;border:1px solid #26343f;border-radius:16px;}
        QLabel#title{font-size:28px;font-weight:800;} QLabel#sub{color:#82919e;font-size:12px;}
        QLabel#sec{color:#71818f;font-size:11px;font-weight:800;} QLabel#big{font-size:28px;font-weight:800;color:#39d7a3;}
        QLabel#metric{font-size:18px;font-weight:800;} QLabel#small{color:#8c9aa7;font-size:11px;}
        QPushButton{background:#16222c;border:1px solid #2b3a46;border-radius:9px;padding:10px;font-weight:700;}
        QPushButton:hover{border-color:#39d7a3;} QPushButton#primary{background:#123a30;border-color:#39d7a3;}
        QProgressBar{background:#1b252e;border:0;border-radius:5px;height:9px;} QProgressBar::chunk{background:#39d7a3;border-radius:5px;}""")
        c=QWidget(); self.setCentralWidget(c); r=QVBoxLayout(c); r.setContentsMargins(24,20,24,16)
        h=QHBoxLayout(); v=QVBoxLayout(); t=QLabel("PROCTOR VISION"); t.setObjectName("title"); s=QLabel("AI-powered examination integrity & attention analytics"); s.setObjectName("sub"); v.addWidget(t);v.addWidget(s);h.addLayout(v);h.addStretch()
        self.status=QLabel("● STARTING"); self.status.setStyleSheet("color:#d9a441;font-weight:800;"); h.addWidget(self.status);r.addLayout(h)
        b=QHBoxLayout()
        cam=QFrame();cam.setObjectName("card");cl=QVBoxLayout(cam);q=QHBoxLayout();x=QLabel("LIVE EXAM FEED");x.setObjectName("sec");q.addWidget(x);q.addStretch();self.fpsl=QLabel("FPS --");self.fpsl.setObjectName("small");q.addWidget(self.fpsl);cl.addLayout(q)
        self.video=QLabel("Initializing...");self.video.setAlignment(Qt.AlignCenter);self.video.setMinimumSize(850,600);self.video.setStyleSheet("background:#030609;border:1px solid #26343f;border-radius:12px;");cl.addWidget(self.video);b.addWidget(cam,3)
        side=QVBoxLayout()
        rc=QFrame();rc.setObjectName("card");rl=QVBoxLayout(rc);z=QLabel("LIVE RISK SCORE");z.setObjectName("sec");self.rv=QLabel("0 / 100");self.rv.setObjectName("big");self.rb=QProgressBar();self.rb.setRange(0,100);self.rt=QLabel("Normal monitoring");self.rt.setObjectName("small");rl.addWidget(z);rl.addWidget(self.rv);rl.addWidget(self.rb);rl.addWidget(self.rt);side.addWidget(rc)
        mc=QFrame();mc.setObjectName("card");ml=QVBoxLayout(mc);z=QLabel("VISION TELEMETRY");z.setObjectName("sec");ml.addWidget(z);g=QHBoxLayout()
        self.fl=QLabel("0\nFACES");self.fl.setObjectName("metric");self.pl=QLabel("0\nPERSONS");self.pl.setObjectName("metric");self.ph=QLabel("0\nPHONES");self.ph.setObjectName("metric");g.addWidget(self.fl);g.addWidget(self.pl);g.addWidget(self.ph);ml.addLayout(g)
        self.al=QLabel("ATTENTION • CENTER");self.al.setObjectName("small");self.el=QLabel("EYES • OPEN");self.el.setObjectName("small");self.ev=QLabel("EVENTS • 0");self.ev.setObjectName("small");ml.addWidget(self.al);ml.addWidget(self.el);ml.addWidget(self.ev);side.addWidget(mc)
        sc=QFrame();sc.setObjectName("card");sl=QVBoxLayout(sc);z=QLabel("SYSTEM STATUS");z.setObjectName("sec");sl.addWidget(z);self.model=QLabel("Models: starting...");self.model.setObjectName("small");self.msg=QLabel("Waiting for camera analysis");self.msg.setWordWrap(True);self.msg.setObjectName("small");sl.addWidget(self.model);sl.addWidget(self.msg);side.addWidget(sc)
        cc=QFrame();cc.setObjectName("card");xl=QVBoxLayout(cc);z=QLabel("EXAM CONTROL");z.setObjectName("sec");xl.addWidget(z);self.btn=QPushButton("START MONITORING");self.btn.setObjectName("primary");self.btn.clicked.connect(self.toggle);rep=QPushButton("EXPORT EXAM REPORT");rep.clicked.connect(self.report);xl.addWidget(self.btn);xl.addWidget(rep);side.addWidget(cc)
        ec=QFrame();ec.setObjectName("card");ll=QVBoxLayout(ec);z=QLabel("RECENT EVENTS");z.setObjectName("sec");ll.addWidget(z);self.notes=QLabel("No events yet.");self.notes.setWordWrap(True);self.notes.setObjectName("small");ll.addWidget(self.notes);side.addWidget(ec,1)
        b.addLayout(side,1);r.addLayout(b)
        foot=QHBoxLayout();q=QLabel("OpenCV • MediaPipe • YOLO • SQLite • PySide6");q.setObjectName("small");foot.addWidget(q);foot.addStretch();q=QLabel("ESC • Stop");q.setObjectName("small");foot.addWidget(q);r.addLayout(foot)

    def load_models(self):
        try:
            if not os.path.exists(FACE_MODEL): urllib.request.urlretrieve(FACE_URL,FACE_MODEL)
            Base=mp.tasks.BaseOptions; V=mp.tasks.vision
            opt=V.FaceLandmarkerOptions(base_options=Base(model_asset_path=FACE_MODEL),running_mode=V.RunningMode.VIDEO,num_faces=5,min_face_detection_confidence=.35,min_face_presence_confidence=.35,min_tracking_confidence=.35)
            self.face=V.FaceLandmarker.create_from_options(opt)
            self.model.setText("MediaPipe: READY")
        except Exception as e:
            self.model.setText("MediaPipe: ERROR — "+str(e)[:90]); self.msg.setText("Face analysis unavailable. See terminal for details.")
        try:
            self.yolo=YOLO("yolo11n.pt")
            self.model.setText(self.model.text()+"\nYOLO: READY")
        except Exception as e:
            self.model.setText(self.model.text()+"\nYOLO: ERROR — "+str(e)[:90]); self.msg.setText("Object analysis unavailable. Check internet/model download.")

    def camera(self):
        self.cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        if not self.cap.isOpened(): self.msg.setText("CAMERA ERROR: webcam could not be opened.")

    def event(self,name,severity="MEDIUM",weight=8,details=""):
        if not self.monitoring:return
        now=time.time()
        if now-self.last_event.get(name,0)<4:return
        self.last_event[name]=now;self.risk=min(100,self.risk+weight);self.events+=1
        snap=""
        if severity in ("HIGH","CRITICAL") and self.frame is not None:
            snap=os.path.join(EVIDENCE,f"{datetime.now():%Y%m%d_%H%M%S}_{name.replace(' ','_')}.jpg");cv2.imwrite(snap,self.frame)
        self.conn.execute("INSERT INTO events(timestamp,event,severity,risk,details,snapshot) VALUES(?,?,?,?,?,?)",(datetime.now().strftime("%H:%M:%S"),name,severity,self.risk,details,snap));self.conn.commit()

    def analyze_face(self, result):
        if not result.face_landmarks:
            self.faces=0
            if self.monitoring and time.time()-self.last_face>2.5:self.event("Face missing","HIGH",12,"No face detected for 2.5+ seconds.")
            return
        self.faces=len(result.face_landmarks);self.last_face=time.time()
        if self.faces>1:self.event("Multiple faces","CRITICAL",30,f"{self.faces} faces detected.")
        p=result.face_landmarks[0]
        e=(ear(p,[33,160,158,133,153,144])+ear(p,[362,385,387,263,373,380]))/2
        self.eye="CLOSED" if e<.18 else "OPEN"
        if self.eye=="CLOSED":
            if self.closed_start is None:self.closed_start=time.time()
            elif time.time()-self.closed_start>2:self.event("Prolonged eye closure","MEDIUM",7,"Eyes closed for extended interval.")
        else:self.closed_start=None
        nose=p[1]; lm=p[33]; rm=p[263]; mid=(lm.x+rm.x)/2;dx=nose.x-mid
        self.head="LEFT" if dx<-.045 else "RIGHT" if dx>.045 else "CENTER"
        if self.head!="CENTER":
            if self.look_start is None:self.look_start=time.time()
            elif time.time()-self.look_start>2:self.event("Looking away","MEDIUM",8,f"Head direction {self.head}.")
        else:self.look_start=None

    def analyze_objects(self,frame):
        self.persons=self.phones=0
        if self.yolo is None:return
        try:
            rs=self.yolo.predict(frame,conf=.30,imgsz=640,verbose=False)
            names=rs[0].names
            for box in rs[0].boxes:
                cls=int(box.cls[0]);conf=float(box.conf[0]);name=names.get(cls,str(cls));x1,y1,x2,y2=map(int,box.xyxy[0])
                if name=="person":
                    self.persons+=1;cv2.rectangle(frame,(x1,y1),(x2,y2),(80,180,255),2);cv2.putText(frame,f"PERSON {conf:.0%}",(x1,max(20,y1-8)),0,.5,(80,180,255),2)
                elif name=="cell phone":
                    self.phones+=1;cv2.rectangle(frame,(x1,y1),(x2,y2),(80,80,240),3);cv2.putText(frame,f"PHONE {conf:.0%}",(x1,max(20,y1-8)),0,.55,(80,80,240),2)
            if self.phones:self.event("Phone detected","CRITICAL",35,"Cell phone detected in camera view.")
        except Exception as e:
            self.msg.setText("YOLO runtime error: "+str(e)[:100])

    def tick(self):
        ok,f=self.cap.read()
        if not ok:return
        f=cv2.flip(f,1);self.frame=f.copy()
        self.frame_no=getattr(self,"frame_no",0)+1
        if self.frame_no%2==0 and self.face:
            try:
                rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB);img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                ts=int(time.time()*1000);ts=max(ts,self.last_ts+1);self.last_ts=ts
                res=self.face.detect_for_video(img,ts);self.analyze_face(res)
                if res.face_landmarks:
                    h,w=f.shape[:2]
                    for face in res.face_landmarks:
                        for i in [1,33,263,61,291]:
                            p=face[i];cv2.circle(f,(int(p.x*w),int(p.y*h)),3,(60,220,165),-1)
            except Exception as e:self.msg.setText("MediaPipe runtime error: "+str(e)[:100])
            self.analyze_objects(f)
        cv2.rectangle(f,(15,15),(430,85),(7,11,15),-1);c=(60,220,165) if self.monitoring else (190,160,70)
        cv2.putText(f,"MONITORING ACTIVE" if self.monitoring else "MONITORING PAUSED",(30,48),0,.7,c,2);cv2.putText(f,f"RISK {self.risk}/100",(30,72),0,.45,(220,225,230),1)
        h,w=f.shape[:2];cv2.putText(f,f"HEAD: {self.head}",(w-210,40),0,.55,(220,225,230),2);cv2.putText(f,f"EVENTS: {self.events}",(w-210,65),0,.5,(160,170,180),1)
        self.fc=getattr(self,"fc",0)+1;el=time.time()-self.ft
        if el>=1:self.fps=self.fc/el;self.fc=0;self.ft=time.time()
        rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB);h,w,ch=rgb.shape;q=QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888);self.video.setPixmap(QPixmap.fromImage(q).scaled(self.video.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        self.fpsl.setText(f"FPS {self.fps:.0f}");self.rv.setText(f"{self.risk} / 100");self.rb.setValue(self.risk)
        self.rt.setText("LOW • Normal" if self.risk<25 else "MEDIUM • Review" if self.risk<60 else "HIGH • Attention" if self.risk<85 else "CRITICAL • Review required")
        self.fl.setText(f"{self.faces}\nFACES");self.pl.setText(f"{self.persons}\nPERSONS");self.ph.setText(f"{self.phones}\nPHONES");self.al.setText(f"ATTENTION • {self.head}");self.el.setText(f"EYES • {self.eye}");self.ev.setText(f"EVENTS • {self.events}")
        rows=self.conn.execute("SELECT timestamp,event,severity FROM events ORDER BY id DESC LIMIT 5").fetchall()
        self.notes.setText("\n".join(f"{a} • {b} [{c}]" for a,b,c in rows) if rows else "No events yet.")

    def toggle(self):
        self.monitoring=not self.monitoring;self.btn.setText("PAUSE MONITORING" if self.monitoring else "START MONITORING");self.status.setText("● MONITORING ACTIVE" if self.monitoring else "● PAUSED")

    def report(self):
        p=os.path.join(REPORTS,f"exam_report_{datetime.now():%Y%m%d_%H%M%S}.csv");rows=self.conn.execute("SELECT timestamp,event,severity,risk,details,snapshot FROM events ORDER BY id").fetchall()
        with open(p,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["Timestamp","Event","Severity","Risk","Details","Snapshot"]);w.writerows(rows)
        QMessageBox.information(self,"Report exported",p)

    def keyPressEvent(self,e):
        if e.key()==Qt.Key_Escape:self.close()
    def closeEvent(self,e):
        self.running=False
        if self.timer:self.timer.stop()
        if self.cap:self.cap.release()
        if self.face:self.face.close()
        self.conn.close();e.accept()

if __name__=="__main__":
    app=QApplication(sys.argv);w=App();w.show();sys.exit(app.exec())
