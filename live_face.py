import torch
import cv2
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import pyttsx3
import time
import threading

# =========================
# IMPROVED VOICE ENGINE
# =========================
class VoiceAnnouncer:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.current_announcement = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        """Worker thread that handles TTS announcements"""
        while self.running:
            announcement = None
            with self.lock:
                if self.current_announcement:
                    announcement = self.current_announcement
                    self.current_announcement = None
            
            if announcement:
                try:
                    # Create fresh engine for each announcement
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 160)
                    engine.setProperty('volume', 1.0)
                    print(f"🔊 Speaking: {announcement}")
                    engine.say(announcement)
                    engine.runAndWait()
                    engine.stop()
                    del engine
                    print(f"✅ Finished speaking: {announcement}")
                except Exception as e:
                    print(f"❌ TTS Error: {e}")
            
            time.sleep(0.1)
    
    def announce(self, name):
        """Queue a name to be announced"""
        with self.lock:
            # Replace current announcement (don't queue duplicates)
            self.current_announcement = name
            print(f"📢 Queued: {name}")
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=3)

# =========================
# CONFIG
# =========================
DB_FILE = "face_db.pkl" 
THRESHOLD = 0.6
CAMERA_INDEX = 0
COOLDOWN_SECONDS = 2  # Minimum time between announcements
ABSENCE_FRAMES = 20  # Frames to wait before marking as absent (~0.7 seconds)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# MODELS
# =========================
mtcnn = MTCNN(
    keep_all=True, 
    device=device,
    post_process=False
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# LOAD DATABASE
# =========================
try:
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)
    print(f"📦 Loaded {len(face_db)} registered people")
    for name in face_db.keys():
        print(f"   - {name}")
except FileNotFoundError:
    print(f"❌ Database file '{DB_FILE}' not found. Please register faces first.")
    exit(1)

# =========================
# INITIALIZE VOICE
# =========================
announcer = VoiceAnnouncer()
time.sleep(0.5)  # Give TTS engine time to initialize

# =========================
# TRACKING STATE
# =========================
last_announcement_time = {}  # When person was last announced
person_present = {}  # Is person currently detected
absence_counter = {}  # Count frames of absence

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

print("\n🧠 Face recognition started | Press Q to quit")
print(f"⚙️ Cooldown: {COOLDOWN_SECONDS}s | Absence threshold: {ABSENCE_FRAMES} frames\n")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        current_time = time.time()
        
        # Convert frame for detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        try:
            boxes, probs = mtcnn.detect(pil_img)
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            boxes, probs = None, None
        
        detected_names_this_frame = set()
        
        # =========================
        # FACE DETECTION & RECOGNITION
        # =========================
        if boxes is not None and probs is not None:
            for box, prob in zip(boxes, probs):
                if prob < 0.9:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Extract and process face
                face_img = pil_img.crop((x1, y1, x2, y2))
                face_tensor = preprocess(face_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    emb = resnet(face_tensor)
                
                # Match against database
                name = "Unknown"
                best_score = 0.0
                
                for known_name, emb_list in face_db.items():
                    for known_emb in emb_list:
                        score = F.cosine_similarity(
                            emb,
                            known_emb.to(device)
                        ).item()
                        if score > best_score:
                            best_score = score
                            name = known_name
                
                if best_score < THRESHOLD:
                    name = "Unknown"
                
                # Track detection
                if name != "Unknown":
                    detected_names_this_frame.add(name)
                
                # Draw bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{name} ({best_score:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
        
        # =========================
        # PRESENCE TRACKING & ANNOUNCEMENTS
        # =========================
        all_known_people = set(face_db.keys())
        
        for person in all_known_people:
            if person in detected_names_this_frame:
                # Person detected
                was_absent = not person_present.get(person, False)
                person_present[person] = True
                absence_counter[person] = 0
                
                # Check if should announce
                last_announced = last_announcement_time.get(person, 0)
                time_since_last = current_time - last_announced
                
                should_announce = (
                    last_announced == 0 or  # Never announced
                    (was_absent and time_since_last >= COOLDOWN_SECONDS)  # Returned after absence
                )
                
                if should_announce:
                    announcer.announce(person)
                    last_announcement_time[person] = current_time
                    status = "FIRST TIME" if last_announced == 0 else "RETURNED"
                    print(f"🎯 {person} detected ({status}) - Announcing...")
                    
            else:
                # Person NOT detected
                if person_present.get(person, False):
                    absence_counter[person] = absence_counter.get(person, 0) + 1
                    
                    if absence_counter[person] >= ABSENCE_FRAMES:
                        person_present[person] = False
                        print(f"👋 {person} left the frame (absent {ABSENCE_FRAMES} frames)")
        
        # =========================
        # DISPLAY INFO
        # =========================
        status_text = f"Frame: {frame_count}"
        if detected_names_this_frame:
            status_text += f" | Present: {', '.join(detected_names_this_frame)}"
        
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Quitting...")
            break

except KeyboardInterrupt:
    print("\n👋 Interrupted by user")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("🧹 Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    announcer.stop()
    print("✅ Cleanup complete")
