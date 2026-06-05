import os
import io
import time
import gc
import threading
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash, abort
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import psutil
from flask_sqlalchemy import SQLAlchemy

from model import SimpleCNN, FastCNN, DNN  # Your model definitions here

# --------------------
# Config
# --------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///usersdata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
app.config['ALLOWED_MODEL_EXTENSIONS'] = {'pth', 'pt'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['MAX_MODEL_SIZE'] = 500 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# --------------------
# Database
# --------------------
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)

with app.app_context():
    db.create_all()

# --------------------
# Globals
# --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
idx_to_label = {}
model_lock = threading.Lock()

training_progress = {
    "status": "not started",
    "epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "learning_rate": 0.0,
    "epoch_time": 0.0,
    "message": "",
    "memory_usage": 0.0,
    "epoch_graph": [],
    "accuracy_graph": [],
    "loss_graph": [],
    "model_path": None,
    "final_loss": None,
    "final_accuracy": None,
    "error": None
}

# --------------------
# Utilities
# --------------------
def allowed_file(filename, file_type='image'):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'model':
        return ext in app.config['ALLOWED_MODEL_EXTENSIONS']
    return ext in app.config['ALLOWED_EXTENSIONS']

def get_memory_usage_mb():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024 / 1024

def preprocess_image(image: Image.Image, image_size=128):
    image = image.convert('RGB')
    image = ImageOps.fit(image, (image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std = np.array([0.229,0.224,0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2,0,1))
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device).float()
    return tensor

def save_checkpoint(model_obj, label_to_idx, model_type, path):
    ckpt = {
        'model_state_dict': model_obj.state_dict(),
        'label_to_idx': label_to_idx,
        'model_type': model_type
    }
    torch.save(ckpt, path)

def load_model_checkpoint(model_path):
    global model, idx_to_label
    with model_lock:
        if not os.path.exists(model_path):
            return False, f"Model file not found: {model_path}"
        try:
            ckpt = torch.load(model_path, map_location=device)
            if 'label_to_idx' not in ckpt or 'model_state_dict' not in ckpt:
                return False, "Invalid checkpoint format"
            model_type = ckpt.get('model_type', 'standard')
            num_classes = len(ckpt['label_to_idx'])
            if model_type == 'fast':
                m = FastCNN(num_classes=num_classes)
            elif model_type == 'dnn':
                m = DNN(num_classes=num_classes)
            else:
                m = SimpleCNN(num_classes=num_classes)
            m.load_state_dict(ckpt['model_state_dict'])
            m.to(device).float()
            m.eval()
            model = m
            idx_to_label = {v:k for k,v in ckpt['label_to_idx'].items()}
            return True, f"Loaded model with classes: {list(ckpt['label_to_idx'].keys())}"
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'username' not in session:
                flash('Please login first', 'error')
                return redirect(url_for('login'))
            if role and session.get('role') != role:
                flash('Access denied', 'error')
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return wrapper
    return decorator

# --------------------
# Routes: Pages
# --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        role = request.form['role']
        if User.query.filter((User.username==username)|(User.email==email)).first():
            flash('User already exists!', 'error')
            return redirect(url_for('register'))
        user = User(username=username, email=email, password=password, role=role)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = user.username
            session['role'] = user.role
            flash(f'Welcome {user.username}!', 'success')
            return redirect(url_for('training_page') if user.role=='admin' else url_for('prediction_page'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/training')
@login_required(role='admin')
def training_page():
    return render_template('training.html')

@app.route('/prediction')
@login_required(role='user')
def prediction_page():
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

# --------------------
# Routes: Image Upload / Prediction
# --------------------
@app.route('/upload_image', methods=['POST'])
@login_required()
def upload_image():
    if 'image_file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'})
    f = request.files['image_file']
    if f.filename == '' or not allowed_file(f.filename, 'image'):
        return jsonify({'success': False, 'message': 'Invalid image file'})
    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    # Create thumbnail
    img = Image.open(path)
    img.thumbnail((300,300))
    thumb_name = 'thumb_' + filename
    thumb_path = os.path.join(app.config['UPLOAD_FOLDER'], thumb_name)
    img.save(thumb_path)
    session['image_path'] = path
    session['thumb_path'] = thumb_path
    return jsonify({'success': True, 'image_url': f'/get_image/{filename}', 'thumb_url': f'/get_thumb/{thumb_name}'})

@app.route('/get_image/<path:filename>')
def get_image(filename):
    p = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(p): abort(404)
    return send_file(p)

@app.route('/get_thumb/<path:filename>')
def get_thumb(filename):
    p = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(p): abort(404)
    return send_file(p)

@app.route('/predict', methods=['POST'])
@login_required()
def predict():
    global model, idx_to_label
    if model is None: return jsonify({'success': False, 'message': 'No model loaded'})
    if 'image_path' not in session: return jsonify({'success': False, 'message': 'No image selected'})
    try:
        image_path = session['image_path']
        image_size = int(request.json.get('image_size', 64))
        img = Image.open(image_path)
        tensor = preprocess_image(img, image_size)
        with torch.no_grad():
            with model_lock:
                outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
        pred_idx = int(pred.item())
        confidence_val = float(confidence.item())
        all_probs = probs.cpu().numpy()[0].tolist()
        class_probs = [{'class': idx_to_label.get(i,str(i)), 'probability': float(p)} for i,p in enumerate(all_probs)]
        class_probs.sort(key=lambda x: x['probability'], reverse=True)
        predicted_label = idx_to_label.get(pred_idx, str(pred_idx))
        return jsonify({'success': True, 'prediction': predicted_label, 'confidence': confidence_val, 'probabilities': class_probs})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Prediction failed: {str(e)}'})

# --------------------
# Routes: Model Upload / Load
# --------------------
@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model_file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'})
    f = request.files['model_file']
    if f.filename == '':
        return jsonify({'success': False, 'message': 'Empty filename'})
    if not allowed_file(f.filename, 'model'):
        return jsonify({'success': False, 'message': 'Invalid model type'})

    f.stream.seek(0, os.SEEK_END)
    size = f.stream.tell()
    f.stream.seek(0)
    if size > app.config['MAX_MODEL_SIZE']:
        return jsonify({'success': False, 'message': f'Model too large'}), 413

    filename = secure_filename(f.filename)
    path = os.path.join(app.config['MODEL_FOLDER'], filename)
    try:
        f.save(path)
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to save file: {str(e)}'})

    ok, message = load_model_checkpoint(path)
    if ok:
        session['model_loaded'] = True
        session['model_path'] = path
        return jsonify({'success': True, 'message': message})
    else:
        try: os.remove(path)
        except: pass
        return jsonify({'success': False, 'message': message})

@app.route('/check_model_status')
@login_required()
def check_model_status():
    with model_lock:
        loaded = model is not None
        classes = list(idx_to_label.values()) if idx_to_label else []
    return jsonify({'model_loaded': loaded, 'classes': classes})

# --------------------
# Routes: Training
# --------------------
def training_thread_fn(data_folder, output_folder, model_type, epochs, batch_size, lr, image_size):
    global model, idx_to_label, training_progress
    try:
        training_progress.update({"status":"preparing","epoch":0,"total_epochs":epochs,"loss":0.0,"accuracy":0.0,"learning_rate":lr,"message":"Preparing dataset...","memory_usage":get_memory_usage_mb()})
        transform = transforms.Compose([transforms.Resize((image_size,image_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])
        dataset = datasets.ImageFolder(root=data_folder, transform=transform)
        if len(dataset)==0: raise Exception("No images found in dataset.")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        num_classes = len(dataset.classes)
        net = FastCNN(num_classes=num_classes) if model_type=='fast' else DNN(num_classes=num_classes) if model_type=='dnn' else SimpleCNN(num_classes=num_classes)
        net = net.to(device).float()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        training_progress['status'] = 'training'

        for ep in range(1, epochs+1):
            t0=time.time(); net.train(); running_loss=0.0; correct=0; total=0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device).float(), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs,1)
                correct += (preds==labels).sum().item()
                total += labels.size(0)
            epoch_loss = running_loss/total
            epoch_acc = 100.0*correct/total
            epoch_time = time.time()-t0
            training_progress['epoch_graph'].append(ep)
            training_progress['accuracy_graph'].append(epoch_acc)
            training_progress['loss_graph'].append(epoch_loss)
            training_progress.update({"epoch":ep,"loss":epoch_loss,"accuracy":epoch_acc,"epoch_time":epoch_time,"memory_usage":get_memory_usage_mb()})

        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"model_{model_type}_{timestamp}.pth"
        model_path = os.path.join(output_folder, model_name)
        save_checkpoint(net, dataset.class_to_idx, model_type, model_path)
        load_model_checkpoint(model_path)
        training_progress.update({"status":"completed","model_path":model_path,"final_loss":training_progress['loss'],"final_accuracy":training_progress['accuracy']})

    except Exception as e:
        training_progress.update({"status":"failed","error":str(e),"memory_usage":get_memory_usage_mb()})
    finally:
        gc.collect()

@app.route('/start_training', methods=['POST'])
@login_required(role='admin')
def start_training():
    data = request.json or {}
    data_folder = data.get('data_folder','./dataset')
    output_folder = data.get('output_folder', app.config['MODEL_FOLDER'])
    model_type = data.get('model_type','standard')
    epochs = int(data.get('epochs',5))
    batch_size = int(data.get('batch_size',32))
    lr = float(data.get('learning_rate',0.001))
    image_size = int(data.get('image_size',64))
    if training_progress.get('status') in ('preparing','training'):
        return jsonify({'success':False,'message':'Training already in progress'})
    t = threading.Thread(target=training_thread_fn,args=(data_folder,output_folder,model_type,epochs,batch_size,lr,image_size),daemon=True)
    t.start()
    return jsonify({'success':True,'message':'Training started'})

@app.route('/training_progress')
@login_required(role='admin')
def get_training_progress():
    return jsonify(training_progress)

@app.route('/training_graph')
@login_required(role='admin')
def training_graph():
    return jsonify({"epochs":training_progress.get("epoch_graph",[]),"accuracy":training_progress.get("accuracy_graph",[]),"loss":training_progress.get("loss_graph",[])})

@app.route('/system_info')
@login_required()
def system_info():
    mem_gb = psutil.virtual_memory().available/1024**3
    cpu_count = psutil.cpu_count(logical=True)
    return jsonify({'device':str(device),'cpu_count':cpu_count,'memory_available':round(mem_gb,2)})

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'success':False,'message':f'Payload too large. Max is {app.config["MAX_CONTENT_LENGTH"]} bytes'}),413

# --------------------
# Main
# --------------------
if __name__ == '__main__':
    print(f"Starting Flask server on device: {device}")
    try:
        if device.type=='cuda': print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except: pass
    app.run(host='0.0.0.0', port=5000, debug=True)
