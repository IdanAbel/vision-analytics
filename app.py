import os
import json
import uuid
import datetime
from flask import Flask, request, render_template, send_file, url_for, redirect
from werkzeug.utils import secure_filename
from process_video import process_video

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
INDEX_FILE = 'analysis_index.json'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_to_index(entry):
    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'w') as f:
            json.dump([], f)
    with open(INDEX_FILE, 'r+') as f:
        data = json.load(f)
        data.insert(0, entry)  # latest first
        f.seek(0)
        json.dump(data, f, indent=2)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No video part", 400
        file = request.files['video']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            unique_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(PROCESSED_FOLDER, unique_id)
            os.makedirs(output_dir, exist_ok=True)

            output_video_path = os.path.join(output_dir, 'tracked_video.mp4')
            heatmap_path = os.path.join(output_dir, 'heatmap.jpg')
            insights_path = os.path.join(output_dir, 'insights.json')

            process_video(upload_path, output_video_path, heatmap_path, insights_path)

            insights_data = {}
            try:
                with open(insights_path, 'r') as f:
                    insights_data = json.load(f)
            except Exception as e:
                print(f"Error reading insights JSON: {e}")

            # Save entry to index
            entry = {
                "id": unique_id,
                "filename": filename,
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "video_path": f"{output_dir}/tracked_video.mp4",
                "heatmap_path": f"{output_dir}/heatmap.jpg",
                "insights_path": f"{output_dir}/insights.json"
            }
            save_to_index(entry)

            return render_template('index.html',
                                   video_file=url_for('download_file', path=entry['video_path']),
                                   heatmap_file=url_for('download_file', path=entry['heatmap_path']),
                                   insights_file=url_for('download_file', path=entry['insights_path']),
                                   insights_data=insights_data)
        else:
            return "File type not allowed", 400

    return render_template('index.html')

@app.route('/processed/<path:path>')
def download_file(path):
    return send_file(path, as_attachment=True)

@app.route('/dashboard')
def dashboard():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE) as f:
            entries = json.load(f)
    else:
        entries = []
    return render_template('dashboard.html', entries=entries)

if __name__ == '__main__':
    app.run(debug=True)