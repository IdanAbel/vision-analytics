import os
from flask import Flask, request, render_template, send_file, url_for
from werkzeug.utils import secure_filename
from process_video import process_video  # מייבאים את הפונקציה מהקובץ process_video.py

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

            output_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'tracked_' + filename)
            heatmap_path = os.path.join(app.config['PROCESSED_FOLDER'], 'heatmap_' + filename + '.jpg')

            process_video(upload_path, output_video_path, heatmap_path)

            return render_template('index.html',
                                   video_file=url_for('download_file', filename='tracked_' + filename),
                                   heatmap_file=url_for('download_file', filename='heatmap_' + filename + '.jpg'))

        else:
            return "File type not allowed", 400

    return render_template('index.html')

@app.route('/processed/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)
