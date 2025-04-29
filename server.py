from flask import Flask
from flask import request
from flask_cors import CORS
import subprocess
from PIL import Image


app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello world</p>"

@app.route('/run_script', methods=['POST'])
def run_script():

    file = request.files['file']

    img = Image.open(file.stream).convert('RGB')

    temp_path = "temp_uploaded_image.png"
    img.save(temp_path)

    result = subprocess.run(["python", "brain_tumor_predict.py", temp_path],capture_output=True,text=True)

    print(result)

    output = result.stdout.strip()
    print("Subprocess output:", output)

    return output, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
