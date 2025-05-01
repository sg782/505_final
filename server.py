from flask import Flask
from flask import request
from flask_cors import CORS
import subprocess
from PIL import Image


app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    #testing just to make sure it all runs
    return "<p>Hello world</p>"

@app.route('/run_script', methods=['POST'])
def run_script():

    
    # exctract file from our payload
    file = request.files['file']

    img = Image.open(file.stream).convert('RGB')

    temp_path = "temp_uploaded_image.png"
    img.save(temp_path)

    # using the subprocess module, we can easily call other scripts to get their outputs
    # we then capture the output printed to the terminal and return that to the frontend 
    result = subprocess.run(["python", "brain_tumor_predict.py", temp_path],capture_output=True,text=True)

    print(result)

    output = result.stdout.strip()
    print("Subprocess output:", output)

    return output, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
