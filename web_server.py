from flask import Flask, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def list_files():
    files = os.listdir('/app/results')
    files = [f for f in files if f.endswith('.json')]
    
    html = '<html><body><h1>Found Vanity Addresses</h1><ul>'
    for file in files:
        html += f'<li><a href="/download/{file}">{file}</a></li>'
    html += '</ul></body></html>'
    return html

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('/app/results', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
