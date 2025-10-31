from flask import Flask, render_template, request, jsonify
from summarizer_model import SummarizerModel
import fitz  
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("🚀 Starting application...")

try:
    summarizer = SummarizerModel()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    summarizer = None

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        print(f"📄 Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        if summarizer is None:
            return jsonify({'error': 'Summarization model is not available. Please try again later.'}), 500
        
        text = ""
        summary_length = request.form.get('summary_length', 'short')
        
        print(f"📨 Received request: summary_length={summary_length}")
        
        if 'pdf_file' in request.files:
            file = request.files['pdf_file']
            if file.filename != '':
                print(f"📂 Processing PDF: {file.filename}")
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                text = extract_text_from_pdf(file_path)
                os.remove(file_path)
        
        if not text and 'text_input' in request.form:
            text = request.form['text_input'].strip()
            print(f"📝 Processing text input: {len(text)} characters")
        
        if not text:
            return jsonify({'error': 'Please upload a PDF or enter some text.'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text is too short. Please provide more content.'}), 400
        
        print("🔄 Generating summary...")
        summary = summarizer.summarize(text, summary_length)
        print("✅ Summary generated successfully!")
        
        return jsonify({
            'summary': summary,
            'input_length': len(text),
            'summary_type': summary_length
        })
        
    except Exception as e:
        print(f"❌ Error in summarize route: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if summarizer else 'unhealthy',
        'model_loaded': summarizer is not None
    })

if __name__ == '__main__':
    print("🌐 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)