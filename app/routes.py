from flask import Blueprint, render_template, request, jsonify, session, current_app
from werkzeug.utils import secure_filename
import os
from app.utils import load_and_compare_files
import uuid
import json

main = Blueprint('main', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def is_text_file(file_path):
    """Check if file is actually a text file (not binary)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)
        return True
    except (UnicodeDecodeError, UnicodeError):
        return False

def is_empty_file(file_path):
    """Check if file is empty (0 bytes)."""
    return os.path.getsize(file_path) == 0

@main.route('/')
def index():
    session.clear()
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_files():
    if 'file_a' not in request.files or 'file_b' not in request.files:
        return jsonify({'error': 'Both files are required'}), 400
    
    file_a = request.files['file_a']
    file_b = request.files['file_b']
    
    if file_a.filename == '' or file_b.filename == '':
        return jsonify({'error': 'No files selected', 'details': 'Please select both files before clicking Compare.'}), 400
    
    if not (allowed_file(file_a.filename) and allowed_file(file_b.filename)):
        return jsonify({
            'error': 'Invalid file type',
            'details': f'Only these file types are allowed: CSV, TXT, TSV, PIPE, DAT\n\nYour files: {file_a.filename}, {file_b.filename}'
        }), 400
    
    file_a_path = None
    file_b_path = None
    result_file_path = None
    
    try:
        session_id = str(uuid.uuid4())
        upload_folder = current_app.config['UPLOAD_FOLDER']
        
        file_a_path = os.path.join(upload_folder, f"{session_id}_a_{secure_filename(file_a.filename)}")
        file_b_path = os.path.join(upload_folder, f"{session_id}_b_{secure_filename(file_b.filename)}")
        
        file_a.save(file_a_path)
        file_b.save(file_b_path)
        
        # Check if files are empty (0 bytes)
        if is_empty_file(file_a_path):
            os.remove(file_a_path)
            os.remove(file_b_path)
            return jsonify({
                'error': f'❌ File A is empty',
                'details': f'The file "{file_a.filename}" contains no data (0 bytes).\n\nPlease upload a valid CSV file with data.'
            }), 400
        
        if is_empty_file(file_b_path):
            os.remove(file_a_path)
            os.remove(file_b_path)
            return jsonify({
                'error': f'❌ File B is empty',
                'details': f'The file "{file_b.filename}" contains no data (0 bytes).\n\nPlease upload a valid CSV file with data.'
            }), 400
        
        # Check if files are text files
        if not is_text_file(file_a_path):
            os.remove(file_a_path)
            os.remove(file_b_path)
            return jsonify({
                'error': f'❌ File A is not a valid text/CSV file',
                'details': f'The file "{file_a.filename}" appears to be a binary file (like Excel .xlsx).\n\nPlease export it to CSV format first:\n1. Open in Excel\n2. File > Save As > CSV (Comma delimited)'
            }), 400
        
        if not is_text_file(file_b_path):
            os.remove(file_a_path)
            os.remove(file_b_path)
            return jsonify({
                'error': f'❌ File B is not a valid text/CSV file',
                'details': f'The file "{file_b.filename}" appears to be a binary file (like Excel .xlsx).\n\nPlease export it to CSV format first:\n1. Open in Excel\n2. File > Save As > CSV (Comma delimited)'
            }), 400
        
        print(f"Processing files: {file_a.filename} vs {file_b.filename}")
        
        # ====== EXTRACT ADVANCED OPTIONS ======
        custom_keys_input = request.form.get('custom_keys', '').strip()
        numeric_tolerance_input = request.form.get('numeric_tolerance', '0.000000001')
        
        # Parse custom keys (comma-separated)
        custom_key_cols = None
        if custom_keys_input:
            custom_key_cols = [k.strip() for k in custom_keys_input.split(',') if k.strip()]
            print(f"Custom keys: {custom_key_cols}")
        else:
            print("Custom keys: None (auto-detect)")
        
        # Parse numeric tolerance
        try:
            numeric_tolerance = float(numeric_tolerance_input)
            print(f"Tolerance: {numeric_tolerance}")
        except ValueError:
            numeric_tolerance = 1e-9  # Default fallback
            print(f"Invalid tolerance, using default: {numeric_tolerance}")
        
        # Call comparison with advanced options
        result = load_and_compare_files(
            file_a_path, 
            file_b_path,
            numeric_tolerance=numeric_tolerance,
            custom_key_cols=custom_key_cols
        )
        # ====== END ADVANCED OPTIONS ======
        
        if 'error' in result:
            print(f"Error in comparison: {result['error']}")
            os.remove(file_a_path)
            os.remove(file_b_path)
            return jsonify({
                'error': result['error'],
                'details': result.get('details', '')
            }), 400
        
        # Store result in file
        result_file_path = os.path.join(upload_folder, f"{session_id}_result.json")
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Store minimal data in session
        session['result_file'] = result_file_path
        session['file_a_name'] = file_a.filename
        session['file_b_name'] = file_b.filename
        
        print(f"Comparison complete: {result.get('stats', {})}")
        
        # Clean up uploaded files
        os.remove(file_a_path)
        os.remove(file_b_path)
        
        return jsonify({
            'success': True,
            'redirect': '/results',
            'warning': result.get('warning'),
            'column_warning': result.get('column_warning')
        })
    
    except Exception as e:
        print(f"Exception in upload: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up files
        for path in [file_a_path, file_b_path, result_file_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        
        return jsonify({
            'error': f'❌ Processing error',
            'details': f'An unexpected error occurred:\n\n{str(e)}\n\nPlease check your file format and try again.'
        }), 500

@main.route('/results')
def results():
    print("Results route accessed")
    print(f"Session data: {dict(session)}")
    
    result_file = session.get('result_file')
    file_a_name = session.get('file_a_name', 'File A')
    file_b_name = session.get('file_b_name', 'File B')
    
    if not result_file:
        print("No result file in session")
        return render_template('index.html', error='No comparison data found. Please upload files.')
    
    if not os.path.exists(result_file):
        print(f"Result file not found: {result_file}")
        return render_template('index.html', error='Comparison data expired. Please upload files again.')
    
    # Load result from file
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
    except Exception as e:
        print(f"Error loading result file: {e}")
        return render_template('index.html', error='Error loading comparison results. Please try again.')
    
    # Clean up result file after loading
    try:
        os.remove(result_file)
    except:
        pass
    
    return render_template('results.html',
                         result=result,
                         file_a_name=file_a_name,
                         file_b_name=file_b_name)

@main.route('/api/comparison-data')
def get_comparison_data():
    result_file = session.get('result_file')
    if not result_file or not os.path.exists(result_file):
        return jsonify({'error': 'No comparison data'}), 404
    
    with open(result_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    return jsonify(result)
