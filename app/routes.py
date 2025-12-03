from flask import Blueprint, render_template, request, jsonify, session, current_app
from werkzeug.utils import secure_filename
import os
from app.utils import load_and_compare_files, compare_dataframes, select_best_key
from app.database_connector import DatabaseConnector
import uuid
import json
import pandas as pd


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


@main.route('/database-compare')
def database_compare_page():
    """Render the database comparison page."""
    return render_template('database_compare.html')


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
        session['comparison_type'] = 'file'
        
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


@main.route('/compare-database', methods=['POST'])
def compare_database():
    """Compare two database tables directly."""
    connector_a = None
    connector_b = None
    result_file_path = None
    
    try:
        data = request.json
        session_id = str(uuid.uuid4())
        
        # Source A configuration
        source_a_config = {
            'db_type': data['source_a']['db_type'],
            'host': data['source_a']['host'],
            'user': data['source_a']['user'],
            'password': data['source_a']['password'],
            'database': data['source_a']['database'],
            'port': data['source_a'].get('port'),
            'table': data['source_a']['table'],
            'schema': data['source_a'].get('schema'),
            'where_clause': data['source_a'].get('where_clause'),
            'limit': data['source_a'].get('limit')
        }
        
        # Source B configuration
        source_b_config = {
            'db_type': data['source_b']['db_type'],
            'host': data['source_b']['host'],
            'user': data['source_b']['user'],
            'password': data['source_b']['password'],
            'database': data['source_b']['database'],
            'port': data['source_b'].get('port'),
            'table': data['source_b']['table'],
            'schema': data['source_b'].get('schema'),
            'where_clause': data['source_b'].get('where_clause'),
            'limit': data['source_b'].get('limit')
        }
        
        # Snowflake-specific fields
        if source_a_config['db_type'] == 'snowflake':
            source_a_config['account'] = data['source_a']['account']
            source_a_config['warehouse'] = data['source_a'].get('warehouse')
            source_a_config['role'] = data['source_a'].get('role')
        
        if source_b_config['db_type'] == 'snowflake':
            source_b_config['account'] = data['source_b']['account']
            source_b_config['warehouse'] = data['source_b'].get('warehouse')
            source_b_config['role'] = data['source_b'].get('role')
        
        print(f"Connecting to Source A ({source_a_config['db_type']}): {source_a_config['table']}")
        
        # Connect to Source A
        connector_a = DatabaseConnector()
        connector_a.connect(source_a_config['db_type'], source_a_config)
        
        print(f"Connecting to Source B ({source_b_config['db_type']}): {source_b_config['table']}")
        
        # Connect to Source B
        connector_b = DatabaseConnector()
        connector_b.connect(source_b_config['db_type'], source_b_config)
        
        # Get table info
        table_info_a = connector_a.get_table_info(
            source_a_config['table'],
            source_a_config.get('schema')
        )
        
        table_info_b = connector_b.get_table_info(
            source_b_config['table'],
            source_b_config.get('schema')
        )
        
        print(f"Table A info: {table_info_a}")
        print(f"Table B info: {table_info_b}")
        
        # Check row limits (1 million max)
        if table_info_a['row_count'] > 1000000:
            return jsonify({
                'error': 'Table A too large',
                'details': f"Table A has {table_info_a['row_count']:,} rows (max: 1,000,000). Use WHERE clause or LIMIT to filter."
            }), 400
        
        if table_info_b['row_count'] > 1000000:
            return jsonify({
                'error': 'Table B too large',
                'details': f"Table B has {table_info_b['row_count']:,} rows (max: 1,000,000). Use WHERE clause or LIMIT to filter."
            }), 400
        
        print("Reading data from tables...")
        
        # Read data from tables
        df_a = connector_a.read_table(
            source_a_config['table'],
            source_a_config.get('schema'),
            source_a_config.get('limit'),
            source_a_config.get('where_clause')
        )
        
        df_b = connector_b.read_table(
            source_b_config['table'],
            source_b_config.get('schema'),
            source_b_config.get('limit'),
            source_b_config.get('where_clause')
        )
        
        print(f"Loaded {len(df_a)} rows from Table A, {len(df_b)} rows from Table B")
        
        # Normalize column names
        df_a.columns = df_a.columns.str.strip().str.lower()
        df_b.columns = df_b.columns.str.strip().str.lower()
        
        # Check column compatibility
        if set(df_a.columns) != set(df_b.columns):
            return jsonify({
                'error': 'Column mismatch',
                'details': f"Table A columns: {list(df_a.columns)}\nTable B columns: {list(df_b.columns)}"
            }), 400
        
        # Auto-select key columns or use custom
        if 'key_columns' in data and data['key_columns']:
            key_cols = data['key_columns']
            key_metadata = {'type': 'custom'}
        else:
            key_cols, key_metadata = select_best_key(df_a)
        
        print(f"Using key columns: {key_cols}")
        
        # Compare dataframes
        comparison_result = compare_dataframes(
            df_a, df_b, key_cols,
            numeric_tolerance=data.get('numeric_tolerance', 1e-9)
        )
        
        # Add metadata
        comparison_result['key_columns'] = key_cols
        comparison_result['key_metadata'] = key_metadata
        comparison_result['all_columns'] = list(df_a.columns)
        comparison_result['source_a_info'] = table_info_a
        comparison_result['source_b_info'] = table_info_b
        
        # Store result in file
        upload_folder = current_app.config['UPLOAD_FOLDER']
        result_file_path = os.path.join(upload_folder, f"{session_id}_result.json")
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, ensure_ascii=False, indent=2)
        
        # Store in session
        session['result_file'] = result_file_path
        session['file_a_name'] = f"{source_a_config.get('schema', '')}.{source_a_config['table']} ({source_a_config['db_type']})"
        session['file_b_name'] = f"{source_b_config.get('schema', '')}.{source_b_config['table']} ({source_b_config['db_type']})"
        session['comparison_type'] = 'database'
        
        print(f"Database comparison complete: {comparison_result.get('stats', {})}")
        
        # Close connections
        if connector_a:
            connector_a.close()
        if connector_b:
            connector_b.close()
        
        return jsonify({
            'success': True,
            'redirect': '/results'
        })
    
    except ConnectionError as e:
        print(f"Connection error: {str(e)}")
        if connector_a:
            connector_a.close()
        if connector_b:
            connector_b.close()
        return jsonify({'error': 'Connection failed', 'details': str(e)}), 500
    
    except Exception as e:
        print(f"Database comparison error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if connector_a:
            connector_a.close()
        if connector_b:
            connector_b.close()
        
        # Clean up result file if created
        if result_file_path and os.path.exists(result_file_path):
            try:
                os.remove(result_file_path)
            except:
                pass
        
        return jsonify({'error': 'Comparison failed', 'details': str(e)}), 500


@main.route('/test-db-connection', methods=['POST'])
def test_db_connection():
    """Test database connection without loading data."""
    connector = None
    
    try:
        data = request.json
        
        connector = DatabaseConnector()
        
        config = {
            'host': data['host'],
            'user': data['user'],
            'password': data['password'],
            'database': data['database'],
            'port': data.get('port')
        }
        
        if data['db_type'] == 'snowflake':
            config['account'] = data['account']
            config['warehouse'] = data.get('warehouse')
            config['schema'] = data.get('schema')
            config['role'] = data.get('role')
        
        print(f"Testing connection to {data['db_type']}: {config['host']}")
        
        connector.connect(data['db_type'], config)
        is_connected, message = connector.test_connection()
        
        if connector:
            connector.close()
        
        return jsonify({
            'success': is_connected,
            'message': message
        })
    
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        if connector:
            connector.close()
        
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@main.route('/results')
def results():
    print("Results route accessed")
    print(f"Session data: {dict(session)}")
    
    result_file = session.get('result_file')
    file_a_name = session.get('file_a_name', 'Source A')
    file_b_name = session.get('file_b_name', 'Source B')
    comparison_type = session.get('comparison_type', 'file')
    
    if not result_file:
        print("No result file in session")
        return render_template('index.html', error='No comparison data found. Please upload files or compare database tables.')
    
    if not os.path.exists(result_file):
        print(f"Result file not found: {result_file}")
        return render_template('index.html', error='Comparison data expired. Please try again.')
    
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
                         file_b_name=file_b_name,
                         comparison_type=comparison_type)


@main.route('/api/comparison-data')
def get_comparison_data():
    result_file = session.get('result_file')
    if not result_file or not os.path.exists(result_file):
        return jsonify({'error': 'No comparison data'}), 404
    
    with open(result_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    return jsonify(result)
