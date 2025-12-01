import pandas as pd
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter
import re
import os

# Constants for memory management
MAX_ROWS = 1000000  # 1 million row limit
CHUNK_SIZE = 100000  # Process 100K rows at a time

def convert_to_python_types(obj):
    """Convert numpy/pandas types to native Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def detect_delimiter(file_path: str, sample_size: int = 5) -> str:
    """Automatically detect the delimiter of a delimited file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = ''.join([f.readline() for _ in range(sample_size)])
    
    try:
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        return delimiter
    except:
        common_delimiters = [',', '|', ';', '\t', ':']
        for delim in common_delimiters:
            if delim in sample:
                return delim
        return ','

def detect_header(file_path: str, delimiter: str) -> bool:
    """
    Auto-detect if CSV file has a header row.
    
    Returns:
        True if header detected, False otherwise
    """
    try:
        # Read first 10 rows without assuming header
        df_no_header = pd.read_csv(file_path, delimiter=delimiter, header=None, nrows=10, encoding='utf-8', low_memory=False)
        
        if len(df_no_header) < 2:
            return True  # Default to header if too few rows
        
        first_row = df_no_header.iloc[0]
        second_row = df_no_header.iloc[1]
        
        # Heuristic 1: Check if first row is all strings
        first_row_all_strings = all(isinstance(val, str) for val in first_row)
        
        # Heuristic 2: Check if first row has typical header patterns
        header_patterns = [
            r'.*_.*',      # Contains underscore (user_id)
            r'.*[A-Z].*',  # Contains uppercase (userId, CustomerName)
            r'^[a-zA-Z]+$' # Only letters (id, name, age)
        ]
        
        likely_headers = 0
        for val in first_row:
            if isinstance(val, str):
                for pattern in header_patterns:
                    if re.match(pattern, str(val)):
                        likely_headers += 1
                        break
        
        # Heuristic 3: Check if data rows have numeric types
        numeric_count = 0
        for val in second_row:
            try:
                float(val)
                numeric_count += 1
            except:
                pass
        
        # Decision 1: If first row is ALL strings and second row is mostly numeric
        if first_row_all_strings and numeric_count > len(second_row) / 2:
            return True
        
        # Decision 2: If >50% of first row matches header patterns
        if likely_headers >= len(first_row) / 2:
            return True
        
        # Decision 3: If first row has mixed types (numbers + strings)
        first_row_types = [type(val).__name__ for val in first_row]
        if 'int' in first_row_types or 'float' in first_row_types:
            return False  # Mixed types = data row, not header
        
        # Default: If uncertain, assume NO header (safer for pure data files)
        return False
    
    except Exception as e:
        print(f"Header detection error: {e}")
        return True  # On error, default to header (pandas default)

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and handle duplicates by adding suffix."""
    # First normalize the column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '_', regex=True)
    
    # Handle duplicate column names
    cols = pd.Series(df.columns)
    
    # Find duplicates and add suffix
    for dup in cols[cols.duplicated()].unique():
        dup_indices = cols[cols == dup].index.tolist()
        # Add suffix to duplicates (keep first occurrence as is, add _1, _2, etc. to rest)
        for i, idx in enumerate(dup_indices):
            if i > 0:  # Skip first occurrence
                cols.iloc[idx] = f"{dup}_{i}"
    
    df.columns = cols.tolist()
    return df

def normalize_value(value, numeric_tolerance: Optional[float] = None) -> any:
    """Normalize a value for comparison."""
    if pd.isna(value):
        return None
    
    if isinstance(value, str):
        return value.strip().lower()
    
    if isinstance(value, (int, float)) and numeric_tolerance is not None:
        return round(float(value), 5)
    
    return value

def identify_candidate_keys(df: pd.DataFrame) -> List[Dict]:
    """Identify all candidate key columns and composite keys."""
    candidates = []
    columns = df.columns.tolist()
    
    id_patterns = [r'.*id$', r'^id.*', r'.*key$', r'^key.*', r'.*uid$', r'^uid.*']
    for pattern in id_patterns:
        id_cols = [col for col in columns if re.match(pattern, col.lower())]
        for col in id_cols:
            uniqueness = df[col].nunique() / len(df) if len(df) > 0 else 0
            null_count = int(df[col].isna().sum())
            candidates.append({
                'columns': [col],
                'type': 'explicit_id',
                'uniqueness': float(uniqueness),
                'null_count': null_count,
                'is_unique': uniqueness == 1.0 and null_count == 0
            })
    
    if len(columns) > 0:
        first_col = columns[0]
        first_uniqueness = df[first_col].nunique() / len(df) if len(df) > 0 else 0
        first_nulls = int(df[first_col].isna().sum())
        if first_uniqueness >= 0.95:
            candidates.append({
                'columns': [first_col],
                'type': 'first_column',
                'uniqueness': float(first_uniqueness),
                'null_count': first_nulls,
                'is_unique': first_uniqueness == 1.0 and first_nulls == 0
            })
    
    if len(columns) >= 2:
        for i in range(min(3, len(columns))):
            for j in range(i + 1, min(i + 3, len(columns))):
                combo_cols = columns[i:j+1]
                if len(combo_cols) <= 3:
                    composite_key = df[combo_cols].apply(lambda row: '||'.join(row.astype(str)), axis=1)
                    composite_uniqueness = composite_key.nunique() / len(df) if len(df) > 0 else 0
                    composite_nulls = int(df[combo_cols].isna().any(axis=1).sum())
                    
                    if composite_uniqueness >= 0.95:
                        candidates.append({
                            'columns': combo_cols,
                            'type': 'composite',
                            'uniqueness': float(composite_uniqueness),
                            'null_count': composite_nulls,
                            'is_unique': composite_uniqueness == 1.0 and composite_nulls == 0
                        })
    
    candidates.sort(key=lambda x: (int(not x['is_unique']), -x['uniqueness'], x['null_count'], len(x['columns'])))
    
    return candidates

def select_best_key(df: pd.DataFrame) -> Tuple[List[str], Dict]:
    """Select the best key for comparison with metadata."""
    candidates = identify_candidate_keys(df)
    
    if not candidates:
        return df.columns.tolist(), {
            'type': 'all_columns',
            'uniqueness': 1.0,
            'null_count': 0,
            'is_unique': False,
            'warning': 'No suitable key found. Using all columns as composite key.'
        }
    
    best = candidates[0]
    
    metadata = {
        'type': best['type'],
        'uniqueness': best['uniqueness'],
        'null_count': best['null_count'],
        'is_unique': best['is_unique'],
        'all_candidates': candidates[:5]
    }
    
    if not best['is_unique']:
        metadata['warning'] = f"Selected key has duplicates ({best['uniqueness']*100:.1f}% unique)"
    
    if best['null_count'] > 0:
        if 'warning' not in metadata:
            metadata['warning'] = ''
        metadata['warning'] += f" Selected key has {best['null_count']} null/blank values"
    
    return best['columns'], metadata

def create_row_signature(row, columns, normalize: bool = True):
    """Create a unique signature for a row."""
    if normalize:
        values = [str(normalize_value(row[col])) for col in columns]
    else:
        values = [str(row[col]) for col in columns]
    return '||'.join(values)

def values_are_equal(val_a, val_b, numeric_tolerance: float = 1e-9) -> bool:
    """Compare two values with normalization and tolerance."""
    if pd.isna(val_a) and pd.isna(val_b):
        return True
    
    if pd.isna(val_a) or pd.isna(val_b):
        return False
    
    if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
        return abs(float(val_a) - float(val_b)) <= numeric_tolerance
    
    if isinstance(val_a, str) and isinstance(val_b, str):
        return normalize_value(val_a) == normalize_value(val_b)
    
    return val_a == val_b

def compare_dataframes(df_a: pd.DataFrame, df_b: pd.DataFrame, key_cols: List[str], 
                       numeric_tolerance: float = 1e-9) -> Dict:
    """
    Compare two dataframes with GREEDY MULTISET matching.
    
    Algorithm:
    - Iterate through File A in order
    - For each row in A, find first unmatched row in B with same key signature
    - Match them together and consume both
    - Remaining B rows are added to only_in_b AND if they match a key from A, 
      also added to matched_with_differences to highlight the duplication
    """
    df_a = df_a.reset_index(drop=True)
    df_b = df_b.reset_index(drop=True)
    
    all_cols = df_a.columns.tolist()
    value_cols = [col for col in all_cols if col not in key_cols]
    
    # Add original position tracking
    df_a['__original_pos__'] = range(len(df_a))
    df_b['__original_pos__'] = range(len(df_b))
    
    # Separate null keys
    null_mask_a = df_a[key_cols].isna().any(axis=1)
    null_mask_b = df_b[key_cols].isna().any(axis=1)
    
    null_idx_a = null_mask_a[null_mask_a].index
    null_idx_b = null_mask_b[null_mask_b].index
    valid_idx_a = null_mask_a[null_mask_a == False].index
    valid_idx_b = null_mask_b[null_mask_b == False].index
    
    null_key_rows_a = df_a.loc[null_idx_a].copy()
    null_key_rows_b = df_b.loc[null_idx_b].copy()
    
    df_a_valid = df_a.loc[valid_idx_a].copy()
    df_b_valid = df_b.loc[valid_idx_b].copy()
    
    # Create signatures
    df_a_valid['__full_signature__'] = df_a_valid.apply(
        lambda row: create_row_signature(row, all_cols, normalize=True), axis=1
    )
    df_b_valid['__full_signature__'] = df_b_valid.apply(
        lambda row: create_row_signature(row, all_cols, normalize=True), axis=1
    )
    
    df_a_valid['__key__'] = df_a_valid.apply(
        lambda row: create_row_signature(row, key_cols, normalize=True), axis=1
    )
    df_b_valid['__key__'] = df_b_valid.apply(
        lambda row: create_row_signature(row, key_cols, normalize=True), axis=1
    )
    
    # Detect duplicate keys for reporting
    key_count_a = df_a_valid['__key__'].value_counts().to_dict()
    key_count_b = df_b_valid['__key__'].value_counts().to_dict()
    
    duplicate_keys_info = {
        'file_a_duplicate_keys': int(sum(count - 1 for count in key_count_a.values() if count > 1)),
        'file_b_duplicate_keys': int(sum(count - 1 for count in key_count_b.values() if count > 1)),
        'duplicate_keys': []
    }
    
    # Build duplicate key list for reporting
    all_keys_with_dups = set([k for k, v in key_count_a.items() if v > 1]) | set([k for k, v in key_count_b.items() if v > 1])
    for dup_key in sorted(list(all_keys_with_dups))[:10]:
        count_a = key_count_a.get(dup_key, 0)
        count_b = key_count_b.get(dup_key, 0)
        duplicate_keys_info['duplicate_keys'].append({
            'key': dup_key,
            'count_a': int(count_a),
            'count_b': int(count_b)
        })
    
    only_in_a = []
    only_in_b = []
    matched_records = []
    matched_with_diff = []
    
    # Track which B rows have been matched and store matched rows from A for duplicate comparison
    matched_b_indices = set()
    matched_rows_from_a = {}  # Store the actual matched rows from A by key
    
    # If all columns are keys
    if set(key_cols) == set(all_cols):
        counter_a = Counter(df_a_valid['__full_signature__'])
        counter_b = Counter(df_b_valid['__full_signature__'])
        all_signatures = sorted(set(counter_a.keys()) | set(counter_b.keys()))
        
        for signature in all_signatures:
            count_a = counter_a.get(signature, 0)
            count_b = counter_b.get(signature, 0)
            min_count = min(count_a, count_b)
            
            if min_count > 0:
                matching_rows = df_a_valid[df_a_valid['__full_signature__'] == signature].sort_values(
                    by='__original_pos__'
                ).head(min_count)
                for _, row in matching_rows.iterrows():
                    matched_records.append({col: convert_to_python_types(row[col]) for col in all_cols})
            
            if count_a > min_count:
                diff = count_a - min_count
                extra_rows = df_a_valid[df_a_valid['__full_signature__'] == signature].sort_values(
                    by='__original_pos__'
                ).tail(diff)
                for _, row in extra_rows.iterrows():
                    only_in_a.append({col: convert_to_python_types(row[col]) for col in all_cols})
            
            if count_b > min_count:
                diff = count_b - min_count
                extra_rows = df_b_valid[df_b_valid['__full_signature__'] == signature].sort_values(
                    by='__original_pos__'
                ).tail(diff)
                
                # Get a matched row from A for comparison
                if count_a > 0:
                    matched_row_a = df_a_valid[df_a_valid['__full_signature__'] == signature].sort_values(
                        by='__original_pos__'
                    ).iloc[0]
                
                for _, row in extra_rows.iterrows():
                    row_dict = {col: convert_to_python_types(row[col]) for col in all_cols}
                    row_dict['__duplicate_warning__'] = f"Duplicate record (appears {count_b} times in File B, {count_a} times in File A)"
                    only_in_b.append(row_dict)
                    
                    # ALSO add to matched_with_differences to show the duplication
                    if count_a > 0:
                        differences = {}
                        for col in all_cols:
                            val_a = matched_row_a[col]
                            val_b = row[col]
                            differences[col] = {
                                'value_a': convert_to_python_types(val_a),
                                'value_b': convert_to_python_types(val_b),
                                'is_different': False  # Same values, just duplicate
                            }
                        
                        matched_with_diff.append({
                            'key': {col: convert_to_python_types(row[col]) for col in key_cols},
                            'columns': differences,
                            '__duplicate_in_b__': True,
                            '__duplicate_note__': f"Duplicate in File B (appears {count_b}x in B, {count_a}x in A)"
                        })
    
    else:
        # GREEDY MATCHING: Iterate through A, consume B rows in order
        df_a_valid = df_a_valid.sort_values(by='__original_pos__').reset_index(drop=True)
        df_b_valid = df_b_valid.sort_values(by='__original_pos__').reset_index(drop=True)
        
        # Iterate through each row in File A
        for idx_a, row_a in df_a_valid.iterrows():
            key_a = row_a['__key__']
            
            # Store this row for potential duplicate comparison later
            if key_a not in matched_rows_from_a:
                matched_rows_from_a[key_a] = []
            matched_rows_from_a[key_a].append(row_a)
            
            # Find first unmatched row in B with the same key
            matched = False
            for idx_b, row_b in df_b_valid.iterrows():
                # Skip if already matched
                if idx_b in matched_b_indices:
                    continue
                
                key_b = row_b['__key__']
                
                # Keys must match
                if key_a != key_b:
                    continue
                
                # Found a match!
                matched = True
                matched_b_indices.add(idx_b)
                
                # Check if rows are identical
                full_sig_a = row_a['__full_signature__']
                full_sig_b = row_b['__full_signature__']
                
                if full_sig_a == full_sig_b:
                    # Exact match → matched_records
                    matched_records.append({col: convert_to_python_types(row_a[col]) for col in all_cols})
                else:
                    # Has differences → matched_with_differences
                    differences = {}
                    has_diff = False
                    
                    for col in all_cols:
                        val_a = row_a[col]
                        val_b = row_b[col]
                        
                        is_different = not values_are_equal(val_a, val_b, numeric_tolerance)
                        
                        differences[col] = {
                            'value_a': convert_to_python_types(val_a),
                            'value_b': convert_to_python_types(val_b),
                            'is_different': bool(is_different)
                        }
                        
                        if is_different and col not in key_cols:
                            has_diff = True
                    
                    if has_diff:
                        matched_with_diff.append({
                            'key': {col: convert_to_python_types(row_a[col]) for col in key_cols},
                            'columns': differences
                        })
                    else:
                        matched_records.append({col: convert_to_python_types(row_a[col]) for col in all_cols})
                
                break  # Move to next row in A
            
            # If no match found in B
            if not matched:
                only_in_a.append({col: convert_to_python_types(row_a[col]) for col in all_cols})
        
        # Add unmatched B rows to only_in_b AND matched_with_differences
        for idx_b, row_b in df_b_valid.iterrows():
            if idx_b not in matched_b_indices:
                row_dict = {col: convert_to_python_types(row_b[col]) for col in all_cols}
                key_b = row_b['__key__']
                
                # Check if this key was matched from A (i.e., it's a duplicate in B)
                if key_b in matched_rows_from_a:
                    # This is a duplicate record in File B
                    count_in_b = key_count_b.get(key_b, 0)
                    count_in_a = key_count_a.get(key_b, 0)
                    row_dict['__duplicate_warning__'] = f"Duplicate record in File B (appears {count_in_b}x in B, {count_in_a}x in A)"
                    
                    # Get the first matched row from A for comparison
                    row_a_for_comparison = matched_rows_from_a[key_b][0]
                    
                    # ALSO add to matched_with_differences
                    differences = {}
                    has_actual_diff = False
                    
                    for col in all_cols:
                        val_a = row_a_for_comparison[col]
                        val_b = row_b[col]
                        
                        is_different = not values_are_equal(val_a, val_b, numeric_tolerance)
                        
                        differences[col] = {
                            'value_a': convert_to_python_types(val_a),
                            'value_b': convert_to_python_types(val_b),
                            'is_different': bool(is_different)
                        }
                        
                        if is_different and col not in key_cols:
                            has_actual_diff = True
                    
                    # Add to matched_with_diff regardless of whether values differ
                    matched_with_diff.append({
                        'key': {col: convert_to_python_types(row_b[col]) for col in key_cols},
                        'columns': differences,
                        '__duplicate_in_b__': True,
                        '__duplicate_note__': f"Duplicate in File B (appears {count_in_b}x in B, {count_in_a}x in A)"
                    })
                
                only_in_b.append(row_dict)
    
    # Add null key rows
    for _, row in null_key_rows_a.iterrows():
        row_dict = {col: convert_to_python_types(row[col]) for col in all_cols}
        row_dict['__null_key_warning__'] = True
        only_in_a.append(row_dict)
    
    for _, row in null_key_rows_b.iterrows():
        row_dict = {col: convert_to_python_types(row[col]) for col in all_cols}
        row_dict['__null_key_warning__'] = True
        only_in_b.append(row_dict)
    
    return {
        'only_in_a': only_in_a,
        'only_in_b': only_in_b,
        'matched_with_differences': matched_with_diff,
        'matched_records': matched_records,
        'duplicate_keys_info': duplicate_keys_info,
        'null_key_rows': {
            'file_a': int(len(null_key_rows_a)),
            'file_b': int(len(null_key_rows_b))
        },
        'stats': {
            'total_a': int(len(df_a)),
            'total_b': int(len(df_b)),
            'only_in_a_count': int(len(only_in_a)),
            'only_in_b_count': int(len(only_in_b)),
            'matched_with_diff_count': int(len(matched_with_diff)),
            'matched_no_diff_count': int(len(matched_records))
        }
    }


def count_rows(file_path: str, delimiter: str, has_header: bool) -> int:
    """Count rows in file without loading entire file."""
    try:
        row_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                row_count += 1
        return row_count - 1 if has_header else row_count
    except:
        return 0

def load_and_compare_files(file_a_path: str, file_b_path: str, 
                          numeric_tolerance: float = 1e-9,
                          custom_key_cols: Optional[List[str]] = None) -> Dict:
    """Main function to load and compare two files with critical protections."""
    try:
        # Get filenames for better error messages
        file_a_name = os.path.basename(file_a_path).split('_', 2)[-1] if '_' in os.path.basename(file_a_path) else os.path.basename(file_a_path)
        file_b_name = os.path.basename(file_b_path).split('_', 2)[-1] if '_' in os.path.basename(file_b_path) else os.path.basename(file_b_path)
        
        delim_a = detect_delimiter(file_a_path)
        delim_b = detect_delimiter(file_b_path)
        
        has_header_a = detect_header(file_a_path, delim_a)
        has_header_b = detect_header(file_b_path, delim_b)
        
        # Count rows before loading
        rows_a = count_rows(file_a_path, delim_a, has_header_a)
        rows_b = count_rows(file_b_path, delim_b, has_header_b)
        
        # Check row limits
        if rows_a > MAX_ROWS:
            return {
                'error': f'File A has too many rows',
                'details': f'"{file_a_name}" has {rows_a:,} rows, exceeding the limit of {MAX_ROWS:,} rows.\n\nPlease split the file or contact support for large file processing.'
            }
        
        if rows_b > MAX_ROWS:
            return {
                'error': f'File B has too many rows',
                'details': f'"{file_b_name}" has {rows_b:,} rows, exceeding the limit of {MAX_ROWS:,} rows.\n\nPlease split the file or contact support for large file processing.'
            }
        
        # Show warning for large files
        large_file_warning = None
        if rows_a > 100000 or rows_b > 100000:
            large_file_warning = f"Large files detected ({file_a_name}: {rows_a:,} rows, {file_b_name}: {rows_b:,} rows). Processing may take 30-60 seconds."
        
        # Load File A with specific error handling
        try:
            if has_header_a:
                df_a = pd.read_csv(file_a_path, delimiter=delim_a, encoding='utf-8', low_memory=False, nrows=MAX_ROWS)
            else:
                df_a = pd.read_csv(file_a_path, delimiter=delim_a, encoding='utf-8', low_memory=False, header=None, nrows=MAX_ROWS)
                df_a.columns = [f'col_{i}' for i in range(len(df_a.columns))]
        except pd.errors.EmptyDataError:
            return {
                'error': f'{file_a_name} is empty',
                'details': f'File A ("{file_a_name}") contains no data or only blank lines.\n\nPlease upload a valid CSV file with headers and data.'
            }
        except Exception as e:
            if 'No columns to parse' in str(e):
                return {
                    'error': f'{file_a_name} is empty',
                    'details': f'File A ("{file_a_name}") has no columns or data.\n\nPlease upload a valid CSV file.'
                }
            raise
        
        # Load File B with specific error handling
        try:
            if has_header_b:
                df_b = pd.read_csv(file_b_path, delimiter=delim_b, encoding='utf-8', low_memory=False, nrows=MAX_ROWS)
            else:
                df_b = pd.read_csv(file_b_path, delimiter=delim_b, encoding='utf-8', low_memory=False, header=None, nrows=MAX_ROWS)
                df_b.columns = [f'col_{i}' for i in range(len(df_b.columns))]
        except pd.errors.EmptyDataError:
            return {
                'error': f'{file_b_name} is empty',
                'details': f'File B ("{file_b_name}") contains no data or only blank lines.\n\nPlease upload a valid CSV file with headers and data.'
            }
        except Exception as e:
            if 'No columns to parse' in str(e):
                return {
                    'error': f'{file_b_name} is empty',
                    'details': f'File B ("{file_b_name}") has no columns or data.\n\nPlease upload a valid CSV file.'
                }
            raise
        
        # Check if DataFrames are empty (no data rows)
        if len(df_a) == 0:
            if len(df_a.columns) > 0:
                return {
                    'error': f'{file_a_name} has no data rows',
                    'details': f'File A ("{file_a_name}") has headers ({", ".join(df_a.columns[:5])}) but no data rows.\n\nPlease add data to the file.'
                }
            else:
                return {
                    'error': f'{file_a_name} is completely empty',
                    'details': f'File A ("{file_a_name}") has no headers and no data.\n\nPlease upload a valid CSV file.'
                }
        
        if len(df_b) == 0:
            if len(df_b.columns) > 0:
                return {
                    'error': f'{file_b_name} has no data rows',
                    'details': f'File B ("{file_b_name}") has headers ({", ".join(df_b.columns[:5])}) but no data rows.\n\nPlease add data to the file.'
                }
            else:
                return {
                    'error': f'{file_b_name} is completely empty',
                    'details': f'File B ("{file_b_name}") has no headers and no data.\n\nPlease upload a valid CSV file.'
                }
        
        # Normalize column names (now handles duplicates)
        df_a = normalize_column_names(df_a)
        df_b = normalize_column_names(df_b)
        
        warning_msg = None
        
        # Handle column mismatch intelligently
        if set(df_a.columns) != set(df_b.columns):
            if len(df_a.columns) == len(df_b.columns):
                warning_msg = f"Column names don't match between {file_a_name} and {file_b_name}. Reloading both files without headers. "
                
                print(f"Column mismatch detected. Attempting to reload without headers...")
                
                df_a = pd.read_csv(file_a_path, delimiter=delim_a, encoding='utf-8', low_memory=False, header=None, nrows=MAX_ROWS)
                df_a.columns = [f'col_{i}' for i in range(len(df_a.columns))]
                
                df_b = pd.read_csv(file_b_path, delimiter=delim_b, encoding='utf-8', low_memory=False, header=None, nrows=MAX_ROWS)
                df_b.columns = [f'col_{i}' for i in range(len(df_b.columns))]
                
                df_a = normalize_column_names(df_a)
                df_b = normalize_column_names(df_b)
                
                has_header_a = False
                has_header_b = False
                warning_msg += "All rows treated as data."
            else:
                return {
                    'error': 'Column count mismatch between files',
                    'details': f'{file_a_name} has {len(df_a.columns)} columns, {file_b_name} has {len(df_b.columns)} columns\n\n{file_a_name} columns: {", ".join(df_a.columns[:10])}\n{file_b_name} columns: {", ".join(df_b.columns[:10])}'
                }
        
        if custom_key_cols:
            key_cols = custom_key_cols
            key_metadata = {'type': 'custom', 'warning': 'User-selected key columns'}
        else:
            key_cols, key_metadata = select_best_key(df_a)
        
        comparison_result = compare_dataframes(df_a, df_b, key_cols, numeric_tolerance)
        comparison_result['key_columns'] = key_cols
        comparison_result['key_metadata'] = key_metadata
        comparison_result['all_columns'] = list(df_a.columns)
        comparison_result['delimiter_a'] = delim_a
        comparison_result['delimiter_b'] = delim_b
        comparison_result['has_header_a'] = has_header_a
        comparison_result['has_header_b'] = has_header_b
        
        if large_file_warning:
            comparison_result['warning'] = large_file_warning
        if warning_msg:
            comparison_result['column_warning'] = warning_msg
        
        return comparison_result
    
    except pd.errors.EmptyDataError as e:
        return {
            'error': 'File is empty',
            'details': 'One of the files is empty or contains only blank lines.\n\nPlease upload valid CSV files with headers and data.'
        }
    except pd.errors.ParserError as e:
        return {
            'error': 'File parsing error',
            'details': f'Could not parse one of the CSV files. The file may be corrupted or have inconsistent formatting.\n\nError details: {str(e)}'
        }
    except UnicodeDecodeError as e:
        return {
            'error': 'File encoding error',
            'details': 'One of the files contains characters that cannot be read. It may be:\n- A binary file (Excel .xlsx)\n- Encoded in a non-UTF-8 format\n\nPlease save as UTF-8 CSV.'
        }
    except Exception as e:
        import traceback
        error_msg = str(e)
        
        if 'No columns to parse' in error_msg:
            return {
                'error': 'File has no columns',
                'details': 'One of the files is empty or contains only whitespace.\n\nPlease upload valid CSV files with headers and data.'
            }
        elif 'Expected' in error_msg and 'fields' in error_msg:
            return {
                'error': 'Inconsistent number of columns',
                'details': f'One of the files has rows with different numbers of columns.\n\nError: {error_msg}\n\nPlease check your CSV files for formatting issues.'
            }
        else:
            return {
                'error': f'Processing error',
                'details': f'An unexpected error occurred:\n\n{error_msg}\n\nPlease check your file format and try again.',
                'traceback': traceback.format_exc()
            }
