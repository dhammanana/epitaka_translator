import os
import glob
import sqlite3

# Define relative paths based on your structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATTERN = os.path.join(BASE_DIR, "../../thai_84000_org/Thai_Mahamakut_Atth/*/Content.dat")
DB_PATH = os.path.join(BASE_DIR, "data/thaimm.sqlite")

def setup_database(db_path):
    """Ensures the destination directory and the SQLite table exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create headings table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS headings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            volume_id TEXT,
            page INTEGER,
            title TEXT
        )
    ''')
    conn.commit()
    return conn

def parse_contents_file(file_path, volume_id):
    """Reads a Contents.dat file and returns parsed rows."""
    parsed_rows = []
    
    # Using utf-8-sig to handle any potential Byte Order Marks (BOM) gracefully
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
                
            if '~~' in line:
                # Split title and page based on your sample: title~~page
                parts = line.split('~~', 1)
                title = parts[0].strip()
                
                try:
                    # Convert '0001' -> 1
                    page = int(parts[1].strip())
                    parsed_rows.append((volume_id, page, title))
                except ValueError:
                    print(f"Warning: Invalid page number format in {file_path} at line {line_num}: '{parts[1]}'")
            else:
                print(f"Warning: Skipping line without delimiter '~~' in {file_path} at line {line_num}")
                
    return parsed_rows

def main():
    print("Initializing database...")
    conn = setup_database(DB_PATH)
    cursor = conn.cursor()
    
    # Find all Contents.dat files matching the path pattern
    target_files = glob.glob(SOURCE_PATTERN)
    
    if not target_files:
        print(f"No Contents.dat files found matching pattern: {SOURCE_PATTERN}")
        return

    print(f"Found {len(target_files)} volume directory/directories to process.")
    
    total_inserted = 0
    
    for file_path in target_files:
        # Extract the volume_id folder name from the path structure
        # ../[volume_id]/Contents.dat -> volume_id is the parent directory name
        volume_id = os.path.basename(os.path.dirname(file_path))
        
        print(f"Processing Volume: {volume_id}...", end="")
        rows_to_insert = parse_contents_file(file_path, volume_id)
        
        if rows_to_insert:
            # Batch insert for efficiency
            cursor.executemany('''
                INSERT INTO headings (volume_id, page, title) 
                VALUES (?, ?, ?)
            ''', rows_to_insert)
            
            total_inserted += len(rows_to_insert)
            print(f" successfully loaded {len(rows_to_insert)} items.")
        else:
            print(" no data found or parsed.")
            
    # Commit changes and close database connection
    conn.commit()
    conn.close()
    
    print("---")
    print(f"Extraction complete! Total records saved to {DB_PATH}: {total_inserted}")

if __name__ == "__main__":
    main()