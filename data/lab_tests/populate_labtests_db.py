# load_loinc.py
import sqlite3
import csv
from pathlib import Path

def create_loinc_db(loinc_csv_path, db_path='lab_tests.db'):
    """Load LOINC CSV into SQLite database"""
    
    print(f"Loading from: {loinc_csv_path}")
    print(f"Creating database: {db_path}\n")
    
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing table
    cursor.execute("DROP TABLE IF EXISTS loinc_tests")
    
    # Create main LOINC table
    cursor.execute("""
        CREATE TABLE loinc_tests (
            loinc_num TEXT PRIMARY KEY,
            component TEXT,
            property TEXT,
            time_aspect TEXT,
            system TEXT,
            scale_type TEXT,
            method_type TEXT,
            class TEXT,
            long_common_name TEXT,
            shortname TEXT,
            consumer_name TEXT,
            common_test_rank INTEGER,
            relatednames TEXT,
            status TEXT,
            -- Searchable fields
            component_lower TEXT,
            shortname_lower TEXT,
            consumer_name_lower TEXT
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX idx_component ON loinc_tests(component_lower)")
    cursor.execute("CREATE INDEX idx_shortname ON loinc_tests(shortname_lower)")
    cursor.execute("CREATE INDEX idx_consumer ON loinc_tests(consumer_name_lower)")
    cursor.execute("CREATE INDEX idx_rank ON loinc_tests(common_test_rank)")
    cursor.execute("CREATE INDEX idx_class ON loinc_tests(class)")
    
    print("Loading LOINC tests...")
    
    count = 0
    with open(loinc_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        batch = []
        
        for row in reader:
            # Only include active tests
            if row['STATUS'] != 'ACTIVE':
                continue
            
            # Extract relevant fields
            loinc_num = row['LOINC_NUM']
            component = row['COMPONENT']
            long_name = row['LONG_COMMON_NAME']
            shortname = row['SHORTNAME']
            consumer_name = row['CONSUMER_NAME']
            
            # Parse common_test_rank (may be empty)
            try:
                rank = int(row['COMMON_TEST_RANK']) if row['COMMON_TEST_RANK'] else 999999
            except:
                rank = 999999
            
            batch.append((
                loinc_num,
                component,
                row['PROPERTY'],
                row['TIME_ASPCT'],
                row['SYSTEM'],
                row['SCALE_TYP'],
                row['METHOD_TYP'],
                row['CLASS'],
                long_name,
                shortname,
                consumer_name,
                rank,
                row['RELATEDNAMES2'],
                row['STATUS'],
                component.lower() if component else '',
                shortname.lower() if shortname else '',
                consumer_name.lower() if consumer_name else ''
            ))
            
            count += 1
            
            if len(batch) >= 1000:
                cursor.executemany("""
                    INSERT INTO loinc_tests VALUES 
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                batch = []
                
                if count % 10000 == 0:
                    print(f"  Loaded {count:,} tests...")
                    conn.commit()
        
        if batch:
            cursor.executemany("""
                INSERT INTO loinc_tests VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
    
    conn.commit()
    
    # Stats
    total = cursor.execute("SELECT COUNT(*) FROM loinc_tests").fetchone()[0]
    db_size = Path(db_path).stat().st_size / 1024 / 1024
    
    print(f"\n✓ Loaded {total:,} active lab tests")
    print(f"✓ Database size: {db_size:.1f} MB")
    
    # Show WBC example
    print("\n--- Searching for 'WBC' ---")
    wbc_results = cursor.execute("""
        SELECT loinc_num, shortname, long_common_name, common_test_rank
        FROM loinc_tests
        WHERE shortname_lower LIKE '%wbc%'
           OR component_lower LIKE '%leukocyte%'
        ORDER BY common_test_rank
        LIMIT 5
    """).fetchall()
    
    for loinc, short, long, rank in wbc_results:
        print(f"  {loinc}: {short} - Rank: {rank}")
    
    # Show common test categories
    print("\n--- Test Categories ---")
    categories = cursor.execute("""
        SELECT class, COUNT(*) as cnt
        FROM loinc_tests
        WHERE class IS NOT NULL AND class != ''
        GROUP BY class
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()
    
    for cat, cnt in categories:
        print(f"  {cat}: {cnt:,} tests")
    
    conn.close()
    print(f"\n✓ Done! Database ready at: {db_path}")

if __name__ == "__main__":
    # Adjust path to your LOINC file
    loinc_csv = "Loinc_2.81/LoincTable/Loinc.csv"
    create_loinc_db(loinc_csv)