import sqlite3
import re
from pathlib import Path

def create_rxnorm_db():
    """Load RXNCONSO.RRF and RXNSAT.RRF into medications.db"""

    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Paths relative to script location
    rxnconso_file = script_dir / "RxNorm_full_01052026/rrf/RXNCONSO.RRF"
    rxnsat_file = script_dir / "RxNorm_full_01052026/rrf/RXNSAT.RRF"
    db_path = script_dir / "medications.db"

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading from: {rxnconso_file}")
    print(f"Loading from: {rxnsat_file}")
    print(f"Creating database: {db_path}\n")

    # Check if source files exist
    if not rxnconso_file.exists():
        print(f"ERROR: RXNCONSO.RRF not found at: {rxnconso_file}")
        print("\nPlease download RxNorm from:")
        print("  https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html")
        print(f"\nExtract to: {script_dir}/RxNorm_full_01052026/")
        return
    
    if not rxnsat_file.exists():
        print(f"ERROR: RXNSAT.RRF not found at: {rxnsat_file}")
        print("\nPlease ensure RXNSAT.RRF is in the RxNorm download")
        return
    
    # Create/connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing tables if any
    cursor.execute("DROP TABLE IF EXISTS medications")
    cursor.execute("DROP TABLE IF EXISTS medication_dosages")
    
    # Create medications table
    cursor.execute("""
        CREATE TABLE medications (
            rxcui TEXT,
            name TEXT NOT NULL,
            term_type TEXT,
            name_lower TEXT
        )
    """)
    
    # Create dosages table
    # Allow multiple forms per rxcui+strength combination
    cursor.execute("""
        CREATE TABLE medication_dosages (
            rxcui TEXT,
            strength TEXT,
            dosage_form TEXT,
            PRIMARY KEY (rxcui, strength, dosage_form)
        )
    """)
    
    # Create indexes for fast lookups
    cursor.execute("CREATE INDEX idx_name_lower ON medications(name_lower)")
    cursor.execute("CREATE INDEX idx_rxcui ON medications(rxcui)")
    cursor.execute("CREATE INDEX idx_dosage_rxcui ON medication_dosages(rxcui)")
    cursor.execute("CREATE INDEX idx_dosage_strength ON medication_dosages(strength)")
    
    print("Step 1/2: Loading RXNCONSO.RRF (medication names)...")
    
    batch = []
    count = 0
    
    with open(rxnconso_file, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split('|')
            
            rxcui = fields[0]
            language = fields[1]      # LAT
            term_type = fields[12]    # TTY
            name = fields[14]         # STR
            suppress = fields[16]     # SUPPRESS
            
            # Filter: English, not suppressed, relevant term types
            if (language == 'ENG' and 
                suppress != 'Y' and 
                term_type in {'IN', 'BN', 'SCD', 'SBD', 'GPCK', 'BPCK', 'PIN', 'MIN'}):
                
                batch.append((rxcui, name, term_type, name.lower()))
                count += 1
                
                # Insert in batches
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO medications VALUES (?, ?, ?, ?)",
                        batch
                    )
                    batch = []
                    
                    if count % 10000 == 0:
                        print(f"  Loaded {count:,} medications...")
                        conn.commit()
    
    # Insert remaining
    if batch:
        cursor.executemany("INSERT INTO medications VALUES (?, ?, ?, ?)", batch)
    
    conn.commit()
    
    total_meds = cursor.execute("SELECT COUNT(*) FROM medications").fetchone()[0]
    print(f"✓ Loaded {total_meds:,} medication names")
    
    # Load dosage information from RXNSAT.RRF
    print("\nStep 2/2: Loading RXNSAT.RRF (dosage strengths)...")
    
    # RXNSAT.RRF format:
    # RXCUI|LUI|SUI|RXAUI|STYPE|CODE|ATUI|SATUI|ATN|SAB|ATV|SUPPRESS|CVF|
    
    dosages = {}
    dosage_count = 0
    
    with open(rxnsat_file, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split('|')
            
            rxcui = fields[0]
            atn = fields[8]   # Attribute name
            atv = fields[10]  # Attribute value
            suppress = fields[11]
            
            if suppress == 'Y':
                continue
            
            if rxcui not in dosages:
                dosages[rxcui] = {'strengths': set(), 'forms': set()}
            
            # Extract strength
            if atn == 'RXN_STRENGTH':
                dosages[rxcui]['strengths'].add(atv)
            
            # Extract dosage form
            elif atn in ['RXN_DOSAGE_FORM', 'DF']:
                dosages[rxcui]['forms'].add(atv)
            
            dosage_count += 1
            
            if dosage_count % 100000 == 0:
                print(f"  Processed {dosage_count:,} attributes...")
    
    # Insert dosages into database
    print("  Inserting dosage data...")
    batch = []
    
    for rxcui, data in dosages.items():
        forms = data['forms'] if data['forms'] else {None}
        for strength in data['strengths']:
            for form in forms:
                batch.append((rxcui, strength, form))

                if len(batch) >= 1000:
                    cursor.executemany("""
                        INSERT OR IGNORE INTO medication_dosages
                        VALUES (?, ?, ?)
                    """, batch)
                    batch = []
    
    if batch:
        cursor.executemany("""
            INSERT OR IGNORE INTO medication_dosages 
            VALUES (?, ?, ?)
        """, batch)
    
    conn.commit()
    
    total_dosages = cursor.execute(
        "SELECT COUNT(*) FROM medication_dosages"
    ).fetchone()[0]
    
    print(f"✓ Loaded {total_dosages:,} dosage entries")
    
    # Stats
    db_size = db_path.stat().st_size / 1024 / 1024
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully loaded RxNorm database")
    print(f"  - {total_meds:,} medication names")
    print(f"  - {total_dosages:,} dosage strengths")
    print(f"  - Database size: {db_size:.1f} MB")
    print(f"{'='*60}")
    
    # Test queries
    print("\n--- Testing: Atenolol ---")
    results = cursor.execute("""
        SELECT m.name, d.strength, d.dosage_form
        FROM medications m
        LEFT JOIN medication_dosages d ON m.rxcui = d.rxcui
        WHERE m.name_lower LIKE '%atenolol%'
        AND m.term_type IN ('SCD', 'SBD')
        LIMIT 5
    """).fetchall()
    
    for name, strength, form in results:
        if strength:
            print(f"  {name}: {strength} ({form or 'N/A'})")
        else:
            print(f"  {name}")
    
    print("\n--- Testing: Metformin ---")
    results = cursor.execute("""
        SELECT DISTINCT d.strength
        FROM medications m
        JOIN medication_dosages d ON m.rxcui = d.rxcui
        WHERE m.name_lower LIKE '%metformin%'
        ORDER BY CAST(REPLACE(d.strength, ' MG', '') AS INTEGER)
        LIMIT 10
    """).fetchall()
    
    if results:
        strengths = [r[0] for r in results]
        print(f"  Available strengths: {', '.join(strengths)}")
    
    conn.close()
    print(f"\n✓ Done! Database ready at: {db_path}")

if __name__ == "__main__":
    create_rxnorm_db()