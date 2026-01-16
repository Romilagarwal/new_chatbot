import pandas as pd
import re
from pathlib import Path

def extract_machine_type(machine_name):
    """Extract machine type from machine name"""
    if pd.isna(machine_name) or str(machine_name).strip() == "":
        return "Unknown"

    machine_str = str(machine_name).upper()

    # Common machine type patterns
    type_map = {
        'WELDING': 'Welding Machine',
        'SPOT': 'Spot Welding',
        'CONVEYOR': 'Conveyor System',
        'ROBOT': 'Robotic System',
        'FEEDER': 'Feeder System',
        'PRINTER': 'Printer',
        'CAMERA': 'Vision System',
        'MOTOR': 'Motor',
        'TURN TABLE': 'Assembly Station',
        'FLIPPER': 'Assembly Station',
        'APMT': 'Assembly Station',
        'BMU': 'Battery Management Unit',
        'LOADING': 'Loading System',
        'INSPECTION': 'Inspection Station'
    }

    for keyword, machine_type in type_map.items():
        if keyword in machine_str:
            return machine_type

    return "General Equipment"

def extract_machine_id(text):
    """Extract machine ID from text (e.g., A1, A5, B/D)"""
    if pd.isna(text):
        return "Unknown"

    # Look for patterns like A1, A5, B/D, etc.
    match = re.search(r'[A-Z]\d+|[A-Z]/[A-Z]', str(text))
    if match:
        return match.group(0)

    # Return first word if no pattern found
    words = str(text).split()
    return words[0] if words else "Unknown"

def process_5_column(df):
    """Process 5-column data (most complete)"""
    print("  Processing 5-column sheet...")

    # Expected columns: Machine Name, Sub Assembly/station, Part Issue Details, Root Cause, Corrective Action
    columns = df.columns.tolist()

    result = pd.DataFrame()
    result['Machine Type'] = df.iloc[:, 0].apply(extract_machine_type)
    result['MACHINE'] = df.iloc[:, 0].fillna('Unknown')
    result['Problem Description'] = df.iloc[:, 2].fillna('') + ' ' + df.iloc[:, 1].fillna('')
    result['Root Cause'] = df.iloc[:, 3].fillna('Not specified')
    result['Action Taken'] = df.iloc[:, 4].fillna('See problem description')

    return result.dropna(subset=['Problem Description'])

def process_4_column(df):
    """Process 4-column data"""
    print("  Processing 4-column sheet...")

    # Expected: B/D, Phenomena, Corrective Actions, Root Cause
    result = pd.DataFrame()
    result['Machine Type'] = df.iloc[:, 0].apply(lambda x: extract_machine_type(str(x)))
    result['MACHINE'] = df.iloc[:, 0].fillna('Unknown')
    result['Problem Description'] = df.iloc[:, 1].fillna('Unknown issue')
    result['Root Cause'] = df.iloc[:, 3].fillna('Not specified')
    result['Action Taken'] = df.iloc[:, 2].fillna('See problem description')

    return result.dropna(subset=['Problem Description'])

def process_3_column(df):
    """Process 3-column data (largest dataset)"""
    print("  Processing 3-column sheet...")

    # Expected: Machine, Failure Analysis, Corrective Action
    result = pd.DataFrame()
    result['Machine Type'] = df.iloc[:, 0].apply(extract_machine_type)
    result['MACHINE'] = df.iloc[:, 0].fillna('Unknown')
    result['Problem Description'] = df.iloc[:, 1].fillna('Unknown issue')
    result['Root Cause'] = 'Identified during failure analysis'
    result['Action Taken'] = df.iloc[:, 2].fillna('See problem description')

    return result.dropna(subset=['Problem Description'])

def process_2_column(df):
    """Process 2-column data"""
    print("  Processing 2-column sheet...")

    # Expected: B/D Phenomena, Corrective Actions
    result = pd.DataFrame()

    # Extract machine ID from phenomena text
    result['MACHINE'] = df.iloc[:, 0].apply(extract_machine_id)
    result['Machine Type'] = result['MACHINE'].apply(extract_machine_type)
    result['Problem Description'] = df.iloc[:, 0].fillna('Unknown issue')
    result['Root Cause'] = 'Not specified'
    result['Action Taken'] = df.iloc[:, 1].fillna('See problem description')

    return result.dropna(subset=['Problem Description'])

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""

    text = str(text).strip()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    return text

def unify_dataset(excel_file, output_csv='unified_dataset.csv'):
    """Main function to unify all sheets"""

    print("\n" + "="*60)
    print("DATASET UNIFICATION")
    print("="*60)
    print(f"Input: {excel_file}")
    print(f"Output: {output_csv}\n")

    try:
        # Read all sheets
        excel_data = pd.read_excel(excel_file, sheet_name=None)

        print(f"Found {len(excel_data)} sheets\n")

        all_data = []

        for sheet_name, df in excel_data.items():
            print(f"Processing sheet: '{sheet_name}'")
            print(f"  Shape: {df.shape}")

            # Skip empty sheets
            if df.empty:
                print("  Skipped: Empty sheet\n")
                continue

            num_cols = len(df.columns)
            print(f"  Columns: {num_cols}")

            # Process based on column count
            if num_cols == 5:
                processed = process_5_column(df)
            elif num_cols == 4:
                processed = process_4_column(df)
            elif num_cols == 3:
                processed = process_3_column(df)
            elif num_cols == 2:
                processed = process_2_column(df)
            else:
                print(f"  Skipped: Unsupported column count ({num_cols})\n")
                continue

            print(f"  Processed: {len(processed)} rows\n")
            all_data.append(processed)

        # Combine all data
        if not all_data:
            print("❌ No data to process!")
            return

        unified_df = pd.concat(all_data, ignore_index=True)

        # Clean all text columns
        for col in unified_df.columns:
            unified_df[col] = unified_df[col].apply(clean_text)

        # Remove duplicates
        before_dedup = len(unified_df)
        unified_df = unified_df.drop_duplicates(
            subset=['MACHINE', 'Problem Description'],
            keep='first'
        )
        after_dedup = len(unified_df)

        # Save to CSV
        unified_df.to_csv(output_csv, index=False, encoding='utf-8')

        print("="*60)
        print("✓ UNIFICATION COMPLETE")
        print("="*60)
        print(f"Total rows processed: {before_dedup}")
        print(f"Duplicates removed: {before_dedup - after_dedup}")
        print(f"Final dataset size: {after_dedup} rows")
        print(f"\nOutput saved to: {output_csv}")

        # Show sample
        print("\nSample rows:")
        print("-"*60)
        print(unified_df.head(3).to_string())

        # Show statistics
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Unique machine types: {unified_df['Machine Type'].nunique()}")
        print(f"Unique machines: {unified_df['MACHINE'].nunique()}")
        print(f"\nTop 5 machine types:")
        print(unified_df['Machine Type'].value_counts().head())
        print("="*60 + "\n")

        return unified_df

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Usage
    excel_file = input("Enter path to your Excel file: ").strip().strip('"')

    if not Path(excel_file).exists():
        print(f"❌ File not found: {excel_file}")
    else:
        unify_dataset(excel_file)
