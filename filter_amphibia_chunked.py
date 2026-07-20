"""
Memory-safe version for large BOLD files (multi-GB TSV/FASTA).
Reads the taxonomy TSV in small chunks instead of loading it all at once,
so it won't crash your machine's RAM.

Usage:
    python filter_amphibia_chunked.py
"""

import pandas as pd
from Bio import SeqIO

TSV_PATH = "coidb.info.tsv"
FASTA_PATH = "coidb.clustered.fasta"
OUTPUT_CSV = "amphibia_dataset.csv"

TARGET_CLASS = "Amphibia"
MIN_SEQS_PER_SPECIES = 5
CHUNK_SIZE = 200_000  # rows per chunk — safe for low memory usage

def find_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

def main():
    print("Step 1: Scanning taxonomy TSV in chunks (this will take a few minutes for a 12GB file)...")

    id_to_species = {}
    class_col = None
    id_col = None
    species_col = None

    reader = pd.read_csv(TSV_PATH, sep="\t", low_memory=False, chunksize=CHUNK_SIZE)

    total_rows_scanned = 0
    for i, chunk in enumerate(reader):
        if class_col is None:
            class_col = find_col(chunk.columns, ["class", "Class", "class_name"])
            id_col = find_col(chunk.columns, ["processid", "recordID", "record_id", "id"])
            species_col = find_col(chunk.columns, ["species", "Species", "species_name"])
            if not all([class_col, id_col, species_col]):
                raise ValueError(f"Could not find expected columns. Found: {list(chunk.columns)}")
            print(f"Using columns -> class: {class_col}, id: {id_col}, species: {species_col}")

        total_rows_scanned += len(chunk)
        matches = chunk[chunk[class_col].astype(str).str.strip() == TARGET_CLASS]
        for _, row in matches.iterrows():
            sp = row[species_col]
            if pd.notna(sp) and str(sp).strip() != "":
                id_to_species[str(row[id_col])] = sp

        if i % 10 == 0:
            print(f"  ...scanned {total_rows_scanned:,} rows so far, {len(id_to_species):,} Amphibia records found")

    print(f"\nDone scanning TSV. Total Amphibia records found: {len(id_to_species):,}")

    if len(id_to_species) == 0:
        print("No Amphibia records found — double check the class column value/spelling.")
        return

    wanted_ids = set(id_to_species.keys())

    print("\nStep 2: Scanning FASTA file for matching sequences (streaming, low memory)...")
    records = []
    scanned = 0
    for record in SeqIO.parse(FASTA_PATH, "fasta"):
        scanned += 1
        header_id = record.id.split("|")[0].split(" ")[0]
        if header_id in wanted_ids:
            species = id_to_species.get(header_id)
            records.append({"record_id": header_id, "species": species, "sequence": str(record.seq)})
        if scanned % 500_000 == 0:
            print(f"  ...scanned {scanned:,} sequences, matched {len(records):,} so far")

    print(f"\nDone scanning FASTA. Matched {len(records):,} Amphibia sequences.")

    out_df = pd.DataFrame(records)
    counts = out_df["species"].value_counts()
    keep_species = counts[counts >= MIN_SEQS_PER_SPECIES].index
    filtered_df = out_df[out_df["species"].isin(keep_species)]

    print(f"\nSpecies before filtering low-count classes: {out_df['species'].nunique()}")
    print(f"Species after filtering (>= {MIN_SEQS_PER_SPECIES} seqs each): {filtered_df['species'].nunique()}")
    print(f"Total usable sequences: {len(filtered_df)}")

    filtered_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved clean dataset to {OUTPUT_CSV}")
    print("\nTop 10 species by sequence count:")
    print(filtered_df["species"].value_counts().head(10))


if __name__ == "__main__":
    main()