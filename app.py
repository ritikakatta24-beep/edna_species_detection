"""
FastAPI backend for eDNA species prediction.

Loads the trained k-NN model (knn_species_model.joblib) and exposes a
/predict endpoint that takes a raw DNA sequence and returns the predicted
species with a confidence score.

Usage:
    pip install fastapi uvicorn scikit-learn joblib numpy
    uvicorn api:app --reload
"""

import io
import os
import csv
import json
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel


app = FastAPI(title="eDNA Species Prediction API")

# ---- Load the trained model once at startup ----
print("Loading trained model...")
bundle = joblib.load("knn_species_model.joblib")
model = bundle["model"]
kmer_index = bundle["kmer_index"]
K = bundle["k"]
print(f"Model loaded. k={K}, vocabulary size={len(kmer_index)}")

# ---- Load species description dataset ----
species_descriptions_db = {}
if os.path.exists("amphibia_species_descriptions.json"):
    print("Loading species description dataset...")
    with open("amphibia_species_descriptions.json", "r", encoding="utf-8") as f:
        desc_list = json.load(f)
        for item in desc_list:
            species_descriptions_db[item["species"]] = item
    print(f"Loaded descriptions for {len(species_descriptions_db)} species.")


def sequence_to_kmer_vector(seq: str, k: int, kmer_index: dict) -> np.ndarray:
    """Same encoding logic used during training -- must match exactly."""
    vec = np.zeros(len(kmer_index), dtype=np.float32)
    seq = seq.upper()
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if kmer in kmer_index:
            vec[kmer_index[kmer]] += 1
            total += 1
    if total > 0:
        vec /= total
    return vec


def parse_fasta(text: str) -> list[dict]:
    """Parse raw FASTA text into a list of {'id': ..., 'sequence': ...} records."""
    records = []
    current_id = None
    current_seq_parts: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            # Save the previous record before starting a new one
            if current_id is not None:
                records.append({
                    "id": current_id,
                    "sequence": "".join(current_seq_parts),
                })
            current_id = line[1:].strip()  # drop the leading '>'
            current_seq_parts = []
        else:
            current_seq_parts.append(line)

    # Save the last record after the loop ends
    if current_id is not None:
        records.append({
            "id": current_id,
            "sequence": "".join(current_seq_parts),
        })

    return records


def parse_csv(text: str) -> list[dict]:
    """Parse CSV text with columns: record_id, species, sequence.
    'species' is treated as the known/actual label (optional -- may be blank),
    used later to check whether the model's prediction matches it."""
    reader = csv.DictReader(io.StringIO(text))

    # Make column matching forgiving of case/whitespace differences
    fieldnames = reader.fieldnames or []
    normalized = {name.strip().lower(): name for name in fieldnames}

    id_col = normalized.get("record_id")
    species_col = normalized.get("species")
    seq_col = normalized.get("sequence")

    if id_col is None or seq_col is None:
        raise ValueError(
            "CSV must contain at least 'record_id' and 'sequence' columns "
            f"(found columns: {fieldnames})"
        )

    records = []
    for row in reader:
        records.append({
            "id": (row.get(id_col) or "").strip(),
            "actual_species": (row.get(species_col) or "").strip() if species_col else "",
            "sequence": (row.get(seq_col) or "").strip(),
        })
    return records


def run_prediction(sequence: str, confidence_threshold: float, confusion_gap: float) -> "PredictionResponse":
    """Core prediction logic, shared by /predict and /predict-batch."""
    vector = sequence_to_kmer_vector(sequence, K, kmer_index)
    vector = vector.reshape(1, -1)

    probabilities = model.predict_proba(vector)[0]
    class_names = model.classes_

    paired = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)

    top_candidates = [
        CandidateSpecies(
            species=sp,
            confidence=round(float(prob), 4),
            species_details=SpeciesDetails(**species_descriptions_db[sp]) if sp in species_descriptions_db else None,
        )
        for sp, prob in paired[:5]
        if prob > 0
    ]

    top_species, top_confidence = paired[0]
    is_confident = top_confidence >= confidence_threshold

    is_confused = False
    if len(top_candidates) >= 2:
        gap = top_candidates[0].confidence - top_candidates[1].confidence
        is_confused = gap <= confusion_gap

    is_uncertain = (not is_confident) or is_confused
    candidates_to_return = top_candidates if is_uncertain else []

    species_details = None
    if is_confident and top_species in species_descriptions_db:
        species_details = SpeciesDetails(**species_descriptions_db[top_species])

    return PredictionResponse(
        predicted_species=top_species if is_confident else "Unknown / no confident match",
        confidence=round(float(top_confidence), 4),
        is_confident=is_confident,
        is_confused=is_confused,
        top_candidates=candidates_to_return,
        species_details=species_details,
    )


class SequenceRequest(BaseModel):
    sequence: str
    confidence_threshold: float = 0.3  # below this, we return "unknown" instead of guessing
    confusion_gap: float = 0.15        # if top species are within this gap of each other, flag as "confused"


class SpeciesDetails(BaseModel):
    species: str
    common_name: str | None = None
    order: str | None = None
    family: str | None = None
    genus: str | None = None
    habitat: str | None = None
    geographic_range: str | None = None
    conservation_status: str | None = None
    description: str | None = None
    sequence_count: int | None = None


class CandidateSpecies(BaseModel):
    species: str
    confidence: float
    species_details: SpeciesDetails | None = None



class PredictionResponse(BaseModel):
    predicted_species: str
    confidence: float
    is_confident: bool
    is_confused: bool
    top_candidates: list[CandidateSpecies]
    species_details: SpeciesDetails | None = None


class BatchResultItem(BaseModel):
    id: str
    result: PredictionResponse | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    total_sequences: int
    results: list[BatchResultItem]


class CsvBatchResultItem(BaseModel):
    id: str
    actual_species: str | None = None
    result: PredictionResponse | None = None
    match: bool | None = None  # True/False if actual_species was known, else None
    error: str | None = None


class CsvBatchPredictionResponse(BaseModel):
    total_sequences: int
    correct: int
    accuracy: float | None = None  # None if no rows had a known actual_species
    results: list[CsvBatchResultItem]


@app.get("/")
def root():
    return {"status": "ok", "message": "eDNA species prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_species(request: SequenceRequest):
    return run_prediction(request.sequence, request.confidence_threshold, request.confusion_gap)


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_species_batch(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.3,
    confusion_gap: float = 0.15,
):
    """Upload a FASTA file (.fasta/.fa/.fna) containing one or more sequences.
    Each record is run through the same prediction logic as /predict, and
    results are returned in the same order as they appear in the file."""
    raw_bytes = await file.read()
    text = raw_bytes.decode("utf-8", errors="ignore")

    records = parse_fasta(text)

    results: list[BatchResultItem] = []
    for record in records:
        record_id = record["id"]
        seq = record["sequence"]

        if not seq:
            results.append(BatchResultItem(id=record_id, error="Empty sequence"))
            continue

        try:
            prediction = run_prediction(seq, confidence_threshold, confusion_gap)
            results.append(BatchResultItem(id=record_id, result=prediction))
        except Exception as exc:
            results.append(BatchResultItem(id=record_id, error=str(exc)))

    return BatchPredictionResponse(total_sequences=len(records), results=results)


@app.post("/predict-batch-csv", response_model=CsvBatchPredictionResponse)
async def predict_species_batch_csv(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.3,
    confusion_gap: float = 0.15,
):
    """Upload a CSV file with columns: record_id, species, sequence
    (this matches the format of the original training dataset -- no '>' needed).
    'species' is treated as the known/actual label -- if present, each
    prediction is checked against it and an overall accuracy is returned."""
    raw_bytes = await file.read()
    text = raw_bytes.decode("utf-8", errors="ignore")

    try:
        records = parse_csv(text)
    except ValueError as exc:
        return CsvBatchPredictionResponse(
            total_sequences=0,
            correct=0,
            accuracy=None,
            results=[CsvBatchResultItem(id="", error=str(exc))],
        )

    results: list[CsvBatchResultItem] = []
    correct = 0
    known_count = 0  # how many rows actually had an actual_species to compare against

    for record in records:
        record_id = record["id"]
        actual_species = record["actual_species"] or None
        seq = record["sequence"]

        if not seq:
            results.append(CsvBatchResultItem(
                id=record_id, actual_species=actual_species, error="Empty sequence"
            ))
            continue

        try:
            prediction = run_prediction(seq, confidence_threshold, confusion_gap)

            match = None
            if actual_species is not None:
                known_count += 1
                match = prediction.predicted_species == actual_species
                if match:
                    correct += 1

            results.append(CsvBatchResultItem(
                id=record_id,
                actual_species=actual_species,
                result=prediction,
                match=match,
            ))
        except Exception as exc:
            results.append(CsvBatchResultItem(
                id=record_id, actual_species=actual_species, error=str(exc)
            ))

    accuracy = round(correct / known_count, 4) if known_count > 0 else None

    return CsvBatchPredictionResponse(
        total_sequences=len(records),
        correct=correct,
        accuracy=accuracy,
        results=results,
    )


@app.get("/species")
def list_species():
    """List all available species in the description database."""
    return {
        "total_species": len(species_descriptions_db),
        "species_list": sorted(list(species_descriptions_db.keys())),
    }


@app.get("/species/{species_name}", response_model=SpeciesDetails)
def get_species_details(species_name: str):
    """Retrieve full description and metadata for a specific amphibian species."""
    if species_name not in species_descriptions_db:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Species '{species_name}' not found in database.")
    return SpeciesDetails(**species_descriptions_db[species_name])