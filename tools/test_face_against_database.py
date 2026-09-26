"""Temporary read-only diagnostic for matching one image to face templates."""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import psycopg


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from database import get_postgres_config


@dataclass
class Match:
    employee_name: str
    employee_id: str | None
    registration_id: str
    designation: str
    gate_no: str
    image_number: int
    similarity: float


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="One image to compare")
    return parser.parse_args()


def _read_only_connection():
    config = get_postgres_config()
    return psycopg.connect(**config)


def _import_production_face_helpers():
    """Import production face helpers without allowing its DB setup to run."""
    import detection_database

    class ReadOnlyDatabaseImportStub:
        def __init__(self, _database_path):
            pass

    original_database_class = detection_database.DetectionDatabase
    detection_database.DetectionDatabase = ReadOnlyDatabaseImportStub
    try:
        import alcon_stream_new
        from alcon_stream_new import (
            DET_SIZE,
            DET_THRESH,
            RECOGNITION_THRESHOLD,
            initialize_face_model,
            normalize_embedding,
            safe_registration_face_inference,
        )
    finally:
        detection_database.DetectionDatabase = original_database_class

    return {
        "det_size": DET_SIZE,
        "det_thresh": DET_THRESH,
        "threshold": RECOGNITION_THRESHOLD,
        "module": alcon_stream_new,
        "initialize_face_model": initialize_face_model,
        "normalize_embedding": normalize_embedding,
        "safe_registration_face_inference": safe_registration_face_inference,
    }


def _initialize_face_stack_only(helpers):
    """Initialize Buffalo_L using production setup without unrelated YOLO I/O.

    The registration diagnostic needs the exact shared registration detector,
    but should not require downloading/loading the person and vehicle models.
    """
    module = helpers["module"]
    if module.face_app is not None:
        return

    original_yolo = module.YOLO
    module.YOLO = lambda _model_path: SimpleNamespace()
    try:
        helpers["initialize_face_model"]()
    finally:
        module.YOLO = original_yolo


def _decode_embedding(blob, employee_name, image_number, normalize_embedding):
    try:
        if not blob:
            raise ValueError("empty BLOB")
        decoded = np.load(io.BytesIO(blob), allow_pickle=False)
        embedding = np.asarray(decoded, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.size != 512:
            raise ValueError(f"dimension is {embedding.size}, expected 512")
        if not np.all(np.isfinite(embedding)):
            raise ValueError("embedding contains non-finite values")
        if norm <= 0.0:
            raise ValueError("embedding norm is zero")
        return normalize_embedding(embedding).astype(np.float32, copy=False)
    except Exception as error:
        print(
            f"[DB WARNING]\nperson={employee_name}\n"
            f"template={image_number}\nreason={error}"
        )
        return None


def _load_matches(test_embedding, normalize_embedding):
    matches = []
    valid_templates = 0
    invalid_templates = 0
    person_ids = set()

    query = """
        SELECT p.employee_name, p.employee_id, p.registration_id,
               p.designation, p.gate_no, e.image_number, e.embedding
        FROM registered_persons AS p
        INNER JOIN registered_face_embeddings AS e
          ON e.registration_id = p.registration_id
        ORDER BY p.employee_name, e.image_number
    """
    with _read_only_connection() as connection:
        rows = connection.execute(query).fetchall()

    for (
        employee_name,
        employee_id,
        registration_id,
        designation,
        gate_no,
        image_number,
        blob,
    ) in rows:
        person_ids.add(registration_id)
        embedding = _decode_embedding(
            blob, employee_name, image_number, normalize_embedding
        )
        if embedding is None:
            invalid_templates += 1
            continue
        valid_templates += 1
        matches.append(
            Match(
                employee_name=employee_name,
                employee_id=employee_id,
                registration_id=registration_id,
                designation=designation,
                gate_no=gate_no,
                image_number=image_number,
                similarity=float(np.dot(test_embedding, embedding)),
            )
        )

    return len(person_ids), len(rows), valid_templates, invalid_templates, matches


def _print_top_matches(matches):
    print("\n========================================")
    print("TOP MATCHES")
    print("========================================")
    for index, match in enumerate(sorted(matches, key=lambda item: item.similarity, reverse=True)[:10], 1):
        print(
            f"\n{index}. Employee: {match.employee_name}\n"
            f"   Employee ID: {match.employee_id}\n"
            f"   Template: image_{match.image_number}\n"
            f"   Similarity: {match.similarity:.4f}"
        )
    print("\n========================================")


def _print_best_person(best_match, matches, threshold):
    print("\n========================================")
    print("BEST MATCH")
    print("========================================")
    print(
        f"Employee Name : {best_match.employee_name}\n"
        f"Employee ID   : {best_match.employee_id}\n"
        f"Designation   : {best_match.designation}\n"
        f"Gate          : {best_match.gate_no}\n"
        f"Template      : image_{best_match.image_number}\n"
        f"Similarity    : {best_match.similarity:.4f}\n"
        f"Threshold     : {threshold:.4f}\n"
        f"Would match   : {'YES' if best_match.similarity >= threshold else 'NO'}"
    )
    print("\n========================================")
    print("BEST PERSON TEMPLATE SCORES")
    print("========================================")
    print(f"Employee: {best_match.employee_name}\n")
    person_matches = {
        match.image_number: match.similarity
        for match in matches
        if match.registration_id == best_match.registration_id
    }
    for image_number in range(1, 6):
        score = person_matches.get(image_number)
        print(
            f"image_{image_number} : "
            f"{score:.4f}" if score is not None else f"image_{image_number} : NOT AVAILABLE"
        )
    print(f"\nBEST     : {best_match.similarity:.4f}")
    print("========================================")


def main():
    args = _parse_args()
    image_path = args.image
    if image_path is None:
        image_path = Path(input("Enter image path: ").strip().strip('"'))

    try:
        raw = image_path.read_bytes()
    except OSError as error:
        print(f"[TEST] ERROR: Unable to read image: {error}")
        return 1

    helpers = _import_production_face_helpers()
    image, orientation = helpers["module"]._decode_registration_image(raw)
    if image is None:
        print("[TEST] ERROR: Unable to read image")
        return 1

    try:
        _initialize_face_stack_only(helpers)
        faces = helpers["safe_registration_face_inference"](image)
    except Exception as error:
        print(f"[TEST] ERROR: Unable to initialize or run Buffalo_L: {error}")
        return 1

    if len(faces) == 0:
        print("[TEST] ERROR: No face detected")
        return 1
    if len(faces) > 1:
        print("[TEST] ERROR: Multiple faces detected. Please provide an image containing one face.")
        return 1

    try:
        helpers["module"]._validate_registration_face_quality(faces[0], image, 1)
    except ValueError as error:
        print(f"[TEST] ERROR: Registration quality check failed: {error}")
        return 1

    try:
        test_embedding = helpers["normalize_embedding"](faces[0].embedding)
        test_embedding = np.asarray(test_embedding, dtype=np.float32).reshape(-1)
        finite = bool(np.all(np.isfinite(test_embedding)))
        norm = float(np.linalg.norm(test_embedding))
        if test_embedding.size != 512 or not finite or norm <= 0.0:
            raise ValueError("test embedding is not a valid 512-D finite vector")
    except Exception as error:
        print(f"[TEST] ERROR: Unable to generate valid face embedding: {error}")
        return 1

    print(
        f"[TEST]\nImage                  : {image_path}\n"
        f"Decoded dimensions     : {image.shape[1]}x{image.shape[0]}\n"
        f"EXIF orientation       : {orientation}\n"
        f"Detector size          : {helpers['det_size']} (fallback: registration-only)\n"
        f"Detector threshold     : {helpers['det_thresh']:.2f}\n"
        f"Detection score        : {float(faces[0].det_score):.4f}\n"
        f"Face bounding box      : {faces[0].bbox.tolist()}\n"
        "Face detected          : YES\n"
        f"Embedding dimension    : {test_embedding.size}\n"
        f"Embedding dtype        : {test_embedding.dtype}\n"
        f"Embedding norm         : {norm:.8f}\n"
        f"Embedding finite       : {finite}"
    )

    try:
        person_count, template_count, valid_count, invalid_count, matches = _load_matches(
            test_embedding, helpers["normalize_embedding"]
        )
    except Exception as error:
        print(f"[TEST] ERROR: Unable to read database: {error}")
        return 1

    if not matches:
        print("[TEST] ERROR: No valid registered face embeddings found")
        return 1

    matches.sort(key=lambda item: item.similarity, reverse=True)
    _print_top_matches(matches)
    best_match = matches[0]
    threshold = helpers["threshold"]
    _print_best_person(best_match, matches, threshold)

    print("\n========================================")
    print("DIAGNOSTIC RESULT")
    print("========================================")
    print(
        f"Database persons       : {person_count}\n"
        f"Database templates     : {template_count}\n"
        f"Valid templates        : {valid_count}\n"
        f"Invalid templates      : {invalid_count}\n\n"
        "Test embedding        : 512-D\n"
        f"Best similarity        : {best_match.similarity:.4f}\n"
        f"Best employee          : {best_match.employee_name}\n"
        f"Best template          : image_{best_match.image_number}"
    )
    print("\n========================================")
    if best_match.similarity >= threshold:
        print(
            "RESULT: MATCHING IS POSSIBLE.\n"
            "The supplied image matches a registered database template above "
            "the production threshold."
        )
    else:
        print(
            "RESULT: BEST MATCH IS BELOW THRESHOLD.\n"
            "The database is being searched, but the supplied image embedding is not "
            "sufficiently similar to any registered template."
        )
    print(
        "If this is an exact copy of a registration image, its matching "
        "template should normally be extremely close to 1.0. A low camera "
        "score alone does not prove the database is broken."
    )
    print(f"Temporary diagnostic file: {Path(__file__).resolve()}")
    print("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
