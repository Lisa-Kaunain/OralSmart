#!/usr/bin/env python3
"""Dataset Preparation Pipeline for OralSmart AI.

Downloads, validates, and prepares the Annotated Intraoral Image Dataset
for Dental Caries Detection (Zenodo Record 14827784, CC-BY-4.0).
Organizes data into train, validation, and held-out test splits.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import struct
import sys
import time
import zlib

from PIL import Image
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_dataset")

ZENODO_RECORD_URL = "https://zenodo.org/api/records/14827784"
ZENODO_ZIP_URL = "https://zenodo.org/api/records/14827784/files/Benchmarking%20Dataset.zip/content"

CLASS_MAPPING = {
    "normal": {
        "id": 0,
        "name": "No Visible Dental Caries",
        "description": "Healthy dentition with no visible carious lesion or cavitation.",
    },
    "D": {
        "id": 1,
        "name": "Dental Caries (Permanent Dentition)",
        "description": "Visible carious lesion / enamel-dentin decay in secondary (permanent) teeth.",
    },
    "d": {
        "id": 2,
        "name": "Dental Caries (Primary Dentition)",
        "description": "Visible carious lesion / decay in primary (deciduous/baby) teeth.",
    },
}


def build_central_directory_index(cd_bin_path):
    """Parse ZIP Central Directory binary dump into a file index."""
    with open(cd_bin_path, "rb") as f:
        cd_data = f.read()

    entries = {}
    pos = 0
    while pos < len(cd_data):
        if cd_data[pos : pos + 4] != b"\x50\x4b\x01\x02":
            break
        (
            version_made,
            version_needed,
            flags,
            comp_method,
            mod_time,
            mod_date,
            crc32,
            comp_size,
            uncomp_size,
            name_len,
            extra_len,
            comment_len,
            disk_start,
            int_attr,
            ext_attr,
            local_offset,
        ) = struct.unpack("<HHHHHHIIIHHHHHII", cd_data[pos + 4 : pos + 46])

        name = cd_data[pos + 46 : pos + 46 + name_len].decode("utf-8", errors="ignore")
        entries[name] = {
            "offset": local_offset,
            "comp_size": comp_size,
            "uncomp_size": uncomp_size,
            "comp_method": comp_method,
        }
        pos += 46 + name_len + extra_len + comment_len

    return entries


def fetch_file_from_zip(session, entry, url=ZENODO_ZIP_URL, max_retries=5):
    """Fetch and decompress a specific file from the remote ZIP using HTTP range."""
    offset = entry["offset"]
    comp_size = entry["comp_size"]
    method = entry["comp_method"]

    range_header = {"Range": f"bytes={offset}-{offset + 500 + comp_size}", "User-Agent": "curl/8.4.0"}
    
    for attempt in range(max_retries):
        try:
            res = session.get(url, headers=range_header, timeout=30)
            if res.status_code == 429:
                retry_after = int(res.headers.get("Retry-After", 15))
                logger.warning("Zenodo rate limit (429) hit. Waiting %d seconds before retry...", retry_after)
                time.sleep(retry_after + 1)
                continue
            res.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raw = res.content
    if raw[:4] != b"\x50\x4b\x03\x04":
        raise ValueError(f"Invalid local zip header signature at offset {offset}")

    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    data_start = 30 + name_len + extra_len
    comp_data = raw[data_start : data_start + comp_size]

    if method == 8:
        decomp = zlib.decompress(comp_data, -15)
    elif method == 0:
        decomp = comp_data
    else:
        raise ValueError(f"Unsupported compression method: {method}")

    return decomp


def validate_and_save_image(image_bytes, save_path):
    """Validate image readability and write to disk."""
    import io

    with io.BytesIO(image_bytes) as buf:
        with Image.open(buf) as img:
            img.verify()

    with open(save_path, "wb") as f:
        f.write(image_bytes)

    # Return md5 hash
    return hashlib.md5(image_bytes).hexdigest()


def prepare_dataset(dataset_dir, samples_per_split=None, max_workers=6):
    """Main preparation function."""
    dataset_dir = Path(dataset_dir)
    cd_path = dataset_dir / "cd.bin"

    if not cd_path.exists():
        raise FileNotFoundError(f"Central directory file not found: {cd_path}")

    logger.info("Loading central directory index from %s", cd_path)
    zip_index = build_central_directory_index(cd_path)
    logger.info("Total entries indexed from ZIP: %d", len(zip_index))

    # Load COCO annotations for each split
    splits_data = {}
    for split in ["train", "valid", "test"]:
        coco_file = dataset_dir / f"{split}_coco.json"
        if not coco_file.exists():
            raise FileNotFoundError(f"COCO file missing: {coco_file}")
        with open(coco_file, "r", encoding="utf-8") as f:
            splits_data[split] = json.load(f)

    # Find negative images (images in ZIP images/ folder not in COCO)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=12, pool_maxsize=12)
    session.mount("https://", adapter)

    prepared_manifests = {}
    hashes_seen = set()
    total_downloaded = 0

    for split in ["train", "valid", "test"]:
        split_dir = dataset_dir / split
        img_out_dir = split_dir / "images"
        img_out_dir.mkdir(parents=True, exist_ok=True)

        coco = splits_data[split]
        cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}

        # Group annotations by image_id
        annotations_by_img = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            annotations_by_img.setdefault(img_id, []).append(ann)

        # Build list of annotated samples
        annotated_samples = []
        for img_info in coco.get("images", []):
            img_id = img_info["id"]
            anns = annotations_by_img.get(img_id, [])
            if not anns:
                continue

            # Determine dominant/primary condition
            has_D = any(cat_map.get(a["category_id"]) == "D" for a in anns)
            has_d = any(cat_map.get(a["category_id"]) == "d" for a in anns)

            if has_D and has_d:
                class_key = "D"  # Mixed dentition; primary recorded
            elif has_D:
                class_key = "D"
            else:
                class_key = "d"

            sample_entry = {
                "file_name": img_info["file_name"],
                "zip_path": f"Benchmarking Dataset/{split}/images/{img_info['file_name']}",
                "class_key": class_key,
                "label_id": CLASS_MAPPING[class_key]["id"],
                "condition": CLASS_MAPPING[class_key]["name"],
                "width": img_info.get("width"),
                "height": img_info.get("height"),
                "bboxes": [
                    {
                        "category": cat_map.get(a["category_id"], "unknown"),
                        "bbox": a["bbox"],
                        "area": a.get("area"),
                    }
                    for a in anns
                ],
            }
            annotated_samples.append(sample_entry)

        # Find negative samples (images with no caries)
        prefix = f"Benchmarking Dataset/{split}/images/"
        all_split_images = [k for k in zip_index if k.startswith(prefix) and k.lower().endswith((".jpg", ".jpeg", ".png"))]
        annotated_filenames = {s["file_name"] for s in annotated_samples}
        negative_filenames = [
            k[len(prefix):] for k in all_split_images if k[len(prefix):] not in annotated_filenames
        ]

        negative_samples = []
        for fn in negative_filenames:
            negative_samples.append({
                "file_name": fn,
                "zip_path": f"{prefix}{fn}",
                "class_key": "normal",
                "label_id": CLASS_MAPPING["normal"]["id"],
                "condition": CLASS_MAPPING["normal"]["name"],
                "width": None,
                "height": None,
                "bboxes": [],
            })

        logger.info(
            "[%s split] Available: %d annotated (D/d), %d negative (normal)",
            split.upper(),
            len(annotated_samples),
            len(negative_samples),
        )

        # Filter by sample budget if specified
        if samples_per_split:
            budget = samples_per_split.get(split, 80)
            d_samples = [s for s in annotated_samples if s["class_key"] == "D"]
            d_prim_samples = [s for s in annotated_samples if s["class_key"] == "d"]

            n_D = int(budget * 0.45)
            n_d = int(budget * 0.25)
            n_norm = budget - n_D - n_d

            selected = (
                d_samples[:n_D]
                + d_prim_samples[:n_d]
                + negative_samples[:n_norm]
            )
        else:
            selected = annotated_samples + negative_samples

        logger.info("[%s split] Selected %d images for downloading/verification", split.upper(), len(selected))

        # Download images concurrently
        downloaded_manifest = []
        download_queue = []
        for item in selected:
            zip_entry = zip_index.get(item["zip_path"])
            if zip_entry:
                download_queue.append((item, zip_entry))
            else:
                logger.warning("File %s not in ZIP index", item["zip_path"])

        def worker(item_entry):
            item, entry = item_entry
            target_path = img_out_dir / item["file_name"]
            if target_path.exists() and target_path.stat().st_size > 0:
                with open(target_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                return item, target_path, file_hash

            time.sleep(0.3)
            img_bytes = fetch_file_from_zip(session, entry)
            file_hash = validate_and_save_image(img_bytes, target_path)
            return item, target_path, file_hash

        start_t = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, pair) for pair in download_queue]
            for fut in as_completed(futures):
                try:
                    item, out_path, file_hash = fut.result()
                    if file_hash in hashes_seen:
                        logger.debug("Duplicate image content detected: %s", item["file_name"])
                    hashes_seen.add(file_hash)

                    record = dict(item)
                    record["local_path"] = str(out_path.relative_to(dataset_dir)).replace("\\", "/")
                    record["md5"] = file_hash
                    downloaded_manifest.append(record)
                    total_downloaded += 1
                except Exception as err:
                    logger.error("Failed to download/verify %s: %s", item["file_name"], err)

        elapsed = time.time() - start_t
        logger.info(
            "[%s split] Processed %d images in %.2f seconds (%.2f img/sec)",
            split.upper(),
            len(downloaded_manifest),
            elapsed,
            len(downloaded_manifest) / max(elapsed, 0.001),
        )

        # Save split manifest
        manifest_path = split_dir / "annotations.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(downloaded_manifest, f, indent=2)

        prepared_manifests[split] = len(downloaded_manifest)

    # Save overall dataset metadata
    meta = {
        "dataset_name": "Annotated Intraoral Image Dataset for Dental Caries Detection",
        "doi": "10.5281/zenodo.14827784",
        "record_url": ZENODO_RECORD_URL,
        "license": "Creative Commons Attribution 4.0 International (CC-BY-4.0)",
        "source_publication": "Ahmed et al., Scientific Data (Nature Portfolio), 2025",
        "classes": [
            CLASS_MAPPING["normal"],
            CLASS_MAPPING["D"],
            CLASS_MAPPING["d"],
        ],
        "splits_summary": prepared_manifests,
        "total_images_processed": total_downloaded,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(dataset_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Dataset preparation complete! Total images prepared: %d", total_downloaded)
    logger.info("Metadata saved to %s", dataset_dir / "dataset_metadata.json")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare OralSmart AI dental dataset")
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).resolve().parent.parent / "dataset"),
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=120,
        help="Number of training samples to download and prepare",
    )
    parser.add_argument(
        "--valid-samples",
        type=int,
        default=40,
        help="Number of validation samples to download and prepare",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=40,
        help="Number of held-out test samples to download and prepare",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download workers",
    )
    args = parser.parse_args()

    budget = {
        "train": args.train_samples,
        "valid": args.valid_samples,
        "test": args.test_samples,
    }

    prepare_dataset(args.dataset_dir, samples_per_split=budget, max_workers=args.workers)
