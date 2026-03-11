import copy
import itertools
import os
import shutil
from pathlib import Path
from typing import List, Optional

from utils.read_write_model import read_model


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_image_relpaths(images_dir: Path) -> List[str]:
    rels = []
    for p in images_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        rels.append(p.relative_to(images_dir).as_posix())
    rels.sort()
    return rels


def _write_pairs_file(image_list: List[str], pairs_path: Path, mode: str = "exhaustive", window: int = 5) -> int:
    mode = (mode or "exhaustive").lower()
    n = len(image_list)
    total = 0
    with open(pairs_path, "w", encoding="utf-8") as f:
        if mode == "window":
            w = max(1, int(window))
            for i in range(n):
                for j in range(i + 1, min(n, i + 1 + w)):
                    f.write(f"{image_list[i]} {image_list[j]}\n")
                    total += 1
        else:
            for i, j in itertools.combinations(range(n), 2):
                f.write(f"{image_list[i]} {image_list[j]}\n")
                total += 1
    return total


def _safe_link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except Exception:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _resolve_model_dir(sparse_root: Path) -> Optional[Path]:
    if not sparse_root.exists():
        return None

    def has_model(p: Path) -> bool:
        return (
            (p / "cameras.bin").exists() or (p / "cameras.txt").exists()
        ) and (
            (p / "images.bin").exists() or (p / "images.txt").exists()
        ) and (
            (p / "points3D.bin").exists() or (p / "points3D.txt").exists()
        )

    # hloc/pycolmap may write either to sparse/, sparse/0/, or nested dirs.
    priority = [sparse_root / "0", sparse_root]
    for c in priority:
        if has_model(c):
            return c

    # Fallback: recursive search for any valid model directory.
    for marker in ("points3D.bin", "points3D.txt"):
        for marker_path in sparse_root.rglob(marker):
            c = marker_path.parent
            if has_model(c):
                return c
    return None


def _run_hloc_extract(extract_features, conf, images_dir: Path, image_list: List[str], feature_path: Path):
    try:
        out = extract_features.main(
            conf=conf,
            image_dir=images_dir,
            image_list=image_list,
            feature_path=feature_path,
            overwrite=True,
        )
        return Path(out) if out is not None else feature_path
    except TypeError:
        out = extract_features.main(conf, images_dir, image_list=image_list, feature_path=feature_path)
        return Path(out) if out is not None else feature_path


def _run_hloc_match(match_features, conf, pairs_path: Path, features_path: Path, matches_path: Path):
    try:
        out = match_features.main(
            conf=conf,
            pairs=pairs_path,
            features=features_path,
            matches=matches_path,
            overwrite=True,
        )
        return Path(out) if out is not None else matches_path
    except TypeError:
        out = match_features.main(conf, pairs_path, features=features_path, matches=matches_path)
        return Path(out) if out is not None else matches_path


def _run_hloc_reconstruction(reconstruction, sfm_dir: Path, images_dir: Path, pairs_path: Path, features_path: Path, matches_path: Path):
    # hloc API has changed across versions, so try a few compatible signatures.
    try:
        return reconstruction.main(
            sfm_dir=sfm_dir,
            image_dir=images_dir,
            pairs=pairs_path,
            features=features_path,
            matches=matches_path,
            image_list=None,
        )
    except TypeError:
        try:
            return reconstruction.main(
                sfm_dir,
                images_dir,
                pairs_path,
                features_path,
                matches_path,
            )
        except TypeError:
            return reconstruction.main(
                sfm_dir,
                images_dir,
                pairs_path,
                features_path,
                matches_path,
                None,
            )


def _safe_tag(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def _build_attempts(primary_extractor: str, primary_matcher: str, extractor_keys, matcher_keys):
    attempts = []

    def add(ext_name: str, match_name: str):
        if ext_name in extractor_keys and match_name in matcher_keys:
            pair = (ext_name, match_name)
            if pair not in attempts:
                attempts.append(pair)

    add(primary_extractor, primary_matcher)

    # Fallbacks that do not require SuperGluePretrainedNetwork.
    add("disk", "disk+lightglue")
    add("aliked-n16", "aliked+lightglue")
    add("sift", "sift+lightglue")
    add("sift", "NN-ratio")
    return attempts


def run_full_colmap_style_deep_reconstruction(
    source_path: str,
    images_subdir: str = "images",
    output_root: Optional[str] = None,
    extractor: str = "disk",
    matcher: str = "disk+lightglue",
    pair_mode: str = "exhaustive",
    pair_window: int = 5,
    max_image_size: int = 1600,
    overwrite: bool = False,
) -> Optional[str]:
    """
    Run full COLMAP-style SfM with deep features + deep matching + triangulation/BA.

    Implementation uses hloc (feature extraction/matching/reconstruction wrapper)
    backed by pycolmap for mapper/triangulation.

    Returns:
        Path to a prepared source directory with:
        - <prepared>/images (symlink/copy to original images dir)
        - <prepared>/sparse/0 (reconstructed model)
        Returns None on failure.
    """
    src = Path(source_path).resolve()
    images_dir = (src / images_subdir).resolve()
    if not images_dir.exists():
        print(f"[DeepCOLMAP][ERROR] Images directory not found: {images_dir}")
        return None

    image_list = _list_image_relpaths(images_dir)
    if len(image_list) < 2:
        print("[DeepCOLMAP][ERROR] Need at least two images for SfM.")
        return None

    try:
        from hloc import extract_features, match_features, reconstruction
    except Exception as exc:
        print(f"[DeepCOLMAP][ERROR] Missing dependencies for full deep COLMAP pipeline: {exc}")
        print("[DeepCOLMAP][HINT] Install with: pip install hloc pycolmap")
        return None

    if output_root is None:
        output_root = str(src / "deep_colmap")
    work_dir = Path(output_root).resolve()
    if overwrite and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = work_dir / "pairs-sfm.txt"
    features_path = work_dir / "features.h5"
    matches_path = work_dir / "matches.h5"
    sfm_dir = work_dir / "sparse"

    num_pairs = _write_pairs_file(image_list, pairs_path, mode=pair_mode, window=pair_window)
    print(f"[DeepCOLMAP] Pair list: {num_pairs} pairs ({pair_mode})")

    extractor_keys = list(extract_features.confs.keys())
    matcher_keys = list(match_features.confs.keys())
    attempts = _build_attempts(extractor, matcher, extractor_keys, matcher_keys)
    if not attempts:
        print(f"[DeepCOLMAP][ERROR] No valid extractor/matcher attempts for '{extractor}' + '{matcher}'.")
        print(f"[DeepCOLMAP][HINT] Available extractors: {extractor_keys}")
        print(f"[DeepCOLMAP][HINT] Available matchers: {matcher_keys}")
        return None

    success = False
    last_exc = None
    for idx, (ext_name, match_name) in enumerate(attempts):
        extractor_conf = copy.deepcopy(extract_features.confs.get(ext_name, {}))
        matcher_conf = copy.deepcopy(match_features.confs.get(match_name, {}))
        if not extractor_conf or not matcher_conf:
            continue

        if max_image_size and max_image_size > 0:
            pre = extractor_conf.get("preprocessing", {})
            if isinstance(pre, dict):
                pre["resize_max"] = int(max_image_size)
                extractor_conf["preprocessing"] = pre

        if idx == 0:
            print(f"[DeepCOLMAP] Extractor: {ext_name} | Matcher: {match_name}")
        else:
            print(f"[DeepCOLMAP] Retrying with fallback extractor/matcher: {ext_name} | {match_name}")

        features_try = work_dir / f"features-{_safe_tag(ext_name)}.h5"
        matches_try = work_dir / f"matches-{_safe_tag(ext_name)}-{_safe_tag(match_name)}.h5"

        try:
            features_path = _run_hloc_extract(extract_features, extractor_conf, images_dir, image_list, features_try)
            matches_path = _run_hloc_match(match_features, matcher_conf, pairs_path, features_path, matches_try)
            recon_out = _run_hloc_reconstruction(reconstruction, sfm_dir, images_dir, pairs_path, features_path, matches_path)
            success = True
            break
        except Exception as exc:
            last_exc = exc
            print(f"[DeepCOLMAP][WARNING] Attempt failed ({ext_name} + {match_name}): {exc}")
            continue

    if not success:
        print("[DeepCOLMAP][ERROR] Deep COLMAP stage failed for all attempted configs.")
        if last_exc is not None:
            print(f"[DeepCOLMAP][ERROR] Last error: {last_exc}")
        print("[DeepCOLMAP][HINT] Verify hloc/pycolmap and matcher dependencies are installed.")
        return None

    model_dir = _resolve_model_dir(sfm_dir)
    if model_dir is None:
        # Some hloc versions return reconstruction objects even if files are not
        # materialized where expected. Try writing first reconstruction explicitly.
        rec_obj = None
        try:
            if isinstance(recon_out, dict) and len(recon_out) > 0:
                rec_obj = next(iter(recon_out.values()))
            elif isinstance(recon_out, (list, tuple)) and len(recon_out) > 0:
                rec_obj = recon_out[0]
            else:
                rec_obj = recon_out
        except Exception:
            rec_obj = None

        if rec_obj is not None and hasattr(rec_obj, "write"):
            try:
                fallback_model_dir = sfm_dir / "0"
                fallback_model_dir.mkdir(parents=True, exist_ok=True)
                rec_obj.write(str(fallback_model_dir))
                model_dir = _resolve_model_dir(sfm_dir)
                if model_dir is not None:
                    print(f"[DeepCOLMAP] Materialized reconstruction at: {model_dir}")
            except Exception as exc:
                print(f"[DeepCOLMAP][WARNING] Could not materialize reconstruction object: {exc}")

    if model_dir is None:
        print("[DeepCOLMAP][ERROR] Reconstruction did not produce a valid sparse model.")
        return None

    prepared_source = work_dir / "source_for_dps"
    prepared_sparse0 = prepared_source / "sparse" / "0"
    prepared_sparse0.mkdir(parents=True, exist_ok=True)

    # Link/copy original images so dps can load RGBs from the same relative path.
    _safe_link_or_copy(images_dir, prepared_source / images_subdir)

    # Copy both .bin and .txt if present.
    for name in (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
    ):
        src_file = model_dir / name
        if src_file.exists():
            shutil.copy2(src_file, prepared_sparse0 / name)

    try:
        cams, imgs, pts = read_model(str(prepared_sparse0), ext="")
        n_cam = 0 if cams is None else len(cams)
        n_img = 0 if imgs is None else len(imgs)
        n_pts = 0 if pts is None else len(pts)
        print(f"[DeepCOLMAP] Reconstructed model: cameras={n_cam}, images={n_img}, points3D={n_pts}")
    except Exception:
        print("[DeepCOLMAP][WARNING] Could not parse reconstructed model summary.")

    print(f"[DeepCOLMAP] Prepared source for dps at: {prepared_source}")
    return str(prepared_source)
