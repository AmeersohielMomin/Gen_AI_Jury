"""FastAPI application serving the GenAI Jury presentation and live demo UI."""

from __future__ import annotations

import csv
import io
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.report_builder import DISCLAIMER_TEXT, build_preliminary_report
from app.schemas import CGANSampleRequest, QuizScoreRequest, ReportExportRequest
from src.constants import (
    ARM_ORDER,
    CGAN_DIR,
    PROJECT_DIR,
    REALISM_DIR,
    VALID_GAN_EPOCHS,
)
from src.evaluation_cache import load_evaluation_cache
from src.explainability import ModelRegistry
from src.roadmap_specs import get_roadmap_payload

STATIC_DIR = Path(__file__).resolve().parent / "static"
PANEL_FILENAME_RE = re.compile(r"^panel_(0[1-9]|1[0-9]|2[0-4])\.png$")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="GenAI Jury — Skin-Lesion cGAN & Healthcare Roadmap Studio",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_eval_cache() -> dict[str, Any]:
    return load_evaluation_cache()


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    return ModelRegistry()


def _sample_lookup() -> dict[str, dict[str, Any]]:
    cache = get_eval_cache()
    return {item["sample_id"]: item for item in cache.get("samples", [])}


def _load_quiz_answer_key() -> list[dict[str, str]]:
    key_path = REALISM_DIR / "PRIVATE_answer_key.csv"
    if not key_path.exists():
        raise HTTPException(status_code=500, detail="Realism survey answer key is missing.")
    rows: list[dict[str, str]] = []
    with key_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "panel_file": row["panel_file"],
                "panel_id": Path(row["panel_file"]).stem,
                "source": row["source"],
            })
    return rows


@app.get("/", response_class=HTMLResponse)
def serve_home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    cache = get_eval_cache()
    return {
        "project_title": "GenAI Jury: Topic 146 — Conditional GAN Skin-Lesion Augmentation",
        "disclaimer": DISCLAIMER_TEXT,
        "generated_at": cache["generated_at"],
        "seed": cache["seed"],
        "target_class": cache["target_class"],
        "class_names": cache["class_names"],
        "diagnosis_full_names": cache["diagnosis_full_names"],
        "splits": cache["splits"],
        "arm_order": cache["arm_order"],
        "arm_summaries": {
            arm: {
                "label": cache["arms"][arm]["label"],
                "short": cache["arms"][arm]["short"],
                "description": cache["arms"][arm]["description"],
                "metrics": cache["arms"][arm]["metrics"],
            }
            for arm in cache["arm_order"]
        },
        "cgan_audit_summary": {
            "epochs_completed": cache["cgan_audit"]["epochs_completed"],
            "candidates_total": cache["cgan_audit"]["candidates_total"],
            "candidates_accepted": cache["cgan_audit"]["candidates_accepted"],
            "synthetic_detector_auc": cache["cgan_audit"]["synthetic_detector_auc"],
        },
    }


@app.get("/api/jury-lab")
def api_jury_lab() -> dict[str, Any]:
    cache = get_eval_cache()
    return {
        "class_names": cache["class_names"],
        "diagnosis_full_names": cache["diagnosis_full_names"],
        "arm_order": cache["arm_order"],
        "arms": cache["arms"],
        "cgan_audit": cache["cgan_audit"],
    }


@app.get("/api/samples")
def api_samples() -> dict[str, Any]:
    cache = get_eval_cache()
    items = []
    for item in cache.get("samples", []):
        items.append({
            **{k: v for k, v in item.items() if k != "rel_path"},
            "image_url": f"/api/media/sample/{item['sample_id']}",
        })
    return {"samples": items}


@app.get("/api/media/sample/{sample_id}")
def api_sample_image(sample_id: str) -> FileResponse:
    sample = _sample_lookup().get(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Unknown sample_id.")
    resolved = (PROJECT_DIR / sample["rel_path"]).resolve()
    if not resolved.is_relative_to(PROJECT_DIR.resolve()) or not resolved.exists():
        raise HTTPException(status_code=404, detail="Sample file unavailable.")
    return FileResponse(resolved)


@app.get("/api/media/epoch/{epoch}")
def api_epoch_image(epoch: int) -> FileResponse:
    if epoch not in VALID_GAN_EPOCHS:
        raise HTTPException(status_code=404, detail="Epoch grid not available.")
    path = CGAN_DIR / f"samples_epoch_{epoch:03d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Epoch grid image missing.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/quiz/panels")
def api_quiz_panels() -> dict[str, Any]:
    entries = _load_quiz_answer_key()
    return {
        "total": len(entries),
        "instructions": (
            "Classify each shuffled 128px/crop panel as Real (ISIC dermoscopy) or Synthetic (cGAN). "
            "Pay attention to checkerboard grids, corner black blocks, hair strands, and pigment texture."
        ),
        "panels": [
            {
                "index": idx,
                "panel_id": entry["panel_id"],
                "filename": entry["panel_file"],
                "image_url": f"/api/quiz/image/{entry['panel_file']}",
            }
            for idx, entry in enumerate(entries, start=1)
        ],
    }


@app.get("/api/quiz/image/{panel_name}")
def api_quiz_image(panel_name: str) -> FileResponse:
    if not PANEL_FILENAME_RE.fullmatch(panel_name):
        raise HTTPException(status_code=404, detail="Panel image not found.")
    path = (REALISM_DIR / panel_name).resolve()
    if not path.is_relative_to(REALISM_DIR.resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail="Panel image missing.")
    return FileResponse(path, media_type="image/png")


@app.post("/api/quiz/score")
def api_quiz_score(request: QuizScoreRequest) -> dict[str, Any]:
    answer_key = _load_quiz_answer_key()
    guess_map = {g.panel_id: g.choice for g in request.guesses}

    revealed = []
    correct_count = 0
    decided_count = 0
    unsure_count = 0
    real_total = 0
    real_correct = 0
    synth_total = 0
    synth_correct = 0

    for entry in answer_key:
        pid = entry["panel_id"]
        actual = entry["source"]
        guess = guess_map.get(pid, "unsure")
        is_correct = guess == actual
        if guess == "unsure":
            unsure_count += 1
        else:
            decided_count += 1
            if is_correct:
                correct_count += 1

        if actual == "real":
            real_total += 1
            if is_correct:
                real_correct += 1
        else:
            synth_total += 1
            if is_correct:
                synth_correct += 1

        revealed.append({
            "panel_id": pid,
            "image_url": f"/api/quiz/image/{entry['panel_file']}",
            "actual_source": actual,
            "your_guess": guess,
            "correct": is_correct,
        })

    cache = get_eval_cache()
    detector_auc = cache["cgan_audit"]["synthetic_detector_auc"]
    return {
        "total_panels": len(answer_key),
        "decided_count": decided_count,
        "unsure_count": unsure_count,
        "correct_count": correct_count,
        "overall_accuracy": round(correct_count / len(answer_key), 4) if answer_key else 0.0,
        "decided_accuracy": round(correct_count / decided_count, 4) if decided_count else 0.0,
        "real_correct": real_correct,
        "real_total": real_total,
        "synthetic_correct": synth_correct,
        "synthetic_total": synth_total,
        "statistical_detector_auc": detector_auc,
        "revealed": revealed,
    }


@app.post("/api/analyze")
async def api_analyze(
    sample_id: str | None = Form(default=None),
    gradcam_mode: str = Form(default="predicted"),
    arms: str = Form(default=",".join(ARM_ORDER)),
    file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    if gradcam_mode not in ("predicted", "scc"):
        raise HTTPException(status_code=400, detail="gradcam_mode must be 'predicted' or 'scc'.")

    selected_arms = [a.strip() for a in arms.split(",") if a.strip()]
    sample_meta: dict[str, Any] | None = None

    if file is not None and file.filename:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded image exceeds 10 MB limit.")
        try:
            pil_image = Image.open(io.BytesIO(raw))
            pil_image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail="Invalid image file. Upload PNG or JPEG.") from exc
        case_name = Path(file.filename).name
    elif sample_id:
        sample_meta = _sample_lookup().get(sample_id)
        if sample_meta is None:
            raise HTTPException(status_code=404, detail="Unknown sample_id.")
        img_path = (PROJECT_DIR / sample_meta["rel_path"]).resolve()
        if not img_path.is_relative_to(PROJECT_DIR.resolve()) or not img_path.exists():
            raise HTTPException(status_code=404, detail="Sample image not found.")
        with Image.open(img_path) as opened:
            pil_image = opened.convert("RGB")
        case_name = sample_meta["image_name"]
    else:
        raise HTTPException(status_code=400, detail="Provide either an image file or a sample_id.")

    result = get_registry().analyze_image(pil_image, arms=selected_arms, gradcam_mode=gradcam_mode)
    result["case_label"] = case_name
    result["sample_metadata"] = (
        {k: v for k, v in sample_meta.items() if k != "rel_path"} if sample_meta else None
    )
    result["disclaimer"] = DISCLAIMER_TEXT
    return result


@app.post("/api/cgan/sample")
def api_cgan_sample(request: CGANSampleRequest) -> dict[str, Any]:
    try:
        return get_registry().sample_cgan(
            diagnosis=request.diagnosis,
            seed=request.seed,
            count=request.count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/report/export")
def api_report_export(request: ReportExportRequest) -> dict[str, str]:
    return build_preliminary_report(
        case_label=request.case_label,
        reviewer_notes=request.reviewer_notes,
        analysis=request.analysis,
    )


@app.get("/api/roadmap")
def api_roadmap() -> dict[str, Any]:
    return get_roadmap_payload()
