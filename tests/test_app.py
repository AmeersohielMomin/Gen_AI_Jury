"""Integration tests for the GenAI Jury FastAPI backend and ML explainability."""

from __future__ import annotations

import io
import json

import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from src.constants import ARTIFACT_DIR, CACHE_PATH

client = TestClient(app)


def test_eval_cache_matches_reference_csv() -> None:
    assert CACHE_PATH.exists(), "Run `python3 -m scripts.build_eval_cache` first."
    cache = json.loads(CACHE_PATH.read_text())
    ref = pd.read_csv(ARTIFACT_DIR / "classifier_comparison.csv").set_index("arm")
    for arm in cache["arm_order"]:
        cached_f1 = cache["arms"][arm]["metrics"]["target_f1"]
        expected_f1 = round(float(ref.loc[arm, "target_f1"]), 4)
        assert abs(cached_f1 - expected_f1) <= 1e-4
    assert cache["cgan_audit"]["synthetic_detector_auc"] > 0.85


def test_core_get_endpoints() -> None:
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json() == {"status": "ok"}

    overview = client.get("/api/overview").json()
    assert overview["splits"]["test"]["images"] == 3664
    assert len(overview["arm_order"]) == 3

    jury_lab = client.get("/api/jury-lab").json()
    assert len(jury_lab["arms"]["real_only"]["confusion_matrix"]) == 8
    assert len(jury_lab["cgan_audit"]["loss_history"]) == 80

    roadmap = client.get("/api/roadmap").json()
    assert len(roadmap["modalities"]) >= 5
    assert len(roadmap["methodology_phases"]) == 9


def test_quiz_isolation_and_scoring() -> None:
    panels_resp = client.get("/api/quiz/panels")
    assert panels_resp.status_code == 200
    panels = panels_resp.json()["panels"]
    assert len(panels) == 24
    # Ensure ground truth is never exposed in panel listing
    for p in panels:
        assert "source" not in p
        assert "actual_source" not in p

    # Block direct download of PRIVATE_answer_key.csv or traversal attempts
    assert client.get("/api/quiz/image/PRIVATE_answer_key.csv").status_code == 404
    assert client.get("/api/quiz/image/panel_99.png").status_code == 404
    assert client.get("/api/quiz/image/panel_01.png").status_code == 200

    score_resp = client.post(
        "/api/quiz/score",
        json={"guesses": [{"panel_id": p["panel_id"], "choice": "real"} for p in panels]},
    )
    assert score_resp.status_code == 200
    scored = score_resp.json()
    assert scored["total_panels"] == 24
    assert scored["real_total"] == 12
    assert scored["synthetic_total"] == 12
    assert scored["correct_count"] == 12


def test_live_analyze_sample_and_upload_plus_report() -> None:
    samples = client.get("/api/samples").json()["samples"]
    assert len(samples) >= 10
    first_id = samples[0]["sample_id"]
    assert client.get(f"/api/media/sample/{first_id}").status_code == 200

    res = client.post("/api/analyze", data={"sample_id": first_id, "gradcam_mode": "scc"})
    assert res.status_code == 200
    body = res.json()
    assert set(body["arms"].keys()) == {"real_only", "repeated_real_scc", "cgan_scc"}
    for arm_val in body["arms"].values():
        assert arm_val["overlay_png"].startswith("data:image/png;base64,")
        assert arm_val["heatmap_png"].startswith("data:image/png;base64,")
        assert arm_val["gradcam_target"] == "SCC"
        assert len(arm_val["probabilities"]) == 8

    # Test upload validation + inference
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(180, 120, 110)).save(buf, format="PNG")
    buf.seek(0)
    up_res = client.post(
        "/api/analyze",
        data={"gradcam_mode": "predicted"},
        files={"file": ("synthetic_probe.png", buf.getvalue(), "image/png")},
    )
    assert up_res.status_code == 200

    # Test sanitized HTML/Markdown report export (XSS escaping)
    report_res = client.post(
        "/api/report/export",
        json={
            "case_label": "Test <script>alert(1)</script>",
            "reviewer_notes": "Check <b>lesion</b> border",
            "analysis": body,
        },
    )
    assert report_res.status_code == 200
    report = report_res.json()
    assert "<script>alert(1)</script>" not in report["html"]
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report["html"]
    assert "RESEARCH PROTOTYPE ONLY" in report["markdown"]


def test_live_cgan_sampler_and_epoch_media() -> None:
    assert client.get("/api/media/epoch/80").status_code == 200
    assert client.get("/api/media/epoch/999").status_code == 404

    resp = client.post("/api/cgan/sample", json={"diagnosis": "SCC", "seed": 146, "count": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["diagnosis"] == "SCC"
    assert len(data["samples"]) == 2
    assert data["samples"][0]["image_png"].startswith("data:image/png;base64,")
