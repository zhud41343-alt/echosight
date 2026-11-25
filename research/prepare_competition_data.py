from __future__ import annotations

import csv
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs" / "competition_data"
OUT.mkdir(parents=True, exist_ok=True)

# These values come from the deterministic synthetic validation report already
# checked into research/outputs/ablation_report.md.
model_metrics = [
    {
        "model": "FastDepth-Lite",
        "parameters_m": 1.02,
        "abs_rel": 0.0706,
        "rmse": 0.0798,
        "delta1": 0.9047,
        "acoustic_rmse": 0.0562,
        "fps_cpu": 88.2,
        "source": "64-scene synthetic validation, seed=999",
    },
    {
        "model": "EchoDepthNet Baseline",
        "parameters_m": 1.08,
        "abs_rel": 0.0710,
        "rmse": 0.0872,
        "delta1": 0.8864,
        "acoustic_rmse": 0.0583,
        "fps_cpu": 94.1,
        "source": "64-scene synthetic validation, seed=999",
    },
    {
        "model": "EchoDepthNet + Residual EG-DFM",
        "parameters_m": 1.11,
        "abs_rel": 0.0754,
        "rmse": 0.0824,
        "delta1": 0.8970,
        "acoustic_rmse": 0.0577,
        "fps_cpu": 74.2,
        "source": "64-scene synthetic validation, seed=999",
    },
]

with (OUT / "model_metrics_synthetic.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=model_metrics[0].keys())
    writer.writeheader()
    writer.writerows(model_metrics)

# A reproducible scene manifest for demonstrations. It describes generated
# scene conditions and expected interaction paths. It is not field data.
rng = random.Random(999)
scene_rows = []
for index in range(1, 65):
    nearest_distance = round(0.5 + rng.random() * 9.5, 2)
    object_count = 1 + rng.randrange(4)
    hazard_level = "near" if nearest_distance < 2 else "mid" if nearest_distance < 5 else "far"
    scene_rows.append(
        {
            "scene_id": f"SYN-{index:03d}",
            "generator": "SyntheticObstacleDepthDataset",
            "seed": 999,
            "image_size": "256x256",
            "object_count": object_count,
            "nearest_distance_band": hazard_level,
            "nearest_distance_m_simulated": nearest_distance,
            "acoustic_matrix": "16x48",
            "failure_injection": "freeze" if index % 8 == 0 else "none",
            "expected_safety_behavior": "stop_continuous_audio_and_clear_stale_result" if index % 8 == 0 else "emit_current_frame_only",
            "data_status": "synthetic_demo_only",
        }
    )

with (OUT / "synthetic_scene_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=scene_rows[0].keys())
    writer.writeheader()
    writer.writerows(scene_rows)

with (OUT / "synthetic_scene_manifest.json").open("w", encoding="utf-8") as f:
    json.dump(
        {
            "dataset_name": "EchoSight competition synthetic scene manifest",
            "status": "synthetic_demo_only",
            "purpose": "reproducible demo and pipeline documentation",
            "warning": "Do not present these rows as real camera, distance, or user-study measurements.",
            "seed": 999,
            "scene_count": len(scene_rows),
            "scenes": scene_rows,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

validation_rows = [
    {"evidence": "Vitest unit tests", "result": "49/49 passed", "command": "npm test", "status": "verified_local"},
    {"evidence": "Research pipeline integrity", "result": "5/5 passed", "command": "python research/tests/test_pipeline_integrity.py", "status": "verified_local"},
    {"evidence": "Production build", "result": "success", "command": "npm run build", "status": "verified_local"},
    {"evidence": "ONNX model", "result": "4.46 MB present", "command": "dist/models/echodepth_net.onnx", "status": "verified_local"},
    {"evidence": "Real phone screen-reader flow", "result": "pending", "command": "iOS VoiceOver / Android TalkBack", "status": "requires_field_test"},
    {"evidence": "Real distance MAE/P90", "result": "pending", "command": "0.5–10 m tape-measure protocol", "status": "requires_field_test"},
    {"evidence": "Visually impaired co-creation", "result": "pending", "command": "approved controlled study protocol", "status": "requires_ethics_and_field_test"},
]
with (OUT / "validation_evidence.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=validation_rows[0].keys())
    writer.writeheader()
    writer.writerows(validation_rows)

summary = {
    "package": "EchoSight competition data package",
    "created_from": "D:/听见光影/research/outputs/ablation_report.md",
    "synthetic_scene_count": len(scene_rows),
    "model_metric_rows": len(model_metrics),
    "verified_engineering_checks": 4,
    "pending_real_world_checks": 3,
    "usage_rule": "Synthetic rows support demos and reproducibility. They must not be described as real-world measurements.",
}
with (OUT / "package_manifest.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

(OUT / "README.md").write_text(
    """# EchoSight 参赛数据包\n\n本目录整理了可用于比赛附件、演示和工程审计的数据。\n\n## 文件\n\n- `model_metrics_synthetic.csv`：基于现有 64 场景确定性合成验证的三模型指标。\n- `synthetic_scene_manifest.csv`：64 条可复现的合成演示场景清单。\n- `synthetic_scene_manifest.json`：同一场景清单的 JSON 版本。\n- `validation_evidence.csv`：已完成工程检查与待完成真实验证的状态表。\n- `package_manifest.json`：数据包元信息。\n\n## 数据边界\n\n合成场景和模拟距离只用于演示、回归和管线说明。它们不能代替真实手机测距、VoiceOver/TalkBack 真机测试或视障用户研究。真实数据完成后，应将原始记录、设备信息、采集日期和计算方法一并归档。\n""",
    encoding="utf-8",
)
print(f"Wrote {len(scene_rows)} synthetic scenes and {len(model_metrics)} model rows to {OUT}")
