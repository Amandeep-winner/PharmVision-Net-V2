"""Extract metrics and figures from the executed notebook into dashboard/data.js and dashboard/img/."""
import base64, json, re
from pathlib import Path

HERE = Path(__file__).parent
nb = json.loads((HERE.parent / "ocr-handwritten-prescription.ipynb").read_text(encoding="utf8"))
cells = nb["cells"]

def text_of(cell):
    return "".join("".join(o["text"]) for o in cell["outputs"] if o.get("output_type") == "stream")

# ---- figures
names = {
    2: ["a_train_dist", "a_val_dist", "a_test_dist", "a_samples"],
    4: ["ocr_len_hist", "ocr_score_hist", "ocr_confusion"],
    5: ["b_len_hist", "b_score_hist"],
    11: ["v_train_dist", "v_val_dist", "v_test_dist", "v_samples", "v_width", "v_height"],
}
for m in ("resnet50", "densenet121", "elasticnet_cnn"):
    names.setdefault(15, []).extend(f"{m}_{k}" for k in ("loss", "auc", "f1", "cm"))
for idx, ns in names.items():
    figs = [o for o in cells[idx]["outputs"] if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})]
    for n, o in zip(ns, figs):
        (HERE / "img" / f"{n}.png").write_bytes(base64.b64decode(o["data"]["image/png"]))

# ---- training logs
log = text_of(cells[15])
ep_re = re.compile(r"\[(\w+)\] ep (\d+)/\d+ \| loss ([\d.]+)/([\d.]+) \| AUC ([\d.]+)/([\d.]+) \| F1 ([\d.]+)/([\d.]+) \| gap_auc=(-?[\d.]+) \| ([\d.]+)s")
models = {}
for m in ep_re.finditer(log):
    k = m.group(1)
    v = [float(x) for x in m.groups()[2:]]
    models.setdefault(k, {"epochs": []})["epochs"].append(dict(
        epoch=int(m.group(2)), train_loss=v[0], val_loss=v[1], train_auc=v[2], val_auc=v[3],
        train_f1=v[4], val_f1=v[5], gap_auc=v[6], sec=v[7]))

blocks = re.split(r"===== TEST: (\w+) =====", log)
for name, body in zip(blocks[1::2], blocks[2::2]):
    models[name]["test_auc"] = float(re.search(r"Test AUC.*?:\s*([\d.]+)", body).group(1))
    models[name]["test_f1"] = float(re.search(r"Test F1.*?:\s*([\d.]+)", body).group(1))
    models[name]["test_acc"] = float(re.search(r"accuracy\s+([\d.]+)", body).group(1))
    models[name]["per_class"] = [
        dict(cls=int(c), precision=float(p), recall=float(r), f1=float(f))
        for c, p, r, f in re.findall(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+10$", body, re.M)]
    models[name]["trainable"] = int(re.search(rf"{name}: trainable params = (\d+)", log).group(1))

# ---- OCR results
ocr_log = text_of(cells[4]) + text_of(cells[5])
a = re.search(r"N=(\d+) \| acc=([\d.]+) \| time=([\d.]+)s", ocr_log)
b = re.search(r"N=(\d+) \| hit_rate.*?=([\d.]+) \| time=([\d.]+)s", ocr_log)
sample_json = json.loads(re.search(r"Sample JSON \(first 2\):\n(\[.*\])", ocr_log, re.S).group(1))

data = dict(
    models=models,
    ocr=dict(
        a=dict(n=int(a[1]), acc=float(a[2]), sec=float(a[3])),
        b=dict(n=int(b[1]), hit_rate=float(b[2]), sec=float(b[3])),
        samples=sample_json),
    dataset=dict(classes=78, train=3120, val=780, test=780, b_images=129, per_class_test=10),
)
(HERE / "data.js").write_text("window.DATA = " + json.dumps(data, indent=1) + ";\n", encoding="utf8")
print({k: (len(v["epochs"]), len(v["per_class"])) for k, v in models.items()}, data["ocr"]["a"], data["ocr"]["b"])
