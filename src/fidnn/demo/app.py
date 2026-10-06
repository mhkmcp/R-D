"""`fidnn ui`: a plain-language Gradio front end over `DemoSession` (needs `uv sync --group ui`)."""

import html

import numpy as np
import pandas as pd

from fidnn.demo.export import gallery
from fidnn.demo.session import DemoSession

WHERE = {"Early layers": "early", "Middle layers": "middle", "Late layers": "late"}
SEVERITY = {  # plain words for the §4.2 strata, most to least drastic
    8: {"Sign bit — drastic": "sign", "Top bit — severe": "msb", "High bits": "high",
        "Middle bits": "mid", "Lowest bits — subtle": "low"},
    32: {"Sign bit": "sign", "Exponent, top — drastic": "exp_msb", "Exponent, low": "exp_low",
         "Mantissa, high": "mant_high", "Mantissa, low — subtle": "mant_low"},
}
OUTCOME = {
    "MASKED": ("No effect", "The answer and the confidence are unchanged."),
    "DEGRADED": ("Shaken, same answer", "The answer is still the same, but the internal numbers moved."),
    "SDC": ("Wrong answer, silently", "The model now gives a different answer and shows no error."),
    "CRASH": ("Broken", "The model's output is unusable on this batch."),
}
TRACK = {"S": "looks harmless from the outside", "H": "visibly harmful"}
FPR = {0.05: "5 % (sensitive)", 0.01: "1 % (balanced)", 0.001: "0.1 % (strict)"}

CSS = """
.card {border-radius: 12px; padding: 14px 16px; border: 1px solid var(--border-color-primary);}
.card h3 {margin: 0 0 4px 0; font-size: 0.85rem; opacity: 0.75; font-weight: 600;}
.card .big {font-size: 1.35rem; font-weight: 700;}
.ok {border-left: 6px solid #2e7d5b;} .bad {border-left: 6px solid #c0392b;}
.neutral {border-left: 6px solid #7a7a7a;}
"""


def _upscale(img: np.ndarray, k: int = 4) -> np.ndarray:
    return img.repeat(k, 0).repeat(k, 1)


def _card(title: str, big: str, body: str, tone: str) -> str:
    return (f'<div class="card {tone}"><h3>{html.escape(title)}</h3>'
            f'<div class="big">{html.escape(big)}</div><div>{html.escape(body)}</div></div>')


def _to_32(pil) -> np.ndarray:
    """Centre-crop to a square and shrink to the 32×32 RGB the model reads."""
    from PIL import Image

    img = pil.convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((32, 32), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _flip_list(session: DemoSession) -> str:
    if not session.flips:
        return "*The model is clean — no bits flipped.*"
    lines = []
    for f in session.flips[:12]:
        d = session.describe_flip(f)
        lines.append(f"- `{f['layer']}` weight #{f['flat_index']}, bit {f['bit']}: "
                     f"{d['old']:+.4g} → {d['new']:+.4g}")
    more = len(session.flips) - 12
    return "\n".join(lines + ([f"- … and {more} more"] if more > 0 else []))


def render(session: DemoSession, image: int, alpha: float):
    """Everything step 3 shows, for one image judged inside its 32-probe batch."""
    ev = session.evaluate(session.batch_for(image))
    cls = session.classes
    clean_p, fault_row = int(ev.clean_logits[0].argmax()), ev.fault_logits[0]
    broken = not np.isfinite(fault_row).all()
    fault_p = -1 if broken else int(fault_row.argmax())
    probs = np.exp(fault_row - fault_row.max()) / np.exp(fault_row - fault_row.max()).sum()
    top = ", ".join(f"{cls[i]} {probs[i]:.0%}" for i in np.argsort(-probs)[:3]) if not broken else "—"

    pred = _card("Model's answer",
                 f"{cls[clean_p]} → {'?' if broken else cls[fault_p]}",
                 f"True label: {cls[int(ev.y[0])]}. Top 3 now: {top}.",
                 "neutral" if fault_p == clean_p else "bad")
    alarm = bool(ev.alarm(alpha)[0])
    det = _card(f"Detector (false-alarm rate {FPR.get(alpha, alpha)})",
                "🔴 Fault detected" if alarm else "🟢 Looks normal",
                "It watches the network's inner layers, not just the answer.",
                "bad" if alarm else "ok")
    lab = ev.labels[0]
    name, why = OUTCOME[lab]
    out = _card("What the fault did", name, f"{why} This kind of fault {TRACK[ev.tracks[0]]}.",
                "ok" if lab == "MASKED" else "bad")

    margin = ev.d1[0] - ev.d1_thresholds
    chart = pd.DataFrame({"layer": ev.taps, "past alarm line": np.clip(margin, -50, 50),
                          "status": np.where(margin > 0, "over its alarm line", "normal")})
    first = ev.first_alarm[0]
    where = (f"The first layer to look abnormal is **{ev.taps[first]}**." if first >= 0 else
             "No single layer crossed its own alarm line.")
    adv = pd.DataFrame({
        "layer": ev.taps, "score (clean)": ev.d1_clean[0].round(3),
        "score (faulty)": ev.d1[0].round(3), "layer threshold": ev.d1_thresholds.round(3)})
    detail = (f"Fused detector (D2) score **{ev.d2[0]:.4f}** vs threshold τ = "
              f"**{ev.taus[alpha]:.4f}** · output-only confidence score (MSP) "
              f"{ev.output_only['msp'][0]:.3f} · outcome label `{lab}` (Track {ev.tracks[0]}) · "
              f"model checksum `{ev.checksum}` (clean: `{session.clean_checksum}`)")
    return pred, det, out, chart, where, adv, detail, _flip_list(session)


def build_app(session: DemoSession):
    import gradio as gr

    picks = gallery(session)
    severity = SEVERITY[8 if "msb" in session.strata() else 32]  # weight strata, not biases
    alphas = sorted(session.taus, reverse=True)

    with gr.Blocks(title="fidnn — fault detection demo") as demo:
        gr.Markdown(
            "# Demo: Faul-Injection Detection\n")
        image = gr.State(int(picks[0]))
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260):
                with gr.Row(equal_height=True):
                    gr.Markdown("### 1 · Pick a picture")
                    upload = gr.UploadButton("Upload image", file_types=["image"], size="sm",
                                             scale=0, min_width=120)
                with gr.Row(visible=False) as preview:
                    original = gr.Image(label="Your upload", interactive=False, height=140)
                    seen = gr.Image(label="What the model sees (32×32)", interactive=False,
                                    height=140)
                gal = gr.Gallery([(_upscale(session.images[i]), session.classes[int(session.y[i])])
                                  for i in picks], columns=4, height="auto",
                                 allow_preview=False, show_label=False)
                chosen = gr.Markdown()
                rnd = gr.Button("Random picture")
                truth = gr.Dropdown(["Don't know"] + list(session.classes), value="Don't know",
                                    label="Uploaded image shows")
                upload_err = gr.Markdown()
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### 2 · Damage the memory")
                where = gr.Radio(list(WHERE), value="Late layers", label="Where")
                how = gr.Radio(list(severity), value=next(iter(severity)), label="Which bit")
                count = gr.Radio([1, 4, 16], value=1, label="How many flips")
                with gr.Row():
                    flip = gr.Button("Flip bits", variant="primary")
                    surprise = gr.Button("Surprise me")
                reset = gr.Button("Reset model")
                flips_md = gr.Markdown()
                with gr.Accordion("Advanced: flip one exact bit", open=False):
                    layer = gr.Dropdown(session.layers(), label="Layer")
                    index = gr.Number(0, precision=0, label="Weight index")
                    bit = gr.Number(7, precision=0, label="Bit (0 = lowest)")
                    exact = gr.Button("Flip this bit")
                    err = gr.Markdown()
            with gr.Column(scale=2, min_width=320):
                gr.Markdown("### 3 · What happened")
                alpha = gr.Radio([(FPR.get(a, str(a)), a) for a in alphas],
                                 value=0.01 if 0.01 in alphas else alphas[0],
                                 label="How often may the detector cry wolf on a healthy model?")
                with gr.Row():
                    pred, det, out = gr.HTML(), gr.HTML(), gr.HTML()
                gr.Markdown("**How unusual each layer looks** — bars above 0 crossed that "
                            "layer's alarm line (clipped at ±50).")
                chart = gr.BarPlot(x="layer", y="past alarm line", color="status",
                                   color_map={"normal": "#4c78a8",
                                              "over its alarm line": "#c0392b"},
                                   sort=None, show_label=False, height=260)
                where_md = gr.Markdown()
                many = gr.Button("Try this damage on 200 pictures")
                many_df = gr.Dataframe(visible=False, label="Over 200 pictures")
                with gr.Accordion("Advanced: scores and checksum", open=False):
                    adv = gr.Dataframe(show_label=False)
                    detail = gr.Markdown()

        outputs = [pred, det, out, chart, where_md, adv, detail, flips_md]

        def show(img, a):
            cls = session.classes[int(session.y[img])]
            what = (f"Selected: **your upload**, treated as **{cls}**" if img >= session.n_pool
                    else f"Selected: **{cls}** (picture #{img})")
            return (what, *render(session, img, a))

        def pick(evt: gr.SelectData, a):
            img = int(picks[evt.index])
            return (img, *show(img, a))

        def random_pick(a):
            img = int(np.random.default_rng().integers(session.n_pool))
            return (img, *show(img, a))

        def do_upload(path, t, img, a):
            from PIL import Image, UnidentifiedImageError

            keep = (gr.Row(), gr.Image(), gr.Image())
            if path is None:
                return ("", *keep, img, *show(img, a))
            try:
                pil = Image.open(path).convert("RGB")
            except UnidentifiedImageError:
                return ("⚠️ That file is not an image.", *keep, img, *show(img, a))
            label = None if t == "Don't know" else session.classes.index(t)
            small = _to_32(pil)
            new = session.add_image(small, label)
            note = ("" if label is not None else
                    "*No label given: the clean model's answer counts as the true one.*")
            return (note, gr.Row(visible=True), np.array(pil), _upscale(small, 8), new,
                    *show(new, a))

        def do_flip(img, w, h, n, a):
            session.add_flips(WHERE[w], severity[h], int(n), np.random.default_rng())
            return show(img, a)

        def do_surprise(img, a):
            rng = np.random.default_rng()
            session.add_flips(str(rng.choice(list(WHERE.values()))),
                              str(rng.choice(list(severity.values())[:2])), 4, rng)
            return show(img, a)

        def do_reset(img, a):
            session.reset()
            return show(img, a)

        def do_exact(img, lay, i, b, a):
            try:
                session.add_flip(lay, int(i), int(b))
                msg = ""
            except (ValueError, StopIteration) as exc:
                msg = f"⚠️ {str(exc) or 'Choose a layer first.'}"
            return (msg, *show(img, a))

        def do_many(a):
            idx = np.arange(min(200, len(session.y)))
            table = session.summary(session.evaluate(idx), a)
            table = table.rename(columns={"group": "pictures", "clean_accuracy": "accuracy before",
                                          "fault_accuracy": "accuracy after",
                                          "alarm_rate": "share flagged by detector"})
            table["pictures"] = table["pictures"].replace(
                {"all probes": "all pictures",
                 "track S": "answer unchanged — damage hidden (Track S)",
                 "track H": "answer wrong or broken (Track H)"})
            return gr.Dataframe(table.round(3), visible=True)

        shown = [chosen, *outputs]
        demo.load(show, [image, alpha], shown)
        gal.select(pick, [alpha], [image, *shown])
        rnd.click(random_pick, [alpha], [image, *shown])
        upload.upload(do_upload, [upload, truth, image, alpha], [upload_err, preview, original, seen, image, *shown])
        flip.click(do_flip, [image, where, how, count, alpha], shown)
        surprise.click(do_surprise, [image, alpha], shown)
        reset.click(do_reset, [image, alpha], shown)
        exact.click(do_exact, [image, layer, index, bit, alpha], [err, *shown])
        alpha.change(show, [image, alpha], shown)
        many.click(do_many, [alpha], [many_df])
    return demo


def prerequisites_app(missing: list[tuple[str, str]]):
    import gradio as gr

    steps = "\n".join(f"{i}. `{cmd}`  \n   (makes `{path}`)"
                      for i, (path, cmd) in enumerate(missing, 1))
    with gr.Blocks(title="fidnn — setup needed") as demo:
        gr.Markdown("# Almost there\nThe demo needs a trained model and fitted detectors. "
                    f"Run these once in a terminal, then start `fidnn ui` again:\n\n{steps}")
    return demo


def launch(model_id: str, precision: str, tap_set: str, seed: int, port: int | None = None,
           share: bool = False) -> None:
    from fidnn.demo.session import missing_prerequisites

    missing = missing_prerequisites(model_id, precision, tap_set, seed)
    demo = (prerequisites_app(missing) if missing else
            build_app(DemoSession.from_artifacts(model_id, precision, tap_set, seed)))
    demo.launch(server_port=port, share=share, css=CSS)
