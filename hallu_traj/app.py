import argparse

import numpy as np

from .classifier import TrajectoryDetector
from .features import states_to_features
from .highlight import to_gradio_items
from .service import AnalysisResult, HallucinationService

SERIES_TITLES = {
    "speed": "Относительная скорость ‖Δx‖/‖x‖",
    "curv": "Кривизна (угол поворота, рад)",
    "stretch": "Локальное растяжение ln(‖v₍ₗ₊₁₎‖/‖v₍ₗ₎‖)",
}


CSS = """
<style>
footer, .footer {display: none !important;}
.gradio-container {max-width: 1100px !important; margin: 0 auto !important;}
.hero {padding: 28px 30px; border-radius: 18px; color: #fff; margin-bottom: 8px;
       background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);}
.hero h1 {margin: 0 0 8px 0; font-size: 30px; font-weight: 700; color: #fff;}
.hero p {margin: 0; font-size: 15px; line-height: 1.5; opacity: .92; color: #fff;}
.chips {margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap;}
.chip {background: rgba(255,255,255,.18); padding: 4px 12px; border-radius: 999px; font-size: 13px; color: #fff;}
.verdict {border-radius: 16px; padding: 18px 20px; border: 1px solid rgba(128,128,128,.25);}
.verdict .title {font-size: 13px; text-transform: uppercase; letter-spacing: .06em; opacity: .7;}
.verdict .score {font-size: 40px; font-weight: 700; line-height: 1.1; margin: 4px 0;}
.verdict .label {font-size: 17px; font-weight: 600; margin-bottom: 12px;}
.verdict .bar {height: 10px; border-radius: 999px; background: rgba(128,128,128,.25); overflow: hidden;}
.verdict .bar > div {height: 100%; border-radius: 999px;}
.verdict .spans {margin-top: 14px; font-size: 14px; line-height: 1.6;}
.verdict .muted {opacity: .65; font-size: 12.5px; margin-top: 10px;}
.verdict.ok {background: rgba(34,197,94,.12); border-color: rgba(34,197,94,.45);}
.verdict.ok .score, .verdict.ok .label {color: #16a34a;}
.verdict.ok .bar > div {background: #22c55e;}
.verdict.mid {background: rgba(245,158,11,.12); border-color: rgba(245,158,11,.5);}
.verdict.mid .score, .verdict.mid .label {color: #d97706;}
.verdict.mid .bar > div {background: #f59e0b;}
.verdict.bad {background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.5);}
.verdict.bad .score, .verdict.bad .label {color: #dc2626;}
.verdict.bad .bar > div {background: #ef4444;}
.verdict.empty {text-align: center; opacity: .6; padding: 40px 20px;}
.note {text-align: center; font-size: 12.5px; opacity: .6; padding: 10px 0 0 0;}
button.primary {border-radius: 12px !important; font-weight: 600 !important; font-size: 16px !important;}
.gr-group, .block {border-radius: 14px !important;}
</style>
"""

HERO = """
<div class="hero">
  <h1>Детектор галлюцинаций LLM</h1>
  <p>Оценка надёжности ответа по динамике скрытых состояний модели по слоям</p>
  <div class="chips"></div>
</div>
"""

EMPTY_CARD = '<div class="verdict empty">Введите вопрос и нажмите «Проверить»</div>'

NOTE = '<div class="note"> </div>'

EXAMPLES = [
    ["What happens if you crack your knuckles a lot?"],
    ["Can you see the Great Wall of China from space?"],
    ["What happens if you swallow gum?"],
    ["What is the capital of France?"],
]


def verdict_html(res: AnalysisResult) -> str:
    import html

    sc = float(res.answer_score)
    if sc < 0.35:
        cls, label = "ok", "Скорее надёжно"
    elif sc < 0.6:
        cls, label = "mid", "Есть сомнения"
    else:
        cls, label = "bad", "Высокий риск галлюцинации"
    if res.spans:
        spans = "".join(f"<li>«{html.escape(sp.text.strip())}» — {sp.mean_score:.2f}</li>" for sp in res.spans)
        spans_html = f'<div class="spans"><b>Подозрительные фразы:</b><ul style="margin:6px 0 0 18px">{spans}</ul></div>'
    else:
        spans_html = '<div class="spans">Подозрительных фраз не найдено.</div>'
    cost = ", ".join(f"{k}: {v:.2f} с" for k, v in res.timings.items())
    return (
        f'<div class="verdict {cls}"><div class="title">Оценка ненадёжности ответа</div>'
        f'<div class="score">{sc:.2f}</div><div class="label">{label}</div>'
        f'<div class="bar"><div style="width:{max(2.0, min(100.0, sc * 100)):.0f}%"></div></div>'
        f'{spans_html}<div class="muted">Время: {html.escape(cost)}</div></div>'
    )


def _frames(res: AnalysisResult, token: int | None):
    import pandas as pd

    frames = {}
    for name in SERIES_TITLES:
        layers, mean_y = res.profile(name, None)
        rows = [{"слой": int(l), "значение": float(v), "кривая": "среднее по ответу"} for l, v in zip(layers, mean_y)]
        if token is not None:
            _, y = res.profile(name, token)
            rows += [{"слой": int(l), "значение": float(v), "кривая": f"токен «{res.tokens[token].strip()}»"} for l, v in zip(layers, y)]
        frames[name] = pd.DataFrame(rows)
    return frames


def build_demo(service: HallucinationService):
    import gradio as gr

    def run(question: str, own_answer: str):
        if not question.strip():
            raise gr.Error("Введите вопрос")
        res = service.analyze(question.strip(), own_answer.strip() or None)
        items = to_gradio_items(res.tokens, res.token_scores, res.threshold)
        fr = _frames(res, None)
        return (res, items, verdict_html(res), fr["speed"], fr["curv"], fr["stretch"])

    def on_select(res: AnalysisResult, evt: gr.SelectData):
        if res is None:
            return gr.skip(), gr.skip(), gr.skip()
        idx = evt.index if isinstance(evt.index, int) else evt.index[0]
        idx = int(np.clip(idx, 0, len(res.tokens) - 1))
        fr = _frames(res, idx)
        return fr["speed"], fr["curv"], fr["stretch"]

    with gr.Blocks(title="Детектор галлюцинаций LLM") as demo:
        gr.HTML(CSS)
        gr.HTML(HERO)
        state = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                q = gr.Textbox(label="Вопрос (на английском)", lines=3, placeholder="Например: What happens if you crack your knuckles a lot?")
                a = gr.Textbox(label="Свой ответ (необязательно)", lines=3)
                btn = gr.Button("Проверить", variant="primary", size="lg")
                gr.Examples(examples=EXAMPLES, inputs=[q, a], label="Примеры вопросов")
            with gr.Column(scale=1):
                out_sum = gr.HTML(EMPTY_CARD)
        hl = gr.HighlightedText(
            label="Ответ модели: подсвечены токены с высокой оценкой ненадёжности (нажмите на токен, чтобы увидеть его траекторию)",
            combine_adjacent=False,
            show_legend=False,
        )
        with gr.Row():
            p1 = gr.LinePlot(x="слой", y="значение", color="кривая", title=SERIES_TITLES["speed"])
            p2 = gr.LinePlot(x="слой", y="значение", color="кривая", title=SERIES_TITLES["curv"])
            p3 = gr.LinePlot(x="слой", y="значение", color="кривая", title=SERIES_TITLES["stretch"])
        gr.HTML(NOTE)
        btn.click(run, [q, a], [state, hl, out_sum, p1, p2, p3])
        hl.select(on_select, [state], [p1, p2, p3])
    return demo


def _mock_service() -> HallucinationService:
    from .mock import MockExtractor

    ext = MockExtractor()
    feats, labels = [], []
    for i in range(60):
        halluc = i % 2
        q = f"{'halluc ' if halluc else ''}question number {i} about something"
        rec = ext.analyze(q)
        feats.append(states_to_features(rec.states))
        labels.append(halluc)
    det = TrajectoryDetector().fit(feats, labels)
    return HallucinationService(ext, det)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="имя модели Hugging Face")
    ap.add_argument("--detector", help="путь к обученному детектору (.joblib)")
    ap.add_argument("--mock", action="store_true", help="синтетическая демонстрация без модели")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    if args.mock:
        service = _mock_service()
    else:
        if not (args.model and args.detector):
            ap.error("укажите --model и --detector (или используйте --mock)")
        from .extract import HiddenStateExtractor

        ext = HiddenStateExtractor(args.model, device=args.device, dtype=args.dtype)
        service = HallucinationService(ext, TrajectoryDetector.load(args.detector))
    demo = build_demo(service)
    try:
        demo.launch(server_port=args.port, share=args.share, show_api=False)
    except TypeError:  
        demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()