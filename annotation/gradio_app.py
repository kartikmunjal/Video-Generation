"""
Human-in-the-loop annotation interface for video preference collection.

Presents pairs of generated video clips side by side and records which is
preferred by a human annotator. Outputs JSONL preference records compatible
with VideoPreferenceDataset.

Usage:
    python annotation/gradio_app.py \
        --input data/prefs_auto.jsonl \
        --output data/prefs_human.jsonl \
        [--port 7860] \
        [--share]   # creates a public URL for remote annotation

The app shows each pair once. Progress is saved after every annotation so
you can interrupt and resume without losing work.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Automated preference JSONL to annotate")
    p.add_argument("--output", required=True, help="Output JSONL for human-labeled pairs")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create public Gradio URL")
    p.add_argument("--max_pairs", type=int, default=None, help="Annotate only first N pairs")
    return p.parse_args()


def load_records(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_record(path: str, record: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def get_annotated_prompts(output_path: str) -> set:
    """Return the set of prompts already annotated (for resume)."""
    if not os.path.exists(output_path):
        return set()
    return {json.loads(l)["prompt"] for l in open(output_path) if l.strip()}


def build_interface(records: List[Dict], output_path: str) -> "gr.Blocks":
    """
    Build the Gradio interface.

    Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │ Progress: 3 / 50 pairs annotated                           │
    ├─────────────────────────────────────────────────────────────┤
    │ Prompt: "a cat jumping over a fence at sunset"             │
    ├───────────────────────┬─────────────────────────────────────┤
    │        Video A        │           Video B                  │
    │    [video player]     │       [video player]               │
    ├───────────────────────┴─────────────────────────────────────┤
    │  [A is better]  [Too similar / tie]  [B is better]         │
    │  Optional note: [text input]                               │
    └─────────────────────────────────────────────────────────────┘
    """
    import gradio as gr

    state = {
        "index": 0,
        "records": records,
        "annotated": get_annotated_prompts(output_path),
        "output_path": output_path,
    }

    def skip_annotated():
        """Advance index past already-annotated records."""
        while (
            state["index"] < len(state["records"])
            and state["records"][state["index"]]["prompt"] in state["annotated"]
        ):
            state["index"] += 1

    skip_annotated()

    def get_current() -> Dict:
        if state["index"] >= len(state["records"]):
            return None
        return state["records"][state["index"]]

    def render_current():
        """Return values for Gradio components from current record."""
        rec = get_current()
        if rec is None:
            return (
                "✅ All pairs annotated!",
                None,
                None,
                f"{len(state['annotated'])} / {len(state['records'])} complete",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        progress = f"{len(state['annotated'])} / {len(state['records'])} annotated"
        return (
            f"**{rec['prompt']}**",
            rec["chosen"],    # video A (could be either chosen or rejected from auto-labeling)
            rec["rejected"],  # video B
            progress,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    def save_annotation(choice: str, note: str) -> tuple:
        """
        Record the human annotation and advance to next pair.

        Args:
            choice: "A" | "tie" | "B"
        """
        rec = get_current()
        if rec is None:
            return render_current()

        if choice == "A":
            chosen, rejected = rec["chosen"], rec["rejected"]
        elif choice == "B":
            chosen, rejected = rec["rejected"], rec["chosen"]
        else:  # tie — skip without recording
            state["index"] += 1
            skip_annotated()
            return render_current() + ("",)

        human_record = {
            "prompt": rec["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "source": "human",
            "note": note.strip() if note else "",
            "auto_chosen": rec.get("chosen"),    # auto label for comparison
            "auto_scores": rec.get("scores", {}),
        }
        append_record(state["output_path"], human_record)
        state["annotated"].add(rec["prompt"])
        state["index"] += 1
        skip_annotated()

        return render_current() + ("",)  # clear note field

    with gr.Blocks(title="Video Preference Annotation") as demo:
        gr.Markdown("# Video Preference Annotation\nRate which generated video better matches the prompt.")

        progress_text = gr.Markdown(f"0 / {len(records)} annotated")
        prompt_text = gr.Markdown("Loading...")

        with gr.Row():
            video_a = gr.Video(label="Video A", interactive=False, autoplay=True)
            video_b = gr.Video(label="Video B", interactive=False, autoplay=True)

        with gr.Row():
            btn_a = gr.Button("👈  A is better", variant="primary")
            btn_tie = gr.Button("🤷  Too similar / tie", variant="secondary")
            btn_b = gr.Button("👉  B is better", variant="primary")

        note_input = gr.Textbox(
            label="Optional: note why (used for reward signal analysis)",
            placeholder="e.g. 'A has smoother motion but B is more prompt-aligned'",
            lines=2,
        )

        # Wire buttons
        outputs = [prompt_text, video_a, video_b, progress_text, btn_a, btn_tie, btn_b, note_input]

        btn_a.click(fn=lambda note: save_annotation("A", note), inputs=[note_input], outputs=outputs)
        btn_tie.click(fn=lambda note: save_annotation("tie", note), inputs=[note_input], outputs=outputs)
        btn_b.click(fn=lambda note: save_annotation("B", note), inputs=[note_input], outputs=outputs)

        # Initialize display
        demo.load(fn=render_current, outputs=outputs[:-1])

    return demo


def main():
    args = parse_args()

    records = load_records(args.input)
    if args.max_pairs:
        records = records[:args.max_pairs]

    already = get_annotated_prompts(args.output) if os.path.exists(args.output) else set()
    remaining = [r for r in records if r["prompt"] not in already]
    print(f"Loaded {len(records)} pairs. {len(already)} already annotated. {len(remaining)} remaining.")

    try:
        import gradio as gr
    except ImportError:
        raise ImportError("gradio is required: pip install gradio>=4.20.0")

    demo = build_interface(records, args.output)
    demo.launch(
        server_port=args.port,
        share=args.share,
        quiet=False,
    )


if __name__ == "__main__":
    main()
