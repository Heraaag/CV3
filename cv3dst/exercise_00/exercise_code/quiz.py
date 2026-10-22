import hashlib
import html
import json
import re
from pathlib import Path
from zipfile import ZipFile

import ipywidgets as w
from IPython.display import display

ROOT = Path(__file__).parent
ANSWERS = ROOT.parent / "quiz_answers.json"
NONE = "none"


def rank(*keys: str) -> str:
    return hashlib.sha256("/".join(keys).encode()).hexdigest()


def user_id(user: str) -> str:
    # only the number selects the questions: u042, 042 and 42 are all user 42
    return str(int(user[1:] if user[:1].isalpha() else user))


def pick(ids: list[str], user: str, ex: str, n: int) -> list[str]:
    # the evaluation server draws the same questions from your user name
    return sorted(ids, key=lambda i: rank(user, ex, i))[:n]


def pick_options(options: list[str], user: str, qid: str, k: int) -> list[str]:
    # at most k options of a question in your own order, the server grades the same ones
    return sorted(options, key=lambda o: rank(user, qid, o))[:k]


def fmt(text: str) -> str:
    return re.sub(r"`([^`]*)`", r"<code>\1</code>", html.escape(text))


def show_quiz(user: str, out: Path = ANSWERS) -> None:
    """Shows your quiz questions as checkboxes and saves every change to `out`."""
    uid = user_id(user)
    bank = json.loads((ROOT / "quiz_questions.json").read_text())
    qs = {q["id"]: q for q in bank["questions"]}
    old = json.loads(out.read_text()) if out.exists() else {}
    old = old["answers"] if old and user_id(old["user"]) == uid else {}

    boxes: dict[str, dict[str, w.Checkbox]] = {}
    status = w.Label()

    def save() -> None:
        ans = {
            qid: [k for k, cb in bs.items() if cb.value] for qid, bs in boxes.items()
        }
        out.write_text(json.dumps({"user": user, "answers": ans}, indent=2))
        status.value = f"{sum(map(bool, ans.values()))}/{len(ans)} questions answered, saved to {out.name}"

    def on_change(qid: str, key: str, change: dict) -> None:
        # "None are correct" excludes all other options
        if change["new"]:
            for k, cb in boxes[qid].items():
                if (k == NONE) != (key == NONE):
                    cb.value = False
        save()

    rows = []
    for i, qid in enumerate(pick(list(qs), uid, bank["exercise"], bank["n"]), 1):
        opts = qs[qid]["options"] | {NONE: "None are correct"}
        keys = pick_options(list(qs[qid]["options"]), uid, qid, bank["max_options"]) + [
            NONE
        ]
        boxes[qid] = {
            k: w.Checkbox(
                value=k in old.get(qid, []), indent=False, layout=w.Layout(width="30px")
            )
            for k in keys
        }
        for k, cb in boxes[qid].items():
            cb.observe(lambda c, qid=qid, k=k: on_change(qid, k, c), names="value")
        rows.append(w.HTMLMath(f"<b>{i}. {fmt(qs[qid]['q'])}</b>"))
        if "asset" in qs[qid]:
            img = (ROOT / "quiz_assets" / f"{qs[qid]['asset']}.png").read_bytes()
            rows.append(w.Image(value=img, layout=w.Layout(max_width="700px")))
        rows += [w.HBox([boxes[qid][k], w.HTMLMath(fmt(opts[k]))]) for k in keys]

    save()
    display(w.VBox(rows + [status]))


def submit_quiz(zip_path: Path, answers: Path = ANSWERS) -> None:
    """Zips your quiz answers for the upload to the quiz exercise on the test server."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w") as z:
        z.write(answers, answers.name)
    print(
        f"Zipped the quiz answers of {json.loads(answers.read_text())['user']} to {zip_path.resolve()}"
    )
