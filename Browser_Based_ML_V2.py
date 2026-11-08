# Run this in terminal
# streamlit run Browser_Based_ML_AR.py
#
# Voice-driven AR-glasses simulation of the JASC fault classifier.
#   - The screen is static: nothing scrolls, nothing is clicked.
#   - Every action is a spoken command. In this simulator the command line at
#     the bottom stands in for the microphone: type what you would say and
#     press Enter.
#   - Options are numbered; say "select 2", "details 2", "next", "previous",
#     "back", "report", "new".
#
# Optional: create .streamlit/config.toml with
#     [theme]
#     base = "dark"

import html
import re
import ssl
from datetime import datetime
from functools import lru_cache

import certifi
import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="JASC Fault Classifier · AR",
    page_icon="🥽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------- config
MODEL_PATH = (
    "/Users/nileshsaumyadasa/Documents/AIE410_Project/SVM_TFIDF/"
    "Model_Results/jasc_linear_svm_Final.joblib"
)
CODES_PATH = (
    "/Users/nileshsaumyadasa/Documents/AIE410_Project/JASC_CODES_DOC/"
    "JASC_Code_Categories.csv"
)
# Must match whatever you used at training time
NEGATIONS_TO_KEEP = {"no", "not", "nor", "without"}

CATEGORY_PAGE_SIZE = 6  # predicted categories per screen
CODE_PAGE_SIZE = 6  # specific codes per screen
REPORT_ROWS = 4  # report entries shown (latest first); each takes two lines
SHOW_DEBUG = False  # print preprocessed text in the terminal after classifying

# Shown on the home screen as examples of what to say
VOICE_SAMPLES = [
    "Cabin door does not latch securely",
    "Hydraulic fluid leaking from nose gear",
    "Cockpit display flickers during cruise",
]

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)


# ---------------------------------------------------------------- loaders
@st.cache_resource
def load_nlp():
    """Download NLTK data once, then build the preprocessing function."""
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(resource, quiet=True)

    stop_words = set(stopwords.words("english")) - NEGATIONS_TO_KEEP
    lemmatize = lru_cache(maxsize=None)(WordNetLemmatizer().lemmatize)

    def preprocess_text(text: str) -> str:
        tokens = re.sub(r"[\W_]+", " ", str(text).lower()).split()
        return " ".join(lemmatize(t) for t in tokens if t not in stop_words)

    return preprocess_text


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_codes():
    """Return ({prefix: parent info}, {prefix: [sub-code infos]})."""
    df = pd.read_csv(CODES_PATH, dtype=str, encoding="utf-8-sig").fillna("")
    df = df.rename(
        columns={"Category": "code", "Short Name": "name", "Description": "description"}
    )
    df["code"] = df["code"].str.strip()
    df["prefix"] = df["code"].str[:2]
    df = df.sort_values("code")

    is_parent = df["code"].str.endswith("00")
    parents = {r["prefix"]: r for r in df[is_parent].to_dict("records")}
    children = {
        prefix: group.to_dict("records")
        for prefix, group in df[~is_parent].groupby("prefix")
    }
    return parents, children


@st.cache_data
def code_lookup():
    """{'1100': row, '1101': row, ...} for every code in the list."""
    parents, children = load_codes()
    out = {r["code"]: r for r in parents.values()}
    for rows in children.values():
        for r in rows:
            out[r["code"]] = r
    return out


def has_general_code(prefix: str) -> bool:
    parents, _ = load_codes()
    return prefix in parents


def parent_info(prefix: str) -> dict:
    """Info for the XX00 code; a placeholder if the CSV has no XX00 row."""
    parents, _ = load_codes()
    return parents.get(
        prefix,
        {
            "code": f"{prefix}00",
            "name": f"JASC {prefix}xx",
            "description": "No general (XX00) entry exists for this category in "
            "the code list. Choose one of the specific codes.",
        },
    )


# ---------------------------------------------------------------- prediction
def predict_categories(text: str):
    """Return ALL distinct 2-digit categories, best first."""
    preprocess_text = load_nlp()
    model = load_model()

    clean = [preprocess_text(text)]
    classes = np.asarray(model.named_steps["classifier"].classes_)
    scores = np.atleast_2d(model.decision_function(clean))[0]

    prefixes = []
    for i in np.argsort(scores)[::-1]:
        prefix = str(classes[i]).strip()[:2]  # works for "11", "1100", 11
        if prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes, clean[0]


# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("prefixes", None)  # all 2-digit categories, ranked
ss.setdefault("cat_page", 0)
ss.setdefault("code_page", 0)
ss.setdefault("cleaned", "")
ss.setdefault("input_text", "")
ss.setdefault("selected_prefix", None)
ss.setdefault("detail", None)  # option currently shown in the details screen
ss.setdefault("final", None)
ss.setdefault("report", [])
ss.setdefault("show_report", False)
ss.setdefault("manual", None)  # None | "issue" | "code"
ss.setdefault("manual_issue", "")
ss.setdefault("heard", "")
ss.setdefault("feedback", "")
ss.setdefault("voice", "")


def run_classification(text):
    ss.prefixes, ss.cleaned = predict_categories(text)
    ss.cat_page = 0
    ss.code_page = 0
    ss.input_text = text
    ss.selected_prefix = None
    ss.detail = None
    ss.final = None
    if SHOW_DEBUG:
        print("preprocessed:", ss.cleaned)


def start_manual():
    """Manual entry: say the issue, then say the full code."""
    start_over()
    ss.manual = "issue"
    ss.manual_issue = ""


def go_home():
    ss.show_report = False
    start_over()


def send_to_report(code, name):
    # Replace this with your real report destination (database, CSV, API...)
    ss.report.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "description": ss.input_text,
            "code": code,
            "name": name,
        }
    )
    ss.final = {"code": code, "name": name}
    ss.detail = None


def start_over():
    ss.manual = None
    ss.prefixes = None
    ss.selected_prefix = None
    ss.detail = None
    ss.final = None


# ---------------------------------------------------------------- screens / items
def current_screen() -> str:
    if ss.show_report:
        return "report"
    if ss.final:
        return "final"
    if ss.manual:
        return f"manual_{ss.manual}"
    if ss.detail:
        return "detail"
    if ss.selected_prefix:
        return "codes"
    if ss.prefixes:
        return "categories"
    return "home"


def category_items():
    items = []
    for prefix in ss.prefixes or []:
        info = parent_info(prefix)
        items.append(
            {
                "kind": "category",
                "prefix": prefix,
                "code": info["code"],
                "name": info["name"],
                "description": info["description"],
                "tag": "",
            }
        )
    if items:
        items[0]["tag"] = "Best match"
    return items


def code_items():
    prefix = ss.selected_prefix
    parent = parent_info(prefix)
    _, children = load_codes()
    items = []
    if has_general_code(prefix):
        items.append(
            {
                "kind": "code",
                "prefix": prefix,
                "code": parent["code"],
                "name": parent["name"],
                "description": parent["description"],
                "tag": "General",
            }
        )
    for row in children.get(prefix, []):
        items.append(
            {
                "kind": "code",
                "prefix": prefix,
                "code": row["code"],
                "name": row["name"],
                "description": row["description"],
                "tag": "",
            }
        )
    return items


def page_of(items, page, size):
    return items[page * size : (page + 1) * size]


def activate(item):
    if item["kind"] == "category":
        ss.selected_prefix = item["prefix"]
        ss.code_page = 0
        ss.detail = None
    else:
        send_to_report(item["code"], item["name"])


# ---------------------------------------------------------------- voice commands
_WORDS = (
    "one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()
_ORDINALS = (
    "first second third fourth fifth sixth seventh eighth ninth tenth"
).split()
NUMS = {w: i for i, w in enumerate(_WORDS, start=1)}
NUMS.update({w: i for i, w in enumerate(_ORDINALS, start=1)})
NUMS.update({"to": 2, "too": 2, "for": 4})  # common speech-to-text homophones


def to_number(word):
    """'6', 'six' and 'sixth' all become 6."""
    return int(word) if word.isdigit() else NUMS.get(word)
CMD_REPORT = {"report", "show report", "open report", "view report"}
CMD_NEXT = {"next", "next page", "more", "show more", "not here"}
CMD_PREV = {"previous", "previous page", "prev", "last page"}
CMD_BACK = {"back", "go back", "close", "cancel"}
CMD_NEW = {
    "new", "start over", "another", "classify another", "new description",
    "edit", "edit description", "again",
    "new session", "start new session", "start a new session",
    "menu", "main menu", "main page", "home", "go home",
}
CMD_MENU = {
    "menu", "main menu", "main page", "home", "go home",
    "new session", "start new session", "start a new session",
}
CMD_MANUAL = {
    "manual entry", "manual", "manual mode", "manual entry mode", "enter manually",
}
DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "to": "2", "too": "2",
    "three": "3", "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9",
}
CMD_YES = {"select", "yes", "confirm", "use it", "use this", "ok", "okay"}

HINTS = {
    "home": "Speak the fault description  ·  “manual entry”  ·  “report”",
    "manual_issue": "Say your issue  ·  “menu” to go back",
    "manual_code": "Say the full code, e.g. “1100”  ·  “menu” to go back",
    "categories": "“select 2”  ·  “details 2”  ·  “next”  ·  “previous”  ·  “back”",
    "codes": "“select 2”  ·  “details 2”  ·  “next”  ·  “previous”  ·  “back”",
    "detail": "“select”  ·  “back”",
    "detail_category": "“select” for exact codes  ·  “back”",
    "final": "“report”  ·  “new” for a new session",
    "report": "“back”  ·  “new” for a new session",
}


def hint_for(screen):
    if screen == "detail" and ss.detail and ss.detail["kind"] == "category":
        return HINTS["detail_category"]
    return HINTS[screen]


def parse_code(t):
    """'1100', '11 00', 'one one zero zero' -> '1100'. None if not 4 digits."""
    digits = ""
    for tok in t.split():
        if tok.isdigit():
            digits += tok
        elif tok in DIGIT_WORDS:
            digits += DIGIT_WORDS[tok]
        else:
            return None
    return digits if len(digits) == 4 else None


def pick_number(t):
    m = re.fullmatch(r"(?:(?:select|pick|choose|number|option|item|use)\s+)?(\w+)", t)
    return to_number(m.group(1)) if m else None


def detail_number(t):
    m = re.fullmatch(
        r"(?:show\s+)?(?:details?|info|describe|more info)\s+(?:on\s+|for\s+|of\s+)?"
        r"(?:number\s+|option\s+)?(\w+)",
        t,
    )
    return to_number(m.group(1)) if m else None


def handle_voice():
    """Runs when a command is 'spoken' (Enter pressed on the command line)."""
    raw = ss.voice.strip()
    ss.voice = ""
    if not raw:
        return
    ss.heard = raw
    ss.feedback = ""
    t = " ".join(re.sub(r"[^\w\s]", " ", raw.lower()).split())
    screen = current_screen()

    if screen == "report":
        if t in CMD_BACK:
            ss.show_report = False  # return to wherever the report was opened from
        elif t in CMD_NEW:
            ss.show_report = False  # main page, new session
            start_over()
        else:
            ss.feedback = "Not understood. Say “back” or “new”."
        return

    if t in CMD_REPORT:
        ss.show_report = True
        return

    if t in CMD_MENU:  # works at any point
        go_home()
        return

    if t in CMD_MANUAL:  # works at any point (restarts if already in manual entry)
        start_manual()
        return

    if screen == "manual_issue":
        if t in CMD_BACK:
            start_over()
        else:
            ss.manual_issue = raw
            ss.manual = "code"
        return

    if screen == "manual_code":
        if t in CMD_BACK:
            ss.manual = "issue"
            return
        code = parse_code(t)
        if code is None:
            ss.feedback = "Say the 4-digit code, for example “1100”."
        elif code not in code_lookup():
            ss.feedback = f"Code {code} is not in the code list. Say it again."
        else:
            ss.input_text = ss.manual_issue
            ss.manual = None
            send_to_report(code, code_lookup()[code]["name"])
        return

    if screen == "home":  # anything said here is the fault description
        run_classification(raw)
        return

    if screen == "final":
        if t in CMD_NEW | CMD_BACK:
            start_over()
        else:
            ss.feedback = "Not understood. Say “report” or “new”."
        return

    if screen == "detail":
        if t in CMD_BACK:
            ss.detail = None
        elif t in CMD_YES:
            activate(ss.detail)  # category -> exact code menu; code -> report
        else:
            ss.feedback = "Not understood. Say “select” or “back”."
        return

    # categories / codes
    if screen == "categories":
        items, page_attr, size = category_items(), "cat_page", CATEGORY_PAGE_SIZE
    else:
        items, page_attr, size = code_items(), "code_page", CODE_PAGE_SIZE
    page = ss[page_attr]
    visible = page_of(items, page, size)

    n = detail_number(t)
    if n is not None:
        if 1 <= n <= len(visible):
            ss.detail = visible[n - 1]
        else:
            ss.feedback = f"There is no option {n} on this screen."
        return

    n = pick_number(t)
    if n is not None:
        if 1 <= n <= len(visible):
            activate(visible[n - 1])
        else:
            ss.feedback = f"There is no option {n} on this screen."
        return

    if t in CMD_NEXT:
        if (page + 1) * size < len(items):
            ss[page_attr] = page + 1
        else:
            ss.feedback = "That is the last page."
    elif t in CMD_PREV:
        if page > 0:
            ss[page_attr] = page - 1
        else:
            ss.feedback = "That is the first page."
    elif t in CMD_BACK:
        if screen == "codes":
            ss.selected_prefix = None
        else:
            start_over()
    elif t in CMD_NEW:
        start_over()
    else:
        ss.feedback = "Not understood. Try " + HINTS[screen].split("  ·  ")[0]


# ---------------------------------------------------------------- AR styling
AR_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@400;500;600&display=swap');

:root{
  --ice:#7FE7FF;
  --ice-soft:rgba(127,231,255,.28);
  --amber:#FFB84D;
  --text:#EAF8FB;
  --dim:#8FB0B8;
  --panel:rgba(8,22,28,.58);
  --hud-font:'Barlow Semi Condensed','Segoe UI',system-ui,sans-serif;
}

/* ---- the world behind the glasses: a dim hangar ---- */
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{overflow:hidden !important;}
.stApp{
  background:
    radial-gradient(ellipse 55% 38% at 50% 60%, rgba(96,140,160,.38), transparent 70%),
    repeating-linear-gradient(90deg, rgba(255,255,255,.04) 0 2px, transparent 2px 150px),
    linear-gradient(180deg,#2a3640 0%,#19222b 52%,#0d1217 52.3%,#06090c 100%);
  color:var(--text);
}
[data-testid="stHeader"],[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"],
[data-testid="stToolbar"],footer{display:none !important;}

/* ---- the lens: fixed size, never scrolls ---- */
[data-testid="stMainBlockContainer"]{
  position:relative;
  max-width:880px !important;
  margin:24px auto !important;
  height:calc(100vh - 48px);
  max-height:780px;
  overflow:hidden;
  padding:22px 34px 0 !important;
  border-radius:42px;
  background:rgba(3,9,12,.6);
  backdrop-filter:blur(1.5px);
  border:2px solid var(--ice-soft);
  box-shadow:
    0 0 0 100vmax rgba(0,0,0,.8),
    0 0 70px rgba(127,231,255,.16),
    inset 0 0 90px rgba(0,0,0,.6);
}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]{gap:.5rem;}

/* ---- type ---- */
.say b{color:var(--ice);font-weight:500;margin-right:10px;}
.stApp p,.stApp label,.stApp li,.stApp h1,.stApp h2,.stApp h3,
.stApp input,[data-testid="stMarkdownContainer"]{
  font-family:var(--hud-font);
  color:var(--text);
}

.hud-bar{display:flex;align-items:baseline;gap:12px;font-family:var(--hud-font);}
.hud-bar b{font-size:1.15rem;font-weight:600;color:var(--ice);letter-spacing:.04em;}
.hud-bar span{color:var(--dim);font-size:1rem;}
.hud-bar .sp{flex:1;}
.hud-bar .mic{color:var(--ice);font-size:.95rem;}
.hud-bar .cnt{color:var(--amber);font-size:.95rem;}

.hud-head{margin:10px 0 8px;}
.hud-head h2{font-family:var(--hud-font);font-weight:500;font-size:2rem;line-height:1.1;margin:8px 0 2px;padding:0;color:var(--text);}
.hud-head p{margin:0;color:var(--dim);font-size:1.1rem;}
.pips{display:flex;gap:8px;}
.pips i{width:34px;height:4px;border-radius:2px;background:rgba(127,231,255,.2);}
.pips i.on{background:var(--ice);box-shadow:0 0 10px var(--ice);}

.quote{font-family:var(--hud-font);font-size:1.1rem;color:var(--text);
  border-left:3px solid var(--amber);padding:2px 0 2px 12px;margin:0 0 6px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}

/* ---- option rows ---- */
.opt{display:flex;align-items:center;gap:16px;padding:11px 16px;margin:0 0 10px;
  background:var(--panel);border:1px solid var(--ice-soft);border-radius:18px;}
.opt .num{flex:none;width:38px;height:38px;border-radius:50%;border:1.5px solid var(--amber);
  color:var(--amber);font-family:var(--hud-font);font-size:1.3rem;font-weight:600;
  display:flex;align-items:center;justify-content:center;}
.opt .num.plain{border-color:transparent;color:var(--dim);font-size:.95rem;font-weight:400;width:auto;}
.opt .code{flex:none;font-family:var(--hud-font);font-size:1.7rem;font-weight:600;color:var(--ice);}
.opt .name{flex:1;min-width:0;font-family:var(--hud-font);font-size:1.25rem;color:var(--text);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.opt .tag{flex:none;padding:1px 10px;border:1px solid var(--amber);border-radius:999px;
  color:var(--amber);font-family:var(--hud-font);font-size:.9rem;}
.opt.top{border-color:rgba(255,184,77,.6);}
.opt .stack{flex:1;min-width:0;}
.opt .line{display:flex;align-items:baseline;gap:14px;}
.opt .sub{margin-top:2px;font-family:var(--hud-font);font-size:1.05rem;color:var(--dim);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}

.pageinfo{font-family:var(--hud-font);color:var(--dim);font-size:1rem;margin:0;}

.big-code{font-family:var(--hud-font);font-size:4.2rem;font-weight:600;color:var(--amber);
  text-shadow:0 0 24px rgba(255,184,77,.55);line-height:1;margin-top:22px;}
.big-name{font-family:var(--hud-font);font-size:1.8rem;margin:6px 0 4px;}
.hud-note{color:var(--dim);font-family:var(--hud-font);font-size:1.05rem;margin:6px 0 8px;}
.desc{font-family:var(--hud-font);font-size:1.25rem;line-height:1.4;color:var(--text);margin-top:10px;
  display:-webkit-box;-webkit-line-clamp:7;-webkit-box-orient:vertical;overflow:hidden;}
.say{font-family:var(--hud-font);font-size:1.2rem;color:var(--text);
  border-left:3px solid var(--ice);padding:2px 0 2px 12px;margin:0 0 8px;}

/* ---- voice line, pinned to the bottom of the lens ---- */
.st-key-voice_bar{position:absolute;left:34px;right:34px;bottom:26px;}
.st-key-voice_bar [data-testid="stVerticalBlock"]{gap:.6rem;}
.st-key-voice_bar [data-testid="stMarkdownContainer"]{margin-bottom:0 !important;}
.st-key-voice_bar [data-testid="stElementContainer"]{margin-bottom:0 !important;}
.heard{font-family:var(--hud-font);font-size:1.05rem;color:var(--dim);margin:0;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.heard.warn{color:var(--amber);}
.hints{font-family:var(--hud-font);font-size:1.05rem;color:var(--ice);margin:0 0 14px;}
.stTextInput input{
  background:rgba(0,0,0,.55) !important;
  color:var(--text) !important;
  font-size:1.2rem !important;
  border:1.5px solid rgba(255,184,77,.7) !important;
  border-radius:999px !important;
  padding:14px 22px !important;
}
.stTextInput input:focus{border-color:var(--amber) !important;box-shadow:0 0 0 3px rgba(255,184,77,.2) !important;}
.stTextInput input::placeholder{color:#6b858c;}
[data-testid="InputInstructions"]{display:none !important;}
[data-testid="stTextInput"] > div,[data-testid="stTextInput"] [data-baseweb]{background:transparent !important;border:none !important;}
"""

st.markdown(f"<style>{AR_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- UI helpers
def esc(s) -> str:
    return html.escape(str(s))


def hud_header(title: str, step: int, subtitle: str = ""):
    pips = "".join(f'<i class="{"on" if i <= step else ""}"></i>' for i in (1, 2, 3))
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f'<div class="hud-head"><div class="pips" role="img" aria-label="Step {step} of 3">'
        f"{pips}</div><h2>{esc(title)}</h2>{sub}</div>",
        unsafe_allow_html=True,
    )


def status_bar():
    st.markdown(
        '<div class="hud-bar"><b>JASC</b><span>Fault classifier</span><i class="sp"></i>'
        f'<span class="cnt">Report {len(ss.report)}</span><span class="mic">● Listening</span></div>',
        unsafe_allow_html=True,
    )


def opt_row(num, code, name, tag="", top=False, plain=False):
    num_cls = "num plain" if plain else "num"
    tag_html = f'<span class="tag">{esc(tag)}</span>' if tag else ""
    return (
        f'<div class="opt{" top" if top else ""}">'
        f'<span class="{num_cls}">{esc(num)}</span>'
        f'<span class="code">{esc(code)}</span>'
        f'<span class="name">{esc(name)}</span>{tag_html}</div>'
    )


def report_row(time, code, name, description):
    return (
        '<div class="opt">'
        f'<span class="num plain">{esc(time)}</span>'
        '<div class="stack"><div class="line">'
        f'<span class="code">{esc(code)}</span><span class="name">{esc(name)}</span></div>'
        f'<div class="sub">“{esc(description)}”</div></div></div>'
    )


def render_options(visible, items, page, size, top_first=False):
    rows = "".join(
        opt_row(i, it["code"], it["name"], it["tag"], top=(top_first and page == 0 and i == 1))
        for i, it in enumerate(visible, start=1)
    )
    st.markdown(rows, unsafe_allow_html=True)
    first = page * size + 1
    last = page * size + len(visible)
    pages = max(1, -(-len(items) // size))
    st.markdown(
        f'<div class="pageinfo">{first}–{last} of {len(items)}  ·  page {page + 1} of {pages}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- boot
with st.spinner("Loading model and language resources..."):
    load_nlp()
    load_model()
    load_codes()

status_bar()
screen = current_screen()

# ---------------------------------------------------------------- screens
if screen == "report":
    hud_header("Sent to report", 3)
    latest = list(reversed(ss.report))[:REPORT_ROWS]
    if not latest:
        st.markdown('<div class="hud-note">Nothing sent yet.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            "".join(
                report_row(e["time"], e["code"], e["name"], e["description"])
                for e in latest
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="pageinfo">Latest {len(latest)} of {len(ss.report)}</div>',
            unsafe_allow_html=True,
        )

elif screen == "final":
    hud_header("Sent to report", 3)
    st.markdown(
        f'<div class="big-code">{esc(ss.final["code"])}</div>'
        f'<div class="big-name">{esc(ss.final["name"])}</div>'
        '<div class="hud-note">Added to this session’s report.</div>'
        '<div class="say"><b>“report”</b>view the report</div>'
        '<div class="say"><b>“new”</b>main menu, start a new session</div>',
        unsafe_allow_html=True,
    )

elif screen == "manual_issue":
    hud_header("Manual entry", 1, "Say your issue.")
    st.markdown(
        '<div class="hud-note">Next, you will be asked for the full code.</div>',
        unsafe_allow_html=True,
    )

elif screen == "manual_code":
    hud_header("Manual entry", 2, "Say the full code.")
    st.markdown(f'<div class="quote">“{esc(ss.manual_issue)}”</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hud-note">Four digits, for example</div>'
        '<div class="say">“1100”</div>'
        '<div class="say">“one one zero zero”</div>',
        unsafe_allow_html=True,
    )

elif screen == "detail":
    d = ss.detail
    hud_header("Details", 2 if d["kind"] == "category" else 3)
    st.markdown(
        f'<div class="opt top">{"" }'
        f'<span class="code">{esc(d["code"])}</span>'
        f'<span class="name">{esc(d["name"])}</span>'
        + (f'<span class="tag">{esc(d["tag"])}</span>' if d["tag"] else "")
        + "</div>"
        f'<div class="desc">{esc(d["description"] or "No description available.")}</div>',
        unsafe_allow_html=True,
    )

elif screen == "codes":
    parent = parent_info(ss.selected_prefix)
    hud_header("Pick the exact code", 3, f"{parent['code']} · {parent['name']}")
    items = code_items()
    if not items:
        st.markdown(
            '<div class="hud-note">No specific codes exist under this category. Say “back”.</div>',
            unsafe_allow_html=True,
        )
    else:
        visible = page_of(items, ss.code_page, CODE_PAGE_SIZE)
        render_options(visible, items, ss.code_page, CODE_PAGE_SIZE)

elif screen == "categories":
    hud_header("Likely categories", 2)
    st.markdown(f'<div class="quote">“{esc(ss.input_text)}”</div>', unsafe_allow_html=True)
    items = category_items()
    visible = page_of(items, ss.cat_page, CATEGORY_PAGE_SIZE)
    render_options(visible, items, ss.cat_page, CATEGORY_PAGE_SIZE, top_first=True)

else:  # home
    hud_header("Describe the fault", 1, "Say what you see.")
    st.markdown('<div class="hud-note">For example</div>', unsafe_allow_html=True)
    st.markdown(
        "".join(f'<div class="say">“{esc(p)}”</div>' for p in VOICE_SAMPLES)
        + '<div class="hud-note">Already know the code?</div>'
        '<div class="say"><b>“manual entry”</b>enter the issue and code yourself</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- voice line
with st.container(key="voice_bar"):
    if ss.feedback:
        st.markdown(f'<div class="heard warn">{esc(ss.feedback)}</div>', unsafe_allow_html=True)
    elif ss.heard:
        st.markdown(f'<div class="heard">Heard: “{esc(ss.heard)}”</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hints">{esc(hint_for(screen))}</div>', unsafe_allow_html=True)
    st.text_input(
        "Voice command",
        key="voice",
        on_change=handle_voice,
        placeholder="🎙  Speak… (type what you would say, then press Enter)",
        label_visibility="collapsed",
    )