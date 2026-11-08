# import re
# import ssl
# from functools import lru_cache

# import certifi
# import joblib
# import nltk
# import numpy as np
# import pandas as pd
# import streamlit as st
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # ---------------------------------------------------------------- config
# MODEL_PATH = (
#     "/Users/nileshsaumyadasa/Documents/AIE410_Project/SVM_TFIDF/"
#     "Model_Results/jasc_linear_svm_Final.joblib"
# )
# # Must match whatever you used at training time
# NEGATIONS_TO_KEEP = {"no", "not", "nor", "without"}

# ssl._create_default_https_context = lambda: ssl.create_default_context(
#     cafile=certifi.where()
# )


# # ---------------------------------------------------------------- loaders
# @st.cache_resource
# def load_nlp():
#     """Download NLTK data once, then build the preprocessing function."""
#     for resource in ("stopwords", "wordnet", "omw-1.4"):
#         nltk.download(resource, quiet=True)

#     stop_words = set(stopwords.words("english")) - NEGATIONS_TO_KEEP
#     lemmatize = lru_cache(maxsize=None)(WordNetLemmatizer().lemmatize)

#     def preprocess_text(text: str) -> str:
#         tokens = re.sub(r"[\W_]+", " ", str(text).lower()).split()
#         return " ".join(lemmatize(t) for t in tokens if t not in stop_words)

#     return preprocess_text


# @st.cache_resource
# def load_model():
#     return joblib.load(MODEL_PATH)


# def predict(text: str, top_k: int = 5):
#     preprocess_text = load_nlp()
#     model = load_model()

#     clean = [preprocess_text(text)]
#     predicted = model.predict(clean)[0]

#     classes = np.asarray(model.named_steps["classifier"].classes_)
#     scores = np.atleast_2d(model.decision_function(clean))[0]

#     top = np.argsort(scores)[::-1][:top_k]
#     df = pd.DataFrame(
#         {"JASC category": classes[top], "Decision score": scores[top]}
#     )
#     return predicted, df, clean[0]


# # ---------------------------------------------------------------- UI
# st.set_page_config(page_title="JASC Fault Classifier", page_icon="🛠️")
# st.title("🛠️ JASC Fault Classifier")
# st.caption("Linear SVM ML Model · Enter a maintenance description to classify it")

# # Load everything up front so the first query is instant. These are cached,
# # so the work only happens once per server process.
# with st.spinner("Loading model and language resources..."):
#     load_nlp()
#     load_model()

# text = st.text_area(
#     "Maintenance / fault description",
#     placeholder="e.g. Cargo door warning indication remains illuminated with door secured.",
#     height=120,
# )
# top_k = st.slider("Number of candidate categories to show", 1, 10, 5)

# if st.button("Classify", type="primary"):
#     if not text.strip():
#         st.warning("Please enter a description first.")
#     else:
#         with st.spinner("Predicting..."):
#             predicted, df, cleaned = predict(text, top_k)

#         st.success(f"Predicted JASC category: **{predicted}**")

#         st.subheader(f"Top {top_k} candidates")
#         st.dataframe(df, hide_index=True, use_container_width=True)
#         #st.bar_chart(df.set_index("JASC category")["Decision score"])

#         with st.expander("Show preprocessed text"):
#             st.code(cleaned or "(empty after preprocessing)")


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
PAGE_SIZE = 5  # predicted categories revealed per step
# True  -> "My category isn't here" REPLACES the list with the next 5
#          (cleaner GUI; a "Previous" button lets the user go back)
# False -> the next 5 are ADDED below the ones already shown
HIDE_PREVIOUS_PAGES = True

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
    parents = {
        r["prefix"]: r for r in df[is_parent].to_dict("records")
    }
    children = {
        prefix: group.to_dict("records")
        for prefix, group in df[~is_parent].groupby("prefix")
    }
    return parents, children


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
ss.setdefault("shown", PAGE_SIZE)  # how many of them are visible
ss.setdefault("cleaned", "")
ss.setdefault("input_text", "")
ss.setdefault("selected_prefix", None)
ss.setdefault("final", None)
ss.setdefault("report", [])


def run_classification(text):
    ss.prefixes, ss.cleaned = predict_categories(text)
    ss.shown = PAGE_SIZE
    ss.input_text = text
    ss.selected_prefix = None
    ss.final = None


def show_more():
    ss.shown += PAGE_SIZE


def show_previous():
    ss.shown = max(PAGE_SIZE, ss.shown - PAGE_SIZE)


def choose_prefix(prefix):
    ss.selected_prefix = prefix


def back_to_categories():
    ss.selected_prefix = None


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


def start_over():
    ss.prefixes = None
    ss.selected_prefix = None
    ss.final = None


def code_row(key, code, name, description, on_click, args, button_label="Select"):
    """One row: code + name | select button | collapsible details."""
    label, pick, details = st.columns([4, 1.3, 2.7], vertical_alignment="top")
    label.markdown(f"**{code}** · {name}")
    pick.button(button_label, key=f"pick_{key}", on_click=on_click, args=args)
    with details.expander("View details"):
        st.caption(f"Code {code}")
        st.write(description or "No description available.")


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="JASC Fault Classifier", page_icon="🛠️", layout="wide")
st.title("🛠️ JASC Fault Classifier")
st.caption("Linear SVM + TF-IDF · describe the fault, then pick the exact JASC code")

# Load everything up front so the first query is instant (cached per process)
with st.spinner("Loading model and language resources..."):
    load_nlp()
    load_model()
    load_codes()

# Sidebar: running report
with st.sidebar:
    st.header("📋 Report")
    if not ss.report:
        st.caption("No codes sent yet.")
    for entry in reversed(ss.report):
        st.markdown(f"**{entry['code']}** · {entry['name']}")
        st.caption(f"{entry['time']} — {entry['description']}")

# Step 0: input
text = st.text_area(
    "Maintenance / fault description",
    placeholder="e.g. Cabin door does not latch securely when closed.",
    height=120,
)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter a description first.")
    else:
        with st.spinner("Predicting..."):
            run_classification(text)

st.divider()

# Step 3: final confirmation
if ss.final:
    st.success(
        f"✅ Code **{ss.final['code']} – {ss.final['name']}** has been sent to the report."
    )
    st.button("Classify another description", on_click=start_over)

# Step 2: specific 4-digit codes for the chosen category
elif ss.selected_prefix:
    parent = parent_info(ss.selected_prefix)
    _, children = load_codes()
    subcodes = children.get(ss.selected_prefix, [])

    st.button("← Back to categories", on_click=back_to_categories)
    st.subheader(f"Specific codes in {parent['code']} · {parent['name']}")
    st.caption("Select the code that best matches the fault.")

    if has_general_code(ss.selected_prefix):
        code_row(
            f"general_{parent['code']}", parent["code"], parent["name"],
            parent["description"], send_to_report, (parent["code"], parent["name"]),
            button_label="Use general code",
        )
        st.caption("☝️ General code: use it if none of the specific codes below fit.")
        st.divider()

    if not subcodes:
        st.info("No specific codes exist under this category.")
    for row in subcodes:
        code_row(
            row["code"], row["code"], row["name"], row["description"],
            send_to_report, (row["code"], row["name"]),
        )

# Step 1: predicted categories
elif ss.prefixes:
    st.subheader("Predicted categories")
    st.caption(f"Description: “{ss.input_text}”")
    start = max(0, ss.shown - PAGE_SIZE) if HIDE_PREVIOUS_PAGES else 0
    visible = ss.prefixes[start : ss.shown]
    for rank, prefix in enumerate(visible, start=start + 1):
        info = parent_info(prefix)
        suffix = "  ·  _top prediction_" if rank == 1 else ""
        label, pick, details = st.columns([4, 1.3, 2.7], vertical_alignment="top")
        label.markdown(f"**{info['code']}** · {info['name']}{suffix}")
        pick.button("Select", key=f"cat_{prefix}", on_click=choose_prefix, args=(prefix,))
        with details.expander("View details"):
            st.caption(f"Code {info['code']}")
            st.write(info["description"] or "No description available.")

    remaining = len(ss.prefixes) - ss.shown
    if HIDE_PREVIOUS_PAGES and start > 0:
        st.button("← Previous categories", on_click=show_previous)
    if remaining > 0:
        nxt = min(PAGE_SIZE, remaining)
        st.button(
            f"🔎 My category isn't here — show next {nxt}",
            on_click=show_more,
        )
    else:
        st.info(
            "That's every category the model knows. If none fit, "
            "start over with a more detailed description."
        )
        st.button("Classify another description", on_click=start_over)

    with st.expander("Show preprocessed text"):
        st.code(ss.cleaned or "(empty after preprocessing)")