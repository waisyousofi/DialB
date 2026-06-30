#!/usr/bin/env python3
"""
Streamlit visualization for beam_trace_inference.py output.

Run:
streamlit run beam_trace_viewer.py

Then upload/select the JSON trace file.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Beam Token Trace Viewer", layout="wide")
st.title("Beam Search Next-Token Trace Viewer")

st.markdown(
    "Upload a trace JSON file created by `beam_trace_inference.py`. "
    "Use the slider or Next/Previous buttons to inspect each decoding step."
)

uploaded = st.file_uploader("Trace JSON", type=["json"])
path_text = st.text_input("Or enter local JSON path", value="beam_trace.json")

trace = None
if uploaded is not None:
    trace = json.load(uploaded)
elif path_text and Path(path_text).exists():
    with open(path_text, "r", encoding="utf-8") as f:
        trace = json.load(f)

if trace is None:
    st.info("Upload a JSON trace or enter a valid local path.")
    st.stop()

steps = trace.get("trace_steps", [])
if not steps:
    st.error("No trace_steps found in this JSON file.")
    st.stop()

metadata = trace.get("metadata", {})
with st.expander("Metadata", expanded=True):
    st.write("**Model:**", metadata.get("model_name", ""))
    st.write("**Source sentence:**", metadata.get("source_sentence", ""))
    st.write("**Best output:**", trace.get("best_output", ""))

if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0

c1, c2, c3 = st.columns([1, 1, 5])
with c1:
    if st.button("Previous", use_container_width=True):
        st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
with c2:
    if st.button("Next", use_container_width=True):
        st.session_state.step_idx = min(len(steps) - 1, st.session_state.step_idx + 1)
with c3:
    st.session_state.step_idx = st.slider(
        "Decoding step",
        min_value=0,
        max_value=len(steps) - 1,
        value=st.session_state.step_idx,
        format="step %d",
    )

step = steps[st.session_state.step_idx]
st.header(f"Step {step['step']}")

selected = step.get("selected_expansions", [])
selected_pairs = {(x["from_beam_pos"], x["token_id"]): x for x in selected}

st.subheader("Selected beam expansions")
selected_df = pd.DataFrame(selected)
if not selected_df.empty:
    st.dataframe(
        selected_df[
            [
                "new_beam_pos",
                "from_beam_pos",
                "token",
                "token_id",
                "token_logprob",
                "new_score",
                "finished",
                "new_text",
            ]
        ],
        use_container_width=True,
    )
else:
    st.write("No selected expansions found.")

st.subheader("Top next-token candidates for each beam")

for beam in step.get("beams_before_selection", []):
    beam_pos = beam["beam_pos"]
    st.markdown(f"### Beam {beam_pos} | score before = `{beam['beam_score_before']:.4f}`")
    if beam.get("prefix_text"):
        st.code(beam["prefix_text"])

    rows = []
    for cand in beam.get("top_candidates", []):
        picked = (beam_pos, cand["token_id"]) in selected_pairs
        selected_info = selected_pairs.get((beam_pos, cand["token_id"]), {})
        rows.append(
            {
                "picked?": "✅" if picked else "",
                "new_beam_pos": selected_info.get("new_beam_pos", ""),
                "rank": cand.get("rank"),
                "token": cand.get("token"),
                "token_id": cand.get("token_id"),
                "prob": cand.get("prob"),
                "token_logprob": cand.get("token_logprob"),
                "cumulative_score": cand.get("cumulative_score"),
                "eos?": cand.get("is_eos"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Final beams")
final_df = pd.DataFrame(trace.get("final_beams", []))
if not final_df.empty:
    st.dataframe(final_df, use_container_width=True)
