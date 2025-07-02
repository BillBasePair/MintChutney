#===== Imports, Setup, and Utility Functions =========
# MintChutney_v250506.py
# SELEX Analysis Dashboard for FASTQ DNA sequences
# By Bill Jackson, Ph.D.
# Base Pair Biotechnologies, Inc.
# May 6, 2025
# Features: FASTQ processing, motif analysis, candidate scoring, ViennaRNA folding, RNAComposer inputs, clipboard FASTA copy
# Updates (Original): Restored V11's Overview tab behavior, fixed StreamlitMixedNumericTypesError, etc.
# Cosmetic Changes (May 6, 2025):
# 1. Changed tab names to "File Upload & Parameters", "SELEX metrics", "Motif Analysis", "Final Candidates".
# 2. In "SELEX metrics" tab, sorted "Frequency vs. Rounds" plot legend by frequency in the final round (highest to lowest).
# 3. Enhanced "Frequency vs. Rounds" plot legend clarity with better colors, larger font, and line width variation.
# 4. Adjusted "Rank Abundance Curve" plot annotations to prevent overlap and ensure ordered placement.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from collections import defaultdict, Counter
import os
import time
import random
import zipfile
import RNA
from Levenshtein import distance
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
import subprocess
import re
import datetime
from itertools import islice
from scipy.stats import linregress
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="MintChutney SELEX Analysis", layout="wide")

# Initialize session state
if "files" not in st.session_state:
    st.session_state["files"] = []
    st.session_state["round_map"] = {}
    st.session_state["params"] = {}
    st.session_state["motif_data"] = {}
    st.session_state["motif_counts"] = {}
    st.session_state["run_log"] = []
    st.session_state["processing_complete"] = False
    st.session_state["calculations_complete"] = False
    st.session_state["output_dir"] = None
    st.session_state["plot_counter"] = 0
    st.session_state["accessibility_weight"] = 0.5
    st.session_state["quikfold_params"] = {
        "foldtype": "DNA4.0",
        "temperature": 25.0,
        "na_conc": 137.0,
        "mg_conc": 1.0,
        "polymer": "0",
        "max_structures": "1"
    }
    st.session_state["analysis_start_time"] = None  # Added for elapsed time tracking

# Script path for display
SCRIPT_PATH = os.path.basename(__file__)

# Inject CSS to scale down elements (from V11)
st.markdown("""
    <style>
    .css-1v0mbdj, .css-1v3fvcr, .css-1cpxqw2, .css-18ni7ap {
        transform: scale(0.9);
        transform-origin: top left;
    }
    </style>
    """, unsafe_allow_html=True)

# Header and scoring formula (from V11)
st.markdown("# 🌿 MintChutney SELEX Analysis\n**Where good aptamers blend.**\nby Bill Jackson, Ph.D.")
st.markdown("> **Score_1 = Frequency × |ΔG|**  \n> **Score_2 = Score_1 × (1 + Accessibility Weight × (Motif Accessibility - 0.5))**")

# Define tabs with updated names
tabs = st.tabs(["File Upload & Parameters", "SELEX metrics", "Motif Analysis", "Final Candidates"])

# Utility functions (unchanged)
def preprocess_sequence(seq, seq_type):
    """Preprocess sequence based on type."""
    if seq_type == "DNA":
        seq = seq.replace("T", "U")
    return seq

def has_repetitive_pattern(seq, max_repeat_length=4):
    """Check for repetitive patterns in sequence."""
    for length in range(2, max_repeat_length + 1):
        for i in range(len(seq) - length + 1):
            pattern = seq[i:i+length]
            if pattern * 3 in seq:
                return True
    return False

def validate_dot_bracket(dot_bracket, seq):
    """Validate dot-bracket structure."""
    st.session_state["run_log"].append(f"Validating dot-bracket: seq_len={len(seq)}, dot_len={len(dot_bracket)}")
    if len(dot_bracket) != len(seq):
        return False, f"Length mismatch: dot-bracket ({len(dot_bracket)}) != sequence ({len(seq)})"
    stack = []
    for i, char in enumerate(dot_bracket):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if not stack:
                return False, "Unmatched closing parenthesis"
            stack.pop()
        elif char != ".":
            return False, f"Invalid character in dot-bracket: {char}"
    if stack:
        return False, f"Unmatched opening parenthesis at positions {stack}"
    return True, ""

def display_rnacomposer_input(seq, dot_bracket, prefix="Candidate"):
    """Display RNAComposer input."""
    rna_seq = seq.replace("T", "U")
    st.session_state["run_log"].append(f"RNAComposer for {prefix}: Sequence={rna_seq[:20]}..., Structure={dot_bracket[:20]}...")
    st.markdown(f"**RNAComposer Input for {prefix}:**")
    st.markdown("**FASTA Sequence:**")
    st.code(f">{prefix}_RNA\n{rna_seq}", language="plaintext")
    st.markdown("**Dot-Bracket Structure:**")
    st.code(dot_bracket, language="plaintext")
    st.markdown(f"Copy the above lines and paste them into [RNAComposer](http://rnacomposer.cs.put.poznan.pl/).")

def plot_dna_structure(fasta, output_path, candidate_id):
    """Generate secondary structure plot using ViennaRNA with improved visualization."""
    try:
        st.session_state["run_log"].append(f"Attempting ViennaRNA folding for {candidate_id}: {fasta[:50]}...")
        
        # Extract sequence from FASTA
        seq = fasta.split("\n")[1].strip()
        quikfold_params = st.session_state["quikfold_params"]

        # Configure ViennaRNA
        RNA.cvar.temperature = quikfold_params["temperature"]
        RNA.cvar.dangles = st.session_state["params"]["dangling_ends"]
        RNA.cvar.noGU = 0

        # Fold sequence
        dot_bracket, mfe = RNA.fold(seq)  # Direct folding (no T->U for DNA consistency)
        st.session_state["run_log"].append(f"ViennaRNA folding successful for {candidate_id}: Dot-Bracket={dot_bracket}, MFE={mfe:.2f}")

        # Generate 2D plot
        coords = RNA.get_xy_coordinates(dot_bracket)  # Get coordinates
        x = []
        y = []
        for i in range(len(seq)):  # Iterate over sequence positions (0-based)
            coord = coords.get(i + 1)  # 1-based indexing
            if coord:
                x.append(coord.X)
                y.append(coord.Y)
            else:
                x.append(0.0)
                y.append(0.0)
        st.session_state["run_log"].append(f"Coordinates for {candidate_id}: x={x[:10]}..., y={y[:10]}..., len={len(x)}")

        # Parse dot-bracket to find base pairs
        stack = []
        pairs = {}  # Map i -> j for paired bases
        for i, char in enumerate(dot_bracket):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    pairs[j] = i
                    pairs[i] = j

        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot the backbone, distinguishing paired and unpaired regions
        i = 0
        while i < len(seq) - 1:
            if i in pairs:  # Start of a paired region (stem)
                j = pairs[i]
                # Plot the stem as straight lines from i to the paired base j
                stem_start = i
                stem_end = j
                # Find the contiguous paired region
                while i in pairs and pairs[i] == j and i < len(seq) - 1:
                    i += 1
                    j -= 1
                # Plot the stem backbone (i-side)
                stem_x_i = x[stem_start:i+1]
                stem_y_i = y[stem_start:i+1]
                ax.plot(stem_x_i, stem_y_i, 'b-', linewidth=1.5, label='Backbone' if stem_start == 0 else "")
                # Plot the stem backbone (j-side)
                stem_x_j = x[j:stem_end+1]
                stem_y_j = y[j:stem_end+1]
                ax.plot(stem_x_j, stem_y_j, 'b-', linewidth=1.5)
                # Draw base-pair lines between paired bases
                for k in range(stem_start, i):
                    pair_k = pairs[k]
                    ax.plot([x[k], x[pair_k]], [y[k], y[pair_k]], 'k--', linewidth=0.5, alpha=0.5)  # Dashed lines for base pairs
            else:  # Unpaired region (loop, bulge, etc.)
                start = i
                while i < len(seq) - 1 and i not in pairs:
                    i += 1
                # Plot the unpaired region as a curve
                loop_x = x[start:i+1]
                loop_y = y[start:i+1]
                ax.plot(loop_x, loop_y, 'r-', linewidth=1.5, label='Unpaired' if start == 0 else "")

        # Color-code nucleotides (A, U, G, C)
        nucleotide_colors = {'A': 'green', 'U': 'red', 'G': 'blue', 'C': 'orange'}
        for i, (xi, yi) in enumerate(zip(x, y)):
            if i % 3 == 0 or i in pairs:  # Label every third position or paired bases
                color = nucleotide_colors.get(seq[i], 'black')
                ax.text(xi, yi, seq[i], fontsize=5, ha='center', va='center', color=color, rotation=45)

        ax.set_title(f"Secondary Structure for {candidate_id} (ViennaRNA, MFE={mfe:.2f})", fontsize=10)
        ax.axis('equal')
        ax.axis('off')
        ax.legend(loc='upper right', fontsize=8)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Display in Streamlit
        st.image(output_path, width=400)
        
        st.session_state["run_log"].append(f"ViennaRNA 2D structure plot saved for {candidate_id} at {output_path}")
        return {"success": True, "dot_bracket": dot_bracket, "source": "ViennaRNA"}
    
    except Exception as e:
        error_msg = f"Failed to generate structure for {candidate_id}: {str(e)}"
        st.session_state["run_log"].append(error_msg)
        st.error(error_msg)
        return {"success": False, "dot_bracket": "", "source": "None"}
#===== End Imports, Setup, and Utility Functions ======

#===== File Upload & Parameters Tab =========
with tabs[0]:
    st.header("📁 Upload and Configure SELEX Parameters")
    st.markdown(f"**Running**: `{SCRIPT_PATH}`")
    
    uploaded_files = st.file_uploader(
        "Upload FASTQ Files (select multiple)", type=["fastq", "fq"], accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.number_input("Top N sequences", min_value=10, max_value=1000, value=50, step=10)
            primer_fwd_len = st.number_input("Forward primer length", min_value=0, max_value=30, value=19)
            primer_rev_len = st.number_input("Reverse primer length", min_value=0, max_value=30, value=19)
            seq_type = st.selectbox("Sequence Type", options=["DNA", "RNA"], index=0)
        with col2:
            temperature = st.number_input("Folding Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
            dangling_ends = st.selectbox("Dangling Ends", options=["None (0)", "Some (1)", "All (2)"], index=2)
            motif_lengths = st.multiselect("Motif lengths", options=list(range(5, 19)), default=[9, 15])
            output_name = st.text_input("Output directory name", value="mintchutney_results", help="All output files will be saved in a folder named after this, with a timestamp.")

        artifact_threshold = st.slider("Artifact Filter: Max % of Single Nucleotide", min_value=50, max_value=100, value=80, step=5, help="Sequences with more than this percentage of any single nucleotide (e.g., GGGG...) will be filtered out.")

        # Quikfold settings
        st.subheader("Folding Settings")
        st.markdown(
            """
            Configure settings for DNA/RNA folding in ViennaRNA. These apply to secondary structure predictions in the Final Candidates tab.
            Defaults: [Na+]=137 mM (physiological concentration in PBS), [Mg2+]=1 mM (SELEX standard for aptamer folding).
            """
        )
        foldtype = st.selectbox(
            "Energy Rules",
            ["DNA4.0", "DNA3.0", "RNA4.0", "RNA3.0"],
            index=0,
            help="Select DNA or RNA energy parameters. Use DNA4.0 for DNA aptamers (default)."
        )
        st.session_state["quikfold_params"]["foldtype"] = foldtype
        quikfold_temp = st.number_input(
            "Folding Temperature (°C)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state["quikfold_params"]["temperature"],
            step=0.1,
            help="Select folding temperature (0-100°C). Default: 25°C."
        )
        st.session_state["quikfold_params"]["temperature"] = quikfold_temp
        na_conc = st.number_input(
            "[Na+] Concentration (mM)",
            min_value=0.0,
            max_value=1000.0,
            value=st.session_state["quikfold_params"]["na_conc"],
            step=10.0,
            help="Sodium concentration in millimolar (0-1000 mM). Default: 137 mM (PBS)."
        )
        st.session_state["quikfold_params"]["na_conc"] = na_conc
        mg_conc = st.number_input(
            "[Mg2+] Concentration (mM)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state["quikfold_params"]["mg_conc"],
            step=1.0,
            help="Magnesium concentration in millimolar (0-100 mM). Default: 1 mM (SELEX standard)."
        )
        st.session_state["quikfold_params"]["mg_conc"] = mg_conc

        round_options = ["Unassigned"] + [f"Round {i}" for i in range(1, 21)]
        round_assignments = {}
        for file in uploaded_files:
            sel = st.selectbox(f"Round for {file.name}", options=round_options, key=f"round_{file.name}")
            round_assignments[file.name] = sel

        if st.button("🚀 Run Analysis"):
            if "Unassigned" in round_assignments.values():
                st.error("Please assign SELEX rounds to all files.")
            else:
                valid_files = []
                for f in uploaded_files:
                    try:
                        text = f.getvalue().decode("utf-8")
                        parser = SeqIO.parse(io.StringIO(text), "fastq")
                        if next(parser, None) is None:
                            st.warning(f"{f.name} appears empty or invalid.")
                            continue
                        valid_files.append(f)
                    except UnicodeDecodeError:
                        st.error(f"{f.name} is not a valid text file. Save as UTF-8 encoded FASTQ.")
                    except Exception as e:
                        st.error(f"Invalid FASTQ file {f.name}: {str(e)}")
                if not valid_files:
                    st.error("No valid FASTQ files uploaded.")
                else:
                    # Store start time for elapsed time tracking
                    st.session_state["analysis_start_time"] = time.time()
                    st.session_state["files"] = valid_files
                    st.session_state["round_map"] = {f.name: round_assignments[f.name] for f in valid_files}
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = os.path.join(os.getcwd(), f"{output_name}_{timestamp}")
                    os.makedirs(output_dir, exist_ok=True)
                    st.session_state["output_dir"] = output_dir
                    st.session_state["params"] = {
                        "top_n": top_n,
                        "primer_fwd": primer_fwd_len,
                        "primer_rev": primer_rev_len,
                        "seq_type": seq_type,
                        "temperature": temperature,
                        "dangling_ends": int(dangling_ends.split("(")[1].split(")")[0]),
                        "motif_lengths": motif_lengths,
                        "output_name": output_name,
                        "filter_near_boundaries": False,
                        "artifact_threshold": artifact_threshold / 100.0
                    }
                    st.session_state["run_log"].append(f"Run started: {datetime.datetime.now()}")
                    st.session_state["run_log"].append(f"Parameters: {st.session_state['params']}")
                    st.session_state["run_log"].append(f"Round assignments: {st.session_state['round_map']}")
                    st.success(f"✅ Parameters saved. Output will be saved in `{output_dir}`. Proceed to next tab.")
#===== End File Upload & Parameters Tab ======

#===== SELEX metrics Tab =========
with tabs[1]:
    st.header("📊 SELEX Effectiveness Dashboard")
    st.markdown(f"**Running**: `{SCRIPT_PATH}`")
    
    round_data = defaultdict(Counter)
    if "files" in st.session_state and st.session_state["files"]:
        unpacking_messages = [
            "🧬 Unpacking DNA sequences like a curious scientist...",
            "📦 Opening sequence boxes like it's Christmas morning...",
            "🧵 Unraveling DNA threads with a detective's focus...",
            "🔬 Peering into sequences with a microscope of wonder...",
            "🪢 Untangling DNA knots like a puzzle master..."
        ]
        with st.spinner(random.choice(unpacking_messages)):
            start_time = time.time()
            for f in st.session_state["files"]:
                round_label = st.session_state["round_map"].get(f.name, "Unassigned")
                st.session_state["run_log"].append(f"Processing file: {f.name}, Round: {round_label}")
                try:
                    text = f.getvalue().decode("utf-8")
                    parser = SeqIO.parse(io.StringIO(text), "fastq")
                    chunk_size = 10000
                    seq_count = 0
                    skipped_count = 0
                    while True:
                        chunk = list(islice(parser, chunk_size))
                        if not chunk:
                            break
                        for r in chunk:
                            full = str(r.seq).upper()
                            primer_fwd = st.session_state["params"]["primer_fwd"]
                            primer_rev = st.session_state["params"]["primer_rev"]
                            if len(full) < primer_fwd + primer_rev + 5:
                                skipped_count += 1
                                continue
                            trimmed = full[primer_fwd:-primer_rev if primer_rev > 0 else None]
                            if (len(trimmed) >= 5 and set(trimmed).issubset("ACGTN") and
                                not any(trimmed.count(n) / len(trimmed) > st.session_state["params"]["artifact_threshold"] for n in "ACGT") and
                                not has_repetitive_pattern(trimmed)):
                                if round_label != "Unassigned":
                                    round_data[round_label][trimmed] += 1
                                    seq_count += 1
                                else:
                                    skipped_count += 1
                            else:
                                skipped_count += 1
                    st.session_state["run_log"].append(f"File {f.name}: {seq_count} sequences processed, {skipped_count} skipped")
                except Exception as e:
                    st.session_state["run_log"].append(f"Error processing {f.name}: {str(e)}")
                    round_data.clear()
                    break

            if round_data:
                sorted_rounds = sorted(
                    round_data.keys(),
                    key=lambda x: int(x.split()[-1]) if x != "Unassigned" else 0
                )
                unique_seqs = []
                total_seqs = []
                for r in sorted_rounds:
                    unique_seqs.append(len(round_data[r]))
                    total_seqs.append(sum(round_data[r].values()))

                st.subheader("🩺 SELEX Health Dashboard")
                diversity_messages = [
                    "🧮 Calculating diversity metrics with a sprinkle of magic...",
                    "📊 Conjuring up diversity stats like a wizard...",
                    "🔢 Crunching numbers with a mathematician's glee...",
                    "🌟 Measuring diversity like a star gazer...",
                    "🎲 Rolling the dice on diversity calculations..."
                ]
                with st.spinner(random.choice(diversity_messages)):
                    def shannon_entropy(counts):
                        total = sum(counts)
                        if total == 0:
                            return 0
                        probs = np.array(counts) / total
                        probs = probs[probs > 0]
                        return -np.sum(probs * np.log2(probs))

                    entropy_values = []
                    for r in sorted_rounds:
                        counts = list(round_data[r].values())
                        entropy = shannon_entropy(counts)
                        entropy_values.append(entropy)

                    entropy_final = entropy_values[-1]
                    entropy_note = (
                        "Optimal diversity (balanced exploration and convergence)" if 6 <= entropy_final <= 10 else
                        "High diversity (selection may be too weak)" if entropy_final > 10 else
                        "Low diversity (potential over-convergence or bias)"
                    )

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 2]})
                    max_entropy = 20
                    effective_seqs = 2 ** entropy_final
                    max_effective_seqs = 2 ** max_entropy
                    tube_height = max_entropy * 0.9
                    bulb_center = (0.5, 0)
                    bulb_radius = 0.15
                    bulb = plt.Circle(bulb_center, bulb_radius, color="lightgray", zorder=1)
                    ax1.add_patch(bulb)
                    tube_left = 0.4
                    tube_right = 0.6
                    tube_bottom = bulb_radius
                    tube_top = tube_height + bulb_radius
                    ax1.fill_betweenx(
                        y=np.linspace(tube_bottom, tube_top, 100),
                        x1=tube_left,
                        x2=tube_right,
                        color="lightgray",
                        zorder=1
                    )
                    fill_height = (entropy_final / max_entropy) * tube_height + bulb_radius
                    ax1.fill_betweenx(
                        y=np.linspace(tube_bottom, fill_height, 100),
                        x1=tube_left,
                        x2=tube_right,
                        color="orange",
                        zorder=2
                    )
                    ax1.hlines(y=fill_height, xmin=tube_left, xmax=tube_right, color="black", linewidth=1, zorder=3)
                    ax1.text(
                        tube_right + 0.05,
                        fill_height,
                        f"{entropy_final:.2f} bits\n≈{int(effective_seqs):,} seqs",
                        verticalalignment="center",
                        fontsize=10,
                        fontfamily="monospace",
                        zorder=4
                    )
                    ax1.set_ylim(0, max_entropy + bulb_radius + 1)
                    ax1.set_yticks(np.arange(0, max_entropy + 1, 5))
                    ax1.set_ylabel("Shannon Entropy (bits)")
                    ax1.set_xlim(0, 1)
                    ax1.set_xticks([])
                    ax1.set_title(f"Final Round Diversity")
                    ax1b = ax1.twinx()
                    ax1b.set_ylim(0, max_entropy + bulb_radius + 1)
                    ax1b.set_yticks(np.arange(0, max_entropy + 1, 5))
                    ax1b.set_yticklabels([f"{int(2**x):,}" for x in np.arange(0, max_entropy + 1, 5)])
                    ax1b.set_ylabel("Effective Number of Sequences")
                    ax2.plot(sorted_rounds, entropy_values, marker="o", color="green", linewidth=2)
                    ax2.set_xlabel("Round")
                    ax2.set_ylabel("Shannon Entropy (bits)")
                    ax2.set_title("Shannon Entropy Trend Across Rounds")
                    ax2.grid(True, linestyle="--", alpha=0.7)
                    ax2.set_xticks(sorted_rounds)
                    ax2.set_xticklabels(sorted_rounds, rotation=45, ha="right")
                    plt.tight_layout()
                    entropy_combined_path = os.path.join(st.session_state["output_dir"], "entropy_combined.png")
                    fig.savefig(entropy_combined_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    st.image(entropy_combined_path)
                    st.markdown(f"**Interpretation**: {entropy_note}")

                    unique_seq_changes = []
                    for i in range(1, len(unique_seqs)):
                        change = (unique_seqs[i-1] - unique_seqs[i]) / unique_seqs[i-1] * 100 if unique_seqs[i-1] > 0 else 0
                        unique_seq_changes.append(change)
                    avg_change = np.mean(unique_seq_changes) if unique_seq_changes else 0
                    unique_seq_status = (
                        "🟢" if 10 <= avg_change <= 50 else
                        "🟡" if (0 <= avg_change < 10 or 50 < avg_change <= 80) else
                        "🔴"
                    )
                    unique_seq_note = (
                        "Healthy convergence" if 10 <= avg_change <= 50 else
                        "Slow or rapid convergence" if (0 <= avg_change < 10 or 50 < avg_change <= 80) else
                        "Over-convergence or contamination"
                    )

                    final_round_total = sum(round_data[sorted_rounds[-1]].values())
                    top_seq_freq = max(round_data[sorted_rounds[-1]].values()) / final_round_total * 100 if final_round_total > 0 else 0
                    enrichment_status = (
                        "🟢" if 5 <= top_seq_freq <= 20 else
                        "🟡" if 20 < top_seq_freq <= 40 else
                        "🔴"
                    )
                    enrichment_note = (
                        "Balanced enrichment" if 5 <= top_seq_freq <= 20 else
                        "Strong enrichment (potential bias)" if 20 < top_seq_freq <= 40 else
                        "Over-dominance (risk of losing diversity)"
                    )

                    overall_status = "🟢" if 6 <= entropy_final <= 10 else "🟡" if entropy_final > 10 else "🔴"
                    overall_note = (
                        "SELEX process is in an optimal state" if 6 <= entropy_final <= 10 else
                        "SELEX process may need attention (diversity too high)" if entropy_final > 10 else
                        "SELEX process has potential issues (diversity too low)"
                    )

                    health_data = [
                        {"Metric": "Unique Sequence Change", "Value": f"{avg_change:.2f}% decrease", "Status": unique_seq_status, "Note": unique_seq_note},
                        {"Metric": "Top Sequence Dominance", "Value": f"{top_seq_freq:.2f}%", "Status": enrichment_status, "Note": enrichment_note},
                        {"Metric": "Overall Health", "Value": "", "Status": overall_status, "Note": overall_note}
                    ]
                    health_df = pd.DataFrame(health_data)
                    st.dataframe(health_df.style.set_properties(**{"font-family": "Courier New"}), use_container_width=True)

                st.subheader("Sequences per Round")
                bar_messages = [
                    "📊 Painting sequence bars with a dash of color...",
                    "🎨 Coloring sequence bars like an artist at work...",
                    "🖌️ Brushing up sequence bars with a splash of style...",
                    "🌈 Adding a rainbow of bars to the sequence plot...",
                    "🧑‍🎨 Crafting sequence bars like a masterpiece..."
                ]
                with st.spinner(random.choice(bar_messages)):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    bar_width = 0.35
                    index = np.arange(len(sorted_rounds))
                    ax.bar(index - bar_width/2, total_seqs, bar_width, label="Total Sequences", color="darkorange")
                    ax.bar(index + bar_width/2, unique_seqs, bar_width, label="Unique Sequences", color="dodgerblue")
                    ax.set_xlabel("Round")
                    ax.set_ylabel("Number of Sequences")
                    ax.set_xticks(index)
                    ax.set_xticklabels(sorted_rounds, rotation=45, ha="right")
                    ax.legend()
                    plt.tight_layout()
                    sequences_combined_path = os.path.join(st.session_state["output_dir"], "sequences_combined.png")
                    fig.savefig(sequences_combined_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    st.image(sequences_combined_path)

                st.subheader("Frequency vs. Rounds")
                enrichment_messages = [
                    "🌟 Tracking sequence stars across rounds...",
                    "🚀 Launching sequence trends into the stratosphere...",
                    "📈 Following sequence trends like a hawk...",
                    "🪐 Mapping sequence orbits across SELEX rounds...",
                    "🔭 Observing sequence trends through a telescope..."
                ]
                with st.spinner(random.choice(enrichment_messages)):
                    top_seqs = {}
                    for r in sorted_rounds:
                        top_seqs[r] = dict(sorted(round_data[r].items(), key=lambda x: x[1], reverse=True)[:5])
                    all_top_seqs = set()
                    for r in top_seqs.values():
                        all_top_seqs.update(r.keys())
                    # Sort sequences by frequency in the final round (highest to lowest)
                    final_round = sorted_rounds[-1]
                    seq_freqs = [(seq, round_data[final_round].get(seq, 0)) for seq in all_top_seqs]
                    sorted_seqs = [seq for seq, freq in sorted(seq_freqs, key=lambda x: x[1], reverse=True)]
                    # Plot in sorted order with enhanced legend clarity
                    trends = {seq: [] for seq in sorted_seqs}
                    for r in sorted_rounds:
                        for seq in sorted_seqs:
                            trends[seq].append(round_data[r].get(seq, 0))
                    fig, ax = plt.subplots(figsize=(12, 6))
                    # Use a distinct color palette and vary line widths
                    colors = sns.color_palette("tab10", len(sorted_seqs))
                    for idx, seq in enumerate(sorted_seqs):
                        counts = trends[seq]
                        # Vary line width: highest frequency gets thicker line
                        linewidth = 3 if idx == 0 else 1.5
                        # Show full trimmed sequence in legend (~32 nt)
                        ax.plot(sorted_rounds, counts, label=f"{seq} (Freq: {round_data[final_round].get(seq, 0)})", 
                                marker='o', color=colors[idx], linewidth=linewidth)
                    ax.set_xlabel("Round")
                    ax.set_ylabel("Frequency")
                    # Enhance legend: larger font, adjusted position
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, prop={'family': 'monospace'})
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                    enrichment_trends_path = os.path.join(st.session_state["output_dir"], "frequency_vs_rounds.png")
                    fig.savefig(enrichment_trends_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    st.image(enrichment_trends_path)

                motif_counts = {k: defaultdict(Counter) for k in st.session_state["params"]["motif_lengths"]}
                kmer_messages = [
                    "🔍 Digging into k-mers like a treasure hunter...",
                    "⛏️ Mining k-mers like a prospector in the wild...",
                    "🪙 Searching for k-mer gold in a sequence mine...",
                    "🏴‍☠️ Hunting for k-mer treasures on a pirate ship...",
                    "🕵️‍♂️ Investigating k-mers like a sequence detective..."
                ]
                with st.spinner(random.choice(kmer_messages)):
                    for round_label in round_data:
                        for seq, count in round_data[round_label].items():
                            for k in st.session_state["params"]["motif_lengths"]:
                                kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
                                for kmer in kmers:
                                    if len(kmer) == k and set(kmer).issubset("ACGTN"):
                                        motif_counts[k][round_label][kmer] += count

                st.subheader("Frequency Distribution (Final Round)")
                freq_dist_messages = [
                    "📉 Sketching frequency distributions with a teal twist...",
                    "🎨 Drawing frequency histograms with a teal brush...",
                    "🖼️ Framing frequency distributions in teal glory...",
                    "🌊 Creating a teal wave of frequency distributions...",
                    "🧑‍🎨 Painting frequency plots with a teal palette..."
                ]
                with st.spinner(random.choice(freq_dist_messages)):
                    final_round = sorted_rounds[-1]
                    freqs = list(round_data[final_round].values())
                    plt.figure(figsize=(8, 6))
                    sns.histplot(freqs, bins=50, log_scale=(False, True), color="teal", edgecolor="black")
                    plt.xlabel("Frequency", fontsize=12, fontweight="bold")
                    plt.ylabel("Number of Sequences (log scale)", fontsize=12, fontweight="bold")
                    plt.title(f"Frequency Distribution ({final_round})", fontsize=14, fontweight="bold")
                    plt.grid(True, linestyle="--", alpha=0.7)
                    freq_dist_path = os.path.join(st.session_state["output_dir"], f"plot_{st.session_state['plot_counter']}.png")
                    plt.savefig(freq_dist_path, dpi=300, bbox_inches="tight")
                    st.session_state["plot_counter"] += 1
                    plt.close()
                    st.image(freq_dist_path)

                st.subheader("Abundance Trajectories of Top k-mers")
                trajectory_messages = [
                    "🚀 Launching k-mer trajectories into orbit...",
                    "🌠 Sending k-mer trajectories on a cosmic journey...",
                    "✈️ Flying k-mer trajectories across the SELEX skies...",
                    "🪂 Dropping k-mer trajectories like a paratrooper...",
                    "🛸 Beaming k-mer trajectories into space..."
                ]
                with st.spinner(random.choice(trajectory_messages)):
                    for k in st.session_state["params"]["motif_lengths"]:
                        top_kmers = sorted(motif_counts[k][final_round].items(), key=lambda x: x[1], reverse=True)[:5]
                        top_kmer_names = [kmer for kmer, _ in top_kmers]
                        plt.figure(figsize=(12, 6))
                        colors = sns.color_palette("husl", len(top_kmer_names))
                        for idx, kmer in enumerate(top_kmer_names):
                            trajectory = [motif_counts[k][r].get(kmer, 0) for r in sorted_rounds]
                            label = kmer  # Show full k-mer in legend
                            plt.plot(sorted_rounds, trajectory, marker="o", color=colors[idx], label=label, linewidth=2)
                        plt.xlabel("Round", fontsize=12, fontweight="bold")
                        plt.ylabel("Frequency", fontsize=12, fontweight="bold")
                        plt.title(f"Abundance Trajectories of Top 5 {k}-mers", fontsize=14, fontweight="bold")
                        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", prop={'family': 'monospace'})
                        plt.xticks(rotation=45, ha="right")
                        plt.grid(True, linestyle="--", alpha=0.7)
                        plt.tight_layout()
                        trajectory_path = os.path.join(st.session_state["output_dir"], f"plot_{st.session_state['plot_counter']}.png")
                        plt.savefig(trajectory_path, dpi=300, bbox_inches="tight")
                        st.session_state["plot_counter"] += 1
                        plt.close()
                        st.image(trajectory_path)

                st.subheader("Rank Abundance Curve (Final Round)")
                rank_abundance_messages = [
                    "📈 Plotting rank abundance with a purple flair...",
                    "🟣 Crafting rank abundance curves in purple hues...",
                    "🎨 Drawing rank abundance with a purple touch...",
                    "💜 Adding a purple glow to rank abundance curves...",
                    "🖌️ Painting rank abundance in shades of purple..."
                ]
                with st.spinner(random.choice(rank_abundance_messages)):
                    final_round = sorted_rounds[-1]
                    freqs = list(round_data[final_round].values())
                    ranks = range(1, len(freqs) + 1)
                    abundances = sorted(freqs, reverse=True)
                    top_5_seqs = sorted(round_data[final_round].items(), key=lambda x: x[1], reverse=True)[:5]
                    top_5_freqs = [freq for seq, freq in top_5_seqs]
                    top_5_ranks = [abundances.index(freq) + 1 for freq in top_5_freqs]
                    # Sort annotations by frequency to ensure consistent ordering
                    sorted_indices = sorted(range(len(top_5_freqs)), key=lambda i: top_5_freqs[i], reverse=True)
                    plt.figure(figsize=(10, 6))
                    plt.plot(ranks, abundances, color="purple", linewidth=2)
                    plt.scatter(top_5_ranks, top_5_freqs, color="red", s=100, zorder=5)
                    # Adjust label positioning to prevent overlap
                    for idx in sorted_indices:
                        rank = top_5_ranks[idx]
                        freq = top_5_freqs[idx]
                        seq = top_5_seqs[idx][0]
                        # Dynamic y_offset based on frequency differences
                        base_offset = 50
                        freq_diff = (max(top_5_freqs) - freq) / max(top_5_freqs) if max(top_5_freqs) > 0 else 0
                        y_offset = base_offset * (idx % 2) - base_offset / 2 + freq_diff * 20
                        plt.annotate(
                            f"{seq} ({freq})",
                            (rank, freq),
                            textcoords="offset points",
                            xytext=(10, y_offset),
                            ha='left',
                            fontsize=6,
                            fontfamily='monospace',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                        )
                    plt.yscale("log")
                    plt.xlabel("Rank", fontsize=12, fontweight="bold")
                    plt.ylabel("Frequency (log scale)", fontsize=12, fontweight="bold")
                    plt.title(f"Rank Abundance Curve ({final_round})", fontsize=14, fontweight="bold")
                    plt.grid(True, linestyle="--", alpha=0.7)
                    plt.tight_layout()
                    rank_abundance_path = os.path.join(st.session_state["output_dir"], f"plot_{st.session_state['plot_counter']}.png")
                    plt.savefig(rank_abundance_path, dpi=300, bbox_inches="tight")
                    st.session_state["plot_counter"] += 1
                    plt.close()
                    st.image(rank_abundance_path)

                elapsed_time = time.time() - start_time
                st.session_state["run_log"].append(f"Visualization completed in {elapsed_time:.2f} seconds")
                st.session_state["processing_complete"] = True
                st.session_state["motif_counts"] = motif_counts
            else:
                st.error("No sequences processed. Check FASTQ files or parameters.")
#===== End SELEX metrics Tab ======

#===== Motif Analysis Tab =========
with tabs[2]:
    st.header("🔍 Motif Heatmap & Mapping")
    st.markdown(f"**Running**: `{SCRIPT_PATH}`")
    
    if "files" in st.session_state and st.session_state["files"]:
        cluster_colors_by_k = {k: {} for k in st.session_state["params"]["motif_lengths"]}
        filter_near_boundaries = st.checkbox("Filter motifs near trimmed boundaries (within 2 bases of start/end)", value=False)
        st.session_state["params"]["filter_near_boundaries"] = filter_near_boundaries
        conservation_cutoff = st.slider("Conservation cutoff for phylogenetic tree (%)", min_value=1, max_value=10, value=1, step=1)
        seqs_by_round = defaultdict(list)
        motif_positions = defaultdict(list)
        motif_counts = st.session_state.get("motif_counts", {k: defaultdict(Counter) for k in st.session_state["params"]["motif_lengths"]})
        motif_analysis_messages = [
            "🔥 Heating up the motif analysis engine...",
            "⚙️ Revving up the motif machine with a roar...",
            "🧬 Spinning the motif analysis wheel of fortune...",
            "🔧 Tuning the motif engine for peak performance...",
            "🚀 Igniting the motif analysis rocket boosters..."
        ]
        with st.spinner(random.choice(motif_analysis_messages)):
            start_time = time.time()
            time_estimation_messages = [
                "⏳ Estimating time by peeking at a few sequences...",
                "⏰ Checking the clock with a few test sequences...",
                "⌛ Guessing time with a sequence sneak peek...",
                "🕒 Timing a few sequences like a stopwatch pro...",
                "⏱️ Measuring time with a sequence test run..."
            ]
            with st.spinner(random.choice(time_estimation_messages)):
                test_seqs = []
                for f in st.session_state["files"]:
                    round_name = st.session_state["round_map"][f.name]
                    text = f.getvalue().decode("utf-8")
                    parser = SeqIO.parse(io.StringIO(text), "fastq")
                    chunk = list(islice(parser, 100))
                    for r in chunk:
                        full = str(r.seq).upper()
                        primer_fwd = st.session_state["params"]["primer_fwd"]
                        primer_rev = st.session_state["params"]["primer_rev"]
                        if len(full) < primer_fwd + primer_rev + 5:
                            continue
                        trimmed = full[primer_fwd:-primer_rev if primer_rev > 0 else None]
                        if (len(trimmed) >= 5 and set(trimmed).issubset("ACGTN") and
                            not any(trimmed.count(n) / len(trimmed) > st.session_state["params"]["artifact_threshold"] for n in "ACGT") and
                            not has_repetitive_pattern(trimmed)):
                            seqs_by_round[round_name].append((full, trimmed))
                    break
                test_seqs = list(seqs_by_round.values())[0][:100] if seqs_by_round else []
                test_start = time.time()
                for full_seq, trimmed_seq in test_seqs:
                    try:
                        RNA.cvar.temperature = st.session_state["params"]["temperature"]
                        RNA.cvar.dangles = st.session_state["params"]["dangling_ends"]
                        RNA.cvar.noGU = 0
                        structure, mfe = RNA.fold(preprocess_sequence(full_seq, st.session_state["params"]["seq_type"]))
                        for k in st.session_state["params"]["motif_lengths"]:
                            kmers = [(trimmed_seq[i:i+k], i) for i in range(len(trimmed_seq) - k + 1)]
                            for kmer, pos in kmers:
                                if len(kmer) == k and set(kmer).issubset("ACGTN"):
                                    full_pos = pos + st.session_state["params"]["primer_fwd"] + 1
                                    if full_pos + k - 1 <= len(structure):
                                        motif_structure = structure[full_pos-1:full_pos+k-1]
                                        accessible_bases = sum(1 for c in motif_structure if c == ".")
                                        accessibility = accessible_bases / k if k > 0 else 0.5
                    except Exception as e:
                        st.session_state["run_log"].append(f"Error folding sequence in estimation: {str(e)}")
                        continue
                test_time = time.time() - test_start
                total_seqs = sum(len(seqs) for seqs in seqs_by_round.values())
                est_time = (test_time / 100) * total_seqs if test_time > 0 else 0
                st.write(f"Estimated motif analysis time: ~{int(est_time)} seconds")

            seqs_by_round.clear()
            collect_messages = [
                "📚 Collecting sequences like a librarian organizing books...",
                "📦 Gathering sequences like a squirrel before winter...",
                "🧺 Picking sequences like apples in an orchard...",
                "📜 Collecting sequences like ancient scrolls...",
                "🗂️ Filing sequences like an organized archivist..."
            ]
            with st.spinner(random.choice(collect_messages)):
                for f in st.session_state["files"]:
                    round_name = st.session_state["round_map"][f.name]
                    try:
                        text = f.getvalue().decode("utf-8")
                        parser = SeqIO.parse(io.StringIO(text), "fastq")
                        chunk_size = 10000
                        count = 0
                        skipped_count = 0
                        while True:
                            chunk = list(islice(parser, chunk_size))
                            if not chunk:
                                break
                            for r in chunk:
                                full = str(r.seq).upper()
                                primer_fwd = st.session_state["params"]["primer_fwd"]
                                primer_rev = st.session_state["params"]["primer_rev"]
                                if len(full) < primer_fwd + primer_rev + 5:
                                    skipped_count += 1
                                    continue
                                trimmed = full[primer_fwd:-primer_rev if primer_rev > 0 else None]
                                if (len(trimmed) >= 5 and set(trimmed).issubset("ACGTN") and
                                    not any(trimmed.count(n) / len(trimmed) > st.session_state["params"]["artifact_threshold"] for n in "ACGT") and
                                    not has_repetitive_pattern(trimmed)):
                                    seqs_by_round[round_name].append((full, trimmed))
                                    count += 1
                                else:
                                    skipped_count += 1
                                if count >= st.session_state["params"]["top_n"]:
                                    break
                            if count >= st.session_state["params"]["top_n"]:
                                break
                        st.session_state["run_log"].append(f"{f.name}: {count} valid sequences for motifs, {skipped_count} skipped")
                    except Exception as e:
                        st.session_state["run_log"].append(f"Error processing {f.name}: {str(e)}")
                        seqs_by_round.clear()
                        break

            length_check_messages = [
                "⚠️ Checking sequence lengths with a magnifying glass...",
                "📏 Measuring sequences like a tailor with a tape...",
                "🔍 Inspecting sequence lengths with eagle eyes...",
                "📐 Gauging sequence lengths like an architect...",
                "🧐 Examining sequence lengths with a scientist's gaze..."
            ]
            with st.spinner(random.choice(length_check_messages)):
                length_warnings_summary = defaultdict(Counter)
                expected_length = 32
                for round_label, seq_pairs in seqs_by_round.items():
                    for full_seq, trimmed_seq in seq_pairs:
                        if abs(len(trimmed_seq) - expected_length) > 0:
                            length_warnings_summary[round_label][len(trimmed_seq)] += 1
                for round_label, counts in length_warnings_summary.items():
                    length_summary = ", ".join(f"{count} at {length} bp" for length, count in sorted(counts.items()))
                    st.warning(f"{round_label}: {length_summary} (expected ~{expected_length} bp). This may be due to SELEX-induced evolution (e.g., PCR artifacts or indels).")

            folding_messages = [
                "🧬 Folding sequences like origami to find motifs...",
                "✂️ Folding sequences like a paper crane artist...",
                "📜 Folding sequences like ancient parchment scrolls...",
                "🪡 Folding sequences like a seamstress with fabric...",
                "🎀 Folding sequences into motif masterpieces..."
            ]
            with st.spinner(random.choice(folding_messages)):
                for round_label, seq_pairs in seqs_by_round.items():
                    for full_seq, trimmed_seq in seq_pairs:
                        try:
                            RNA.cvar.temperature = st.session_state["params"]["temperature"]
                            RNA.cvar.dangles = st.session_state["params"]["dangling_ends"]
                            RNA.cvar.noGU = 0
                            structure, mfe = RNA.fold(preprocess_sequence(full_seq, st.session_state["params"]["seq_type"]))
                            for k in st.session_state["params"]["motif_lengths"]:
                                kmers = [(trimmed_seq[i:i+k], i) for i in range(len(trimmed_seq) - k + 1)]
                                for kmer, pos in kmers:
                                    if len(kmer) == k and set(kmer).issubset("ACGTN"):
                                        full_pos = pos + st.session_state["params"]["primer_fwd"] + 1
                                        if full_pos + k - 1 <= len(structure):
                                            if filter_near_boundaries and (pos <= 2 or pos + k >= len(trimmed_seq) - 2):
                                                continue
                                            motif_structure = structure[full_pos-1:full_pos+k-1]
                                            accessible_bases = sum(1 for c in motif_structure if c == ".")
                                            accessibility = accessible_bases / k if k > 0 else 0.5
                                            motif_positions[round_label].append({
                                                "Motif": kmer,
                                                "Start": full_pos,
                                                "Structure": motif_structure,
                                                "Accessibility": accessibility
                                            })
                        except Exception as e:
                            st.session_state["run_log"].append(f"Error folding sequence in {round_label}: {str(e)}")
                            continue

            st.session_state["motif_data"] = motif_positions
            st.session_state["motif_counts"] = motif_counts
            elapsed_time = time.time() - start_time
            st.session_state["run_log"].append(f"Motif analysis completed in {elapsed_time:.2f} seconds")
            st.session_state["processing_complete"] = True

            if not any(motif_positions.values()) or not any(motif_counts[k] for k in motif_counts):
                st.error("No motifs found after processing. This may be due to sequence length mismatches, folding errors, or strict filtering. Try disabling the boundary filter or checking your FASTQ data.")
            else:
                def normalize_counts(df):
                    min_val = df.min().min()
                    max_val = df.max().max()
                    if max_val == min_val:
                        return df * 0
                    return (df - min_val) / (max_val - min_val)

                all_mapped_motifs = {k: {} for k in st.session_state["params"]["motif_lengths"]}
                for k in st.session_state["params"]["motif_lengths"]:
                    st.subheader(f"📊 {k}-mer Motif Heatmap")
                    heatmap_messages = [
                        f"🌈 Painting {k}-mer heatmaps with a warm glow...",
                        f"🔥 Heating up {k}-mer heatmaps with fiery colors...",
                        f"🎨 Coloring {k}-mer heatmaps like a sunset...",
                        f"🖌️ Brushing {k}-mer heatmaps with warm tones...",
                        f"🌅 Creating {k}-mer heatmaps with a golden hue..."
                    ]
                    with st.spinner(random.choice(heatmap_messages)):
                        df_raw = pd.DataFrame(motif_counts[k]).fillna(0)
                        sorted_cols = sorted(
                            df_raw.columns,
                            key=lambda x: int(x.split()[-1]) if x != "Unassigned" else 0
                        )
                        df_raw = df_raw[sorted_cols]
                        if df_raw.empty or df_raw.sum().sum() == 0:
                            st.warning(f"No {k}-mer motifs found for heatmap.")
                            df_top = pd.DataFrame()  # Empty DataFrame to allow continuation
                            df_norm = pd.DataFrame()
                        else:
                            df_top = df_raw.loc[df_raw.sum(axis=1).sort_values(ascending=False).head(25).index]
                            df_norm = normalize_counts(df_top)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Raw Frequency**")
                                try:
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    sns.heatmap(
                                        df_top,
                                        cmap="YlOrRd",
                                        annot=True,
                                        fmt=".2f",
                                        ax=ax,
                                        cbar_kws={"label": "Raw Frequency"}
                                    )
                                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontfamily="monospace")
                                    ax.set_yticklabels(ax.get_yticklabels(), fontfamily="monospace")
                                    raw_heatmap_path = os.path.join(st.session_state["output_dir"], f"{k}mer_raw_heatmap.png")
                                    fig.savefig(raw_heatmap_path, dpi=300, bbox_inches="tight")
                                    st.image(raw_heatmap_path, caption="Raw Frequency Heatmap")
                                except Exception as e:
                                    st.error(f"Error rendering raw frequency heatmap for {k}-mers: {e}")
                                finally:
                                    plt.close(fig)
                                raw_csv_path = os.path.join(st.session_state["output_dir"], f"{k}mer_raw.csv")
                                df_top.to_csv(raw_csv_path, index=True)
                                with open(raw_csv_path, "rb") as f:
                                    st.download_button("⬇ Download Raw CSV", f, f"{k}mer_raw.csv")

                            with col2:
                                st.markdown("**Normalized Frequency**")
                                try:
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    sns.heatmap(
                                        df_norm,
                                        cmap="YlOrRd",
                                        annot=True,
                                        fmt=".2f",
                                        ax=ax,
                                        cbar_kws={"label": "Normalized Frequency (Within Top 25)"}
                                    )
                                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontfamily="monospace")
                                    ax.set_yticklabels(ax.get_yticklabels(), fontfamily="monospace")
                                    norm_heatmap_path = os.path.join(st.session_state["output_dir"], f"{k}mer_norm_heatmap.png")
                                    fig.savefig(norm_heatmap_path, dpi=300, bbox_inches="tight")
                                    st.image(norm_heatmap_path, caption="Normalized Frequency Heatmap")
                                except Exception as e:
                                    st.error(f"Error rendering normalized frequency heatmap for {k}-mers: {e}")
                                finally:
                                    plt.close(fig)
                                norm_csv_path = os.path.join(st.session_state["output_dir"], f"{k}mer_norm.csv")
                                df_norm.to_csv(norm_csv_path, index=True)
                                with open(norm_csv_path, "rb") as f:
                                    st.download_button("⬇ Download Normalized CSV", f, f"{k}mer_norm.csv")

                    final_round = sorted_cols[-1]  # Already computed
                    st.subheader(f"{k}-mer Motif Mapping (Last Round: {final_round})")
                    mapping_messages = [
                        f"🗺️ Mapping {k}-mer motifs like an explorer...",
                        f"📍 Pinning {k}-mer motifs on a SELEX map...",
                        f"🧭 Navigating {k}-mer motifs like a cartographer...",
                        f"🌍 Charting {k}-mer motifs on a global scale...",
                        f"🚩 Marking {k}-mer motifs like a trailblazer..."
                    ]
                    with st.spinner(random.choice(mapping_messages)):
                        motif_table = []
                        motif_index = {motif: idx + 1 for idx, motif in enumerate(df_top.index[:10])}
                        all_mapped_motifs[k] = motif_index
                        round_sequences = []
                        for full_seq, _ in seqs_by_round[final_round]:
                            round_sequences.append(full_seq)
                        total_seqs = len(round_sequences)
                        for round_label in [final_round]:
                            motif_count = 0
                            for idx, entry in enumerate(motif_positions[round_label]):
                                if len(entry["Motif"]) == k and (not df_top.empty and entry["Motif"] in df_top.index):
                                    if motif_count >= 10:
                                        break
                                    accessible_bases = sum(1 for c in entry["Structure"] if c == ".")
                                    accessibility = accessible_bases / k if k > 0 else 0.5
                                    motif_number = motif_index.get(entry["Motif"])
                                    motif_label = f"{entry['Motif']} (Motif {motif_number})" if motif_number else entry["Motif"]
                                    representative_seq = None
                                    for full_seq in round_sequences:
                                        if len(full_seq) >= entry["Start"] + len(entry["Motif"]) - 1 and full_seq[entry["Start"]-1:entry["Start"]+len(entry["Motif"])-1] == entry["Motif"]:
                                            representative_seq = full_seq
                                            break
                                    if not representative_seq:
                                        full_seq_counts = Counter(round_sequences)
                                        representative_seq = max(full_seq_counts.items(), key=lambda x: x[1])[0]
                                    RNA.cvar.temperature = st.session_state["params"]["temperature"]
                                    RNA.cvar.dangles = st.session_state["params"]["dangling_ends"]
                                    RNA.cvar.noGU = 0
                                    structure, _ = RNA.fold(preprocess_sequence(representative_seq, st.session_state["params"]["seq_type"]))
                                    motif_positions_count = sum(1 for seq in round_sequences if entry["Motif"] in seq)
                                    fraction = motif_positions_count / total_seqs if total_seqs > 0 else 0
                                    motif_table.append({
                                        "Round": round_label,
                                        "Motif": motif_label,
                                        "Start": entry["Start"],
                                        "Structure": entry["Structure"],
                                        "Accessibility": round(accessibility, 3),
                                        "Full Sequence": representative_seq,
                                        "Dot-Bracket": structure,
                                        "Conservation": f"Found in {motif_positions_count}/{total_seqs} sequences ({fraction*100:.2f}%)"
                                    })
                                    st.session_state["run_log"].append(f"Motif {idx+1} ({round_label}, {k}-mer): Motif={entry['Motif']}, Structure={entry['Structure']}, Accessibility={accessibility:.3f}")
                                    motif_count += 1
                        if motif_table:
                            df_motif = pd.DataFrame(motif_table)
                            def color_motif(val, kmer_length):
                                if not val:
                                    return ""
                                motif = val.split(" (Motif")[0]
                                color = cluster_colors_by_k.get(kmer_length, {}).get(motif, (0, 0, 0))
                                hex_color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
                                return f"color: {hex_color}; font-family: Courier New;"
                            def highlight_motif(row):
                                if not row["Round"]:
                                    return ""
                                full_seq = row["Full Sequence"]
                                start = row["Start"]
                                motif = row["Motif"].split(" (Motif")[0]
                                start_idx = start - 1
                                end_idx = start_idx + len(motif)
                                if start_idx < 0 or end_idx > len(full_seq):
                                    return full_seq
                                highlighted_seq = (
                                    f"{full_seq[:start_idx]}"
                                    f"<b><span style='color:red'>{full_seq[start_idx:end_idx]}</span></b>"
                                    f"{full_seq[end_idx:]}"
                                )
                                return highlighted_seq
                            df_motif["Full Sequence"] = df_motif.apply(highlight_motif, axis=1)
                            styled_df = (df_motif.style
                                         .applymap(lambda x: color_motif(x, k), subset=["Motif"])
                                         .set_properties(**{"font-family": "Courier New"})
                                         .set_table_styles([{"selector": "td", "props": [("white-space", "pre-wrap")]}])
                                         .hide())
                            st.markdown(styled_df.to_html(escape=False), unsafe_allow_html=True)
                            mapping_csv_path = os.path.join(st.session_state["output_dir"], f"{k}mer_mapping.csv")
                            df_motif.to_csv(mapping_csv_path, index=True)
                            with open(mapping_csv_path, "rb") as f:
                                st.download_button(f"⬇ Download {k}-mer Mapping CSV", f, f"{k}mer_mapping.csv")
                            # Add FASTA download button
                            fasta_content = ""
                            for idx, row in df_motif.iterrows():
                                motif_number = row["Motif"].split(" (Motif ")[1].split(")")[0] if "Motif" in row["Motif"] else idx + 1
                                fasta_content += f">Motif_{motif_number}_{row['Round'].replace(' ', '_')}\n{row['Full Sequence']}\n"
                            mapping_fasta_path = os.path.join(st.session_state["output_dir"], f"{k}mer_mapping.fasta")
                            with open(mapping_fasta_path, "w") as f:
                                f.write(fasta_content)
                            with open(mapping_fasta_path, "rb") as f:
                                st.download_button(f"⬇ Download {k}-mer Mapping FASTA", f, f"{k}mer_mapping.fasta")
                        else:
                            st.warning(f"No {k}-mer motifs found for mapping.")

                    st.subheader(f"{k}-mer Phylogenetic Tree")
                    tree_messages = [
                        f"🌳 Growing a {k}-mer phylogenetic tree with care...",
                        f"🌲 Branching out {k}-mer motifs like a forest...",
                        f"🌴 Sprouting {k}-mer trees in a SELEX jungle...",
                        f"🌱 Planting {k}-mer motifs in a phylogenetic garden...",
                        f"🌿 Cultivating {k}-mer trees with evolutionary flair..."
                    ]
                    with st.spinner(random.choice(tree_messages)):
                        top_kmers = df_top.index[:10].tolist()
                        if len(top_kmers) < 2:
                            st.warning(f"Not enough {k}-mer motifs for clustering.")
                            continue
                        distances = [[distance(k1, k2) for k2 in top_kmers] for k1 in top_kmers]
                        dist_array = ssd.squareform(distances)
                        linkage_matrix = linkage(dist_array, method="average")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        dendro = dendrogram(
                            linkage_matrix,
                            labels=[f"{kmer} (Motif {all_mapped_motifs[k].get(kmer, 'N/A')})" for kmer in top_kmers],
                            leaf_rotation=45,
                            leaf_font_size=10,
                            ax=ax
                        )
                        cluster_colors = sns.color_palette("tab10", n_colors=len(top_kmers))
                        for idx, kmer in enumerate(top_kmers):
                            cluster_colors_by_k[k][kmer] = cluster_colors[idx]
                        ax.set_title(f"Phylogenetic Tree of Top {k}-mers (Levenshtein Distance)")
                        ax.set_ylabel("Distance")
                        ax.set_xticklabels(ax.get_xticklabels(), fontfamily="monospace")
                        plt.tight_layout()
                        tree_path = os.path.join(st.session_state["output_dir"], f"{k}mer_phylogenetic_tree.png")
                        fig.savefig(tree_path, dpi=300, bbox_inches="tight")
                        plt.close(fig)
                        st.image(tree_path, caption=f"{k}-mer Phylogenetic Tree")
#===== End Motif Analysis Tab ======

#===== Final Candidates Tab =========
with tabs[3]:
    st.header("📄 Final Candidate Scoring")
    st.markdown(f"**Running**: `{SCRIPT_PATH}`")
    
    if "files" in st.session_state and st.session_state["files"]:
        # Display elapsed time
        if st.session_state["analysis_start_time"] is not None:
            elapsed_time = time.time() - st.session_state["analysis_start_time"]
            st.markdown(f"**Elapsed Time Since Analysis Start**: {elapsed_time:.2f} seconds")
        else:
            st.warning("Analysis start time not recorded. Run analysis from Overview tab to track elapsed time.")

        quikfold_settings = st.session_state.get("quikfold_params", {
            "foldtype": "DNA4.0",
            "temperature": 25.0,
            "na_conc": 137.0,
            "mg_conc": 1.0,
            "polymer": "0",
            "max_structures": "1"
        })
        st.info(
            f"DNA/RNA folding performed using ViennaRNA with settings: "
            f"Energy rules={quikfold_settings['foldtype']}, "
            f"Temperature={quikfold_settings['temperature']}°C, "
            f"[Na+]={quikfold_settings['na_conc']} mM, "
            f"[Mg2+]={quikfold_settings['mg_conc']} mM, "
            f"Dangling ends={st.session_state['params']['dangling_ends']}. "
            f"2D structures plotted using ViennaRNA."
        )
        st.markdown("#### Accessibility Weight")
        
        if not st.session_state["calculations_complete"]:
            accessibility_weight = st.slider(
                "Frequency & Stability vs. Accessibility Priority",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state["accessibility_weight"],
                step=0.01,
                format="%.2f",
                help="0.0: Rank by frequency and stability (Score_1). 1.0: Prioritize unpaired motifs (Score_2 boosts up to 50%)."
            )
            st.session_state["accessibility_weight"] = accessibility_weight
        else:
            st.markdown(f"**Frequency & Stability vs. Accessibility Priority (Fixed):** {st.session_state['accessibility_weight']:.2f}")
            accessibility_weight = st.session_state["accessibility_weight"]

        st.markdown(f"**Effect**: At {accessibility_weight:.2f}, low accessibility (0.111) changes Score_2 by {(accessibility_weight * (0.111 - 0.5) * 100):.1f}%; high (1.0) by {(accessibility_weight * (1.0 - 0.5) * 100):.1f}%.")

        scoring_messages = [
            "⚙️ Scoring candidates with precision and care...",
            "🏆 Awarding scores to candidates like a judge...",
            "📊 Evaluating candidates with a sharp eye...",
            "🧑‍⚖️ Judging candidates with a fair scale...",
            "🔢 Scoring candidates like a math whiz..."
        ]
        with st.spinner(random.choice(scoring_messages)):
            start_time = time.time()
            if not round_data:
                st.error("No sequences processed. Check FASTQ files, primer lengths, or file format.")
                st.session_state["run_log"].append("No sequences in round_data. Check FASTQ files and parameters.")
            else:
                try:
                    final_round = max(
                        round_data.keys(),
                        key=lambda x: int(x.split()[-1]) if x != "Unassigned" else 0
                    )
                    st.session_state["run_log"].append(f"Selected final round: {final_round}")
                except ValueError as e:
                    st.error(f"Error selecting final round: {str(e)}. Ensure FASTQ files have assigned round numbers (e.g., 'Round 1').")
                    st.session_state["run_log"].append(f"Round selection error: {str(e)}")
                    final_round = None

                if final_round:
                    freq_dict = round_data[final_round]
                    st.session_state["run_log"].append(f"Sequences in {final_round}: {len(freq_dict)}")
                    filtered_seqs = []
                    for seq, count in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:st.session_state["params"]["top_n"]]:
                        if (not any(seq.count(n) / len(seq) > st.session_state["params"]["artifact_threshold"] for n in "ACGT") and
                            not has_repetitive_pattern(seq) and
                            'N' not in seq):
                            filtered_seqs.append((seq, count))
                    st.session_state["run_log"].append(f"Filtered sequences for {final_round}: {len(filtered_seqs)}")
                    top_final_seqs = []
                    # Reconstruct full sequences by re-reading from the FASTQ files
                    full_seq_map = {}
                    for f in st.session_state["files"]:
                        if st.session_state["round_map"][f.name] != final_round:
                            continue
                        try:
                            text = f.getvalue().decode("utf-8")
                            parser = SeqIO.parse(io.StringIO(text), "fastq")
                            for r in parser:
                                full_seq = str(r.seq).upper()
                                primer_fwd = st.session_state["params"]["primer_fwd"]
                                primer_rev = st.session_state["params"]["primer_rev"]
                                if len(full_seq) < primer_fwd + primer_rev + 5:
                                    continue
                                trimmed = full_seq[primer_fwd:-primer_rev if primer_rev > 0 else None]
                                if trimmed in freq_dict:
                                    full_seq_map[trimmed] = full_seq
                        except Exception as e:
                            st.session_state["run_log"].append(f"Error re-reading {f.name}: {str(e)}")
                            continue

                    for seq, count in filtered_seqs:
                        try:
                            # Use the full sequence from the FASTQ file, including primers
                            full_seq = full_seq_map.get(seq, seq)  # Fallback to trimmed seq if not found
                            RNA.cvar.temperature = st.session_state["params"]["temperature"]
                            RNA.cvar.dangles = st.session_state["params"]["dangling_ends"]
                            RNA.cvar.noGU = 0
                            structure, mfe = RNA.fold(preprocess_sequence(full_seq, st.session_state["params"]["seq_type"]))
                            max_accessibility = 0.0
                            best_motif = None
                            best_structure = ""
                            for k in st.session_state["params"]["motif_lengths"]:
                                kmers = [(full_seq[i:i+k], i) for i in range(len(full_seq) - k + 1)]
                                for kmer, pos in kmers:
                                    if len(kmer) == k and set(kmer).issubset("ACGTN"):
                                        full_pos = pos + 1
                                        if full_pos + k - 1 <= len(structure):
                                            motif_structure = structure[full_pos-1:full_pos+k-1]
                                            accessible_bases = sum(1 for c in motif_structure if c == ".")
                                            accessibility = accessible_bases / k if k > 0 else 0.5
                                            if accessibility > max_accessibility:
                                                max_accessibility = accessibility
                                                best_motif = kmer
                                                best_structure = motif_structure

                            total = sum(freq_dict.values())
                            frequency = count / total if total > 0 else 0
                            score_1 = frequency * abs(mfe)
                            score_2 = score_1 * (1 + accessibility_weight * (max_accessibility - 0.5))
                            top_final_seqs.append({
                                "Sequence": full_seq,
                                "Trimmed": seq,
                                "Frequency": round(frequency * 100, 3),
                                "MFE (kcal/mol)": round(mfe, 2),
                                "Score_1": round(score_1, 3),
                                "Score_2": round(score_2, 3),
                                "Best Motif": best_motif,
                                "Motif Structure": best_structure,
                                "Motif Accessibility": round(max_accessibility, 3),
                                "Dot-Bracket": structure
                            })
                        except Exception as e:
                            st.session_state["run_log"].append(f"Error folding sequence {seq}: {str(e)}")
                            continue

                    if top_final_seqs:
                        df_final = pd.DataFrame(top_final_seqs)
                        df_final.sort_values(by="Score_2", ascending=False, inplace=True)
                        df_final.reset_index(drop=True, inplace=True)
                        df_final.index += 1

                        st.markdown("### Top Candidate Aptamers")
                        st.markdown(
                            f"**Scoring Formula**: Score_1 = Frequency × |ΔG|, Score_2 = Score_1 × (1 + {accessibility_weight:.2f} × (Motif Accessibility - 0.5))"
                        )
                        st.dataframe(
                            df_final.style.set_properties(**{"font-family": "Courier New"}).set_table_styles(
                                [{"selector": "td", "props": [("white-space", "pre-wrap")]}]
                            ),
                            height=300,
                            use_container_width=True
                        )

                        final_csv_path = os.path.join(st.session_state["output_dir"], "final_candidates.csv")
                        df_final.to_csv(final_csv_path, index=True)
                        with open(final_csv_path, "rb") as f:
                            st.download_button("⬇ Download Final Candidates CSV", f, "final_candidates.csv")

                        # Add FASTA download button for final candidates
                        fasta_content = ""
                        for idx, row in df_final.iterrows():
                            fasta_content += f">Candidate_{idx}_Score2_{row['Score_2']:.3f}\n{row['Sequence']}\n"
                        final_fasta_path = os.path.join(st.session_state["output_dir"], "final_candidates.fasta")
                        with open(final_fasta_path, "w") as f:
                            f.write(fasta_content)
                        with open(final_fasta_path, "rb") as f:
                            st.download_button("⬇ Download Final Candidates FASTA", f, "final_candidates.fasta")

                        st.markdown("### Candidate Details")
                        for idx, row in df_final.iterrows():
                            if idx > 5:  # Limit to top 5 candidates
                                break
                            st.markdown(f"#### Candidate {idx}")
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"**Sequence**: `{row['Sequence']}`")
                                st.markdown(f"**Trimmed**: `{row['Trimmed']}`")
                                st.markdown(f"**Frequency**: {row['Frequency']}%")
                                st.markdown(f"**MFE**: {row['MFE (kcal/mol)']} kcal/mol")
                                st.markdown(f"**Score_1**: {row['Score_1']}")
                                st.markdown(f"**Score_2**: {row['Score_2']}")
                                st.markdown(f"**Best Motif**: `{row['Best Motif']}`")
                                st.markdown(f"**Motif Structure**: `{row['Motif Structure']}`")
                                st.markdown(f"**Motif Accessibility**: {row['Motif Accessibility']}")
                                st.markdown(f"**Dot-Bracket**: `{row['Dot-Bracket']}`")
                            with col2:
                                fasta_entry = f">Candidate_{idx}_Score2_{row['Score_2']:.3f}\n{row['Sequence']}"
                                structure_path = os.path.join(st.session_state["output_dir"], f"structure_candidate_{idx}.png")
                                result = plot_dna_structure(fasta_entry, structure_path, f"Candidate_{idx}")
                                if result["success"]:
                                    display_rnacomposer_input(row['Sequence'], result["dot_bracket"], f"Candidate_{idx}")

                        st.session_state["calculations_complete"] = True
                        elapsed_time = time.time() - start_time
                        st.session_state["run_log"].append(f"Final candidate scoring completed in {elapsed_time:.2f} seconds")
                    else:
                        st.warning("No candidates found after filtering. Try adjusting parameters (e.g., increase Top N, reduce artifact threshold).")
#===== End Final Candidates Tab ======