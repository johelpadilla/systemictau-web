# Systemic Tau v5.6.0 (The Synthesis Release) 🧬

Welcome to the **Systemic Tau Paradigm** analytical engine. Version 5.6.0 represents a monumental leap in time-series topology, transitioning from static correlations to dynamic, hyper-dimensional causal mapping. This engine is specifically designed to detect early warning signals (EWS) of systemic collapse in complex non-linear environments.

## 🚀 Quick Start (Mac / Linux)

1. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the App:**
   Simply double-click the `run.command` file, OR run:
   ```bash
   streamlit run app.py
   ```

## 🌟 What's New in v5.6.0?

This milestone release establishes the definitive mathematical foundation for the engine:

- **1. Topological Data Analysis (TDA):** Complete integration of Algebraic Topology. Compute Betti Numbers ($H_0$, $H_1$) and topological invariants (Total/Max Persistence, Persistence Entropy) to mathematically map the creation and collapse of "holes" in the system's structural manifold.
- **2. Ordinal Memory & Symbolic Transfer Entropy:** Measure directional information flow and true non-linear coupling without distributional assumptions. Powered by a high-speed Numba core, it offers both Rank Mutual Information (Lite) and strict Symbolic TE (Full) modes.
- **3. Adaptive Breathing Window ($W_t$):** The temporal sliding window now auto-scales in real-time. It "inhales" (expands) during stable regimes to suppress noise, and "exhales" (contracts) during volatile transitions to pinpoint exact rupture moments.
- **4. Linear Baseline Testing:** The IAAFT surrogate test has been redesigned to quantify the exact percentage of a critical transition ($t^*$) driven by linear vs non-linear topological factors.
- **5. Enterprise Aesthetic & PDF Engine:** Features a cohesive Light Mode standard (`config.toml`), responsive Glassmorphic KPI cards, a streamlined control sidebar, and a vastly improved PDF generation engine that auto-formats professional academic cover pages.

## 🏗 Architecture

- **`app.py`**: The main Streamlit GUI frontend.
- **`systemictau/`**: The core mathematical backend (Numba-optimized). Includes `topology.py`, `ordinal_memory.py`, and `analysis.py`.
- **`data/`**: Directory for sample datasets (e.g., DengAI).
- **`utils/`**: Helper utilities, including `export_pdf.py`.

---
*Powered by the Magna Synthesis v6 and the Principle of Ontological Ascent.*
