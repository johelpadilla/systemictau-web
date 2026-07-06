import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from systemictau.results import OntologicalAscentResult
from systemictau.report import generate_academic_report
import numpy as np

# Create a mock result
res = OntologicalAscentResult(
    X=np.random.rand(100, 3),
    taus_global=np.random.rand(100),
    window_size=10,
    T_series=np.linspace(0, 10, 100),
    episodes=[],
    t_frob=50,
    t_ks=50,
    t_star=50,
    frob_max=0.5,
    ks_max=0.5,
    fractal_D=1.5,
    metadata={"num_components": 3},
    hp_z=1.0,
    lam=0.5,
    tt=10.0,
    M_max=10.0,
    M_mean=5.0,
    taus_per_module=np.random.rand(100, 3)
)

print("Generating report markdown...")
try:
    report_md = generate_academic_report(res, lang="es", include_significance_appendix=False)
except Exception as e:
    print(f"Error generating md: {e}")
    sys.exit(1)

print(f"Generated MD size: {len(report_md)}")

print("Generating PDF...")
from utils.export_pdf import convert_markdown_to_pdf
try:
    pdf = convert_markdown_to_pdf(report_md)
    print(f"PDF generated! Size: {len(pdf)}")
except Exception as e:
    print(f"Error generating PDF: {e}")
