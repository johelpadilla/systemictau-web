from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from ..results import OntologicalAscentResult

@dataclass(frozen=True)
class SurrogateValidationResult:
    """
    Immutable encapsulation of the IAAFT surrogate validation results.
    """
    real_result: OntologicalAscentResult
    surrogate_t_stars: List[Optional[float]]
    percentile_rank: float
    is_significant: bool
    n_surrogates: int
    mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """
        Generates a human-readable interpretation of the surrogate testing.
        """
        base_t_star = self.real_result.t_star
        if base_t_star is None:
            return "No critical transition (t*) was found in the real data. Surrogate validation is not applicable."
            
        n_valid = sum(1 for t in self.surrogate_t_stars if t is not None)
        
        msg = [
            f"### IAAFT Surrogate Validation (N={self.n_surrogates})",
            f"**Real Transition (t*):** {base_t_star:.2f}",
            f"**Execution Mode:** {self.mode}",
            f"**Valid Surrogates with t*:** {n_valid} / {self.n_surrogates}",
            f"**Statistical Percentile:** {self.percentile_rank:.2f}%",
            f"**Significance (p < 0.05 equivalent):** {'PASSED' if self.is_significant else 'FAILED'}",
            "",
            "**Interpretation:**"
        ]
        
        if self.is_significant:
            msg.append(
                f"The empirical critical transition at t*={base_t_star:.2f} is mathematically significant "
                f"against linear surrogate models. It falls in the extreme {self.percentile_rank:.2f} percentile "
                "of the surrogate distribution, proving that the topological lock-in is a genuine product of "
                "non-linear spatial coupling, and cannot be replicated by random phase shifts of the underlying "
                "power spectra."
            )
        else:
            msg.append(
                f"The empirical critical transition failed to achieve statistical significance against "
                f"the surrogate models (Percentile: {self.percentile_rank:.2f}%). This indicates that the observed "
                "topological shift could potentially be a trivial artifact of the individual linear properties "
                "(autocorrelation and amplitude distribution) of the variables, rather than a genuine non-linear "
                "systemic transition."
            )
            
        return "\n".join(msg)
