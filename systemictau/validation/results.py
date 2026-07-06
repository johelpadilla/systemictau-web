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
        Legacy summary string (Deprecado en v5.0).
        Se mantiene por compatibilidad, pero la UI usa componentes estructurados en app.py.
        """
        return "Surrogate validation completed. See structured results in UI."
