import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


@dataclass(frozen=True)
class OntologicalAscentResult:
    """
    Contenedor inmutable de resultados del análisis completo de Tau Sistémico (v4.6.0).
    
    Encapsula el ascenso ontológico desde la Serie Temporal hasta el Bloqueo Crítico (Lock-in)
    y el eventual punto de transición (t*).
    """

    # === Datos de entrada ===
    X: np.ndarray                          # Matriz original (T, N)
    window_size: int                       # Tamaño de ventana usado

    # === Capa 1: Persistencia y Criticidad ===
    taus_global: np.ndarray
    T_series: np.ndarray                   # Tiempo sistémico acumulado (RECD)
    hp_z: float                            # Z-score de hiper-persistencia
    lam: float                             # Laminaridad (RQA)
    tt: float                              # Trapping Time (RQA)
    M_max: float                           # Masa crítica máxima
    M_mean: float                          # Masa crítica media

    # === Capa 2: Joint Episodes ===
    episodes: List[Dict[str, Any]]         # Lista de episodios conjuntos detectados

    # === Capa 3: Reorganización ===
    t_star: Optional[int] = None           # Punto de reorganización consensuado
    t_frob: Optional[int] = None
    t_ks: Optional[int] = None
    frob_max: Optional[float] = None
    ks_max: Optional[float] = None

    # === Dimensión Fractal ===
    fractal_D: Optional[float] = None

    # === Metadatos y Validación ===
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    figures: Optional[Dict[str, str]] = None
    
    # === Additions for report compatibility ===
    dtk_series: np.ndarray = field(default_factory=lambda: np.array([]))
    taus_per_module: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # === Topological Data Analysis (TDA) ===
    tda_results: Optional[Dict[str, Any]] = None
    
    # === Ordinal Memory (Symbolic Transfer Entropy) ===
    ordinal_results: Optional[Dict[str, Any]] = None
    
    # === Nested RECD (Extramental Clock) ===
    nested_recd_results: Optional[Dict[str, Any]] = None
    
    component_names: List[str] = field(default_factory=list)
    
    # === Teorema 24v ===
    dyadic_k_series: np.ndarray = field(default_factory=lambda: np.array([]))
    estimated_period_series: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self, level: str = "detailed", lang: str = "en") -> str:
        """
        Genera una narrativa del ascenso ontológico.
        """
        num_episodes = len(self.episodes)
        t_star_text = str(self.t_star) if self.t_star else "None"
        recd_max = float(np.max(self.T_series)) if len(self.T_series) > 0 else 0.0

        if lang == "es":
            if level == "short":
                return f"El sistema exhibió {num_episodes} episodios conjuntos, acumulando un RECD máximo de {recd_max:.2f}, antes de una transición crítica en t*={t_star_text}."
            else:
                return (
                    f"Ascenso Ontológico Detectado:\n"
                    f"- Capa 1 (Microscópica): El sistema alcanzó una dimensión fractal de {self.fractal_D:.2f}, con una hiper-persistencia Z={self.hp_z:.2f}.\n"
                    f"- Capa 2 (Mesoscópica): Se cristalizaron {num_episodes} episodios de entrelazamiento denso.\n"
                    f"- Capa 3 (Macroscópica): El tiempo sistémico (RECD) se aceleró a {recd_max:.2f}, marcando la transición crítica en t*={t_star_text}.\n\n"
                    f"Este ascenso representa una transición de un régimen de alta independencia espacial hacia un estado de acoplamiento topológico irreversible."
                )
        else: # English default
            if level == "short":
                return f"The system exhibited {num_episodes} joint episodes, accumulating a maximum RECD of {recd_max:.2f}, before a critical transition at t*={t_star_text}."
            else:
                return (
                    f"Ontological Ascent Detected:\n"
                    f"- Layer 1 (Microscopic): The system reached a fractal dimension of {self.fractal_D:.2f}, with hyper-persistence Z={self.hp_z:.2f}.\n"
                    f"- Layer 2 (Mesoscopic): {num_episodes} episodes of dense entanglement crystallized.\n"
                    f"- Layer 3 (Macroscopic): Systemic time (RECD) accelerated to {recd_max:.2f}, marking the critical transition at t*={t_star_text}.\n\n"
                    f"This ascent represents a transition from a regime of high spatial independence to an irreversible state of topological coupling."
                )

    def to_json(self, full_dump: bool = False) -> str:
        """
        Serializes the object to a JSON string.
        By default (full_dump=False), it saves metadata and scalar markers (saving space).
        If full_dump=True, it serializes all internal numpy arrays as well.
        """
        data = {
            "version": "4.6.0",
            "metadata": self.metadata,
            "hp_z": self.hp_z,
            "lam": self.lam,
            "tt": self.tt,
            "M_max": self.M_max,
            "M_mean": self.M_mean,
            "episodes": self.episodes,
            "t_star": self.t_star,
            "t_frob": self.t_frob,
            "t_ks": self.t_ks,
            "frob_max": self.frob_max,
            "ks_max": self.ks_max,
            "fractal_D": self.fractal_D,
            "warnings": self.warnings,
            "component_names": self.component_names,
            "window_size": self.window_size
        }

        if full_dump:
            data["X"] = self.X.tolist()
            data["taus_global"] = self.taus_global.tolist()
            data["T_series"] = self.T_series.tolist()
            data["dtk_series"] = self.dtk_series.tolist()
            data["taus_per_module"] = self.taus_per_module.tolist()
            data["dyadic_k_series"] = self.dyadic_k_series.tolist()
            data["estimated_period_series"] = self.estimated_period_series.tolist()

        return json.dumps(data, cls=NumpyEncoder, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "OntologicalAscentResult":
        """
        Deserializes from a JSON string. Missing large arrays are initialized as empty.
        """
        data = json.loads(json_str)
        
        return cls(
            X=np.array(data.get("X", [])),
            window_size=data.get("window_size", 13),
            taus_global=np.array(data.get("taus_global", [])),
            T_series=np.array(data.get("T_series", [])),
            hp_z=data.get("hp_z", 0.0),
            lam=data.get("lam", 0.0),
            tt=data.get("tt", 0.0),
            M_max=data.get("M_max", 0.0),
            M_mean=data.get("M_mean", 0.0),
            episodes=data.get("episodes", []),
            t_star=data.get("t_star"),
            t_frob=data.get("t_frob"),
            t_ks=data.get("t_ks"),
            frob_max=data.get("frob_max"),
            ks_max=data.get("ks_max"),
            fractal_D=data.get("fractal_D"),
            warnings=data.get("warnings", []),
            metadata=data.get("metadata", {}),
            dtk_series=np.array(data.get("dtk_series", [])),
            taus_per_module=np.array(data.get("taus_per_module", [])),
            component_names=data.get("component_names", []),
            dyadic_k_series=np.array(data.get("dyadic_k_series", [])),
            estimated_period_series=np.array(data.get("estimated_period_series", []))
        )
