"""
Academic reporting module for Systemic Tau.
This module is the primary entry point for generating publication-ready Markdown reports.
"""

from typing import Optional, List
from .results import OntologicalAscentResult
from .report import generate_academic_report as _generate_report

def generate_academic_report(
    result: OntologicalAscentResult,
    output_path: Optional[str] = None,
    location_name: Optional[str] = None,
    variables: Optional[List[str]] = None,
    language: str = "en"
) -> str:
    """
    Generate an academic Markdown report from an OntologicalAscentResult object.
    
    Args:
        result (OntologicalAscentResult): The completed analysis result.
        output_path (str, optional): Path to save the Markdown report. If None, only returns string.
        location_name (str, optional): Location or context name to append to the title.
        variables (list[str], optional): List of variable names used in the analysis.
        language (str, optional): Language code ('en' or 'es'). Defaults to 'en'.
        
    Returns:
        str: The generated Markdown report as a string.
    """
    return _generate_report(
        result=result,
        output_path=output_path,
        location_name=location_name,
        variables=variables,
        language=language
    )

__all__ = ["generate_academic_report"]
