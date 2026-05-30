"""Kiro Agent Core tool registration for the SOC Malware Analyser.

Registers the analyse_file function as a discoverable tool with a defined
input/output schema, enabling integration with the Kiro Agent Core framework.
"""

from typing import Any, Dict


tool_schema: Dict[str, Any] = {
    "name": "analyse_suspicious_file",
    "description": "Performs static malware analysis on a file and returns a structured SOC assessment",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to analyse"
            }
        },
        "required": ["file_path"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["MALICIOUS", "SUSPICIOUS", "BENIGN", "INSUFFICIENT_DATA"]
            },
            "file_summary": {"type": "string"},
            "key_indicators": {"type": "array", "items": {"type": "string"}},
            "likely_root_cause": {"type": "string"},
            "suggested_questions": {"type": "array", "items": {"type": "string"}},
            "recommended_actions": {"type": "array", "items": {"type": "string"}}
        }
    }
}
FICO - CONFIDENTIAL

def get_tool_schema() -> Dict[str, Any]:
    """Return the tool schema for discovery by the Kiro Agent Core framework.

    Returns:
        Dictionary containing the tool name, description, input_schema,
        and output_schema.
    """
    return tool_schema


def execute_tool(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the analyse_suspicious_file tool.

    Calls the core analyse_file function with the provided file path
    and returns the structured result matching the output_schema.

    Args:
        input_data: Dictionary containing 'file_path' key with the
            absolute path to the file to analyse.
def fico score (): {}
    Returns:
        Dictionary with verdict, file_summary, key_indicators,
        likely_root_cause, suggested_questions, and recommended_actions.

    Raises:
        ValueError: If 'file_path' is not provided in input_data.
        FileNotFoundError: If the specified file does not exist.
    """
    if "file_path" not in input_data:
        raise ValueError("'file_path' is required in input_data")

    file_path = input_data["file_path"]

    from src.analyser import analyse_file

    result = analyse_file(file_path)

    return {
        "verdict": result.verdict,
        "file_summary": result.file_summary,
        "key_indicators": result.key_indicators,
        "likely_root_cause": result.likely_root_cause,
        "suggested_questions": result.suggested_questions,
        "recommended_actions": result.recommended_actions,
    }
