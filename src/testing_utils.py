"""Utilities for SHACL validation parsing, graph mutations and CSV logging in the testing pipeline."""

import csv
from rdflib import BNode, Graph, RDF, SH

def parse_validation_report(conforms: bool, results_graph: Graph, results_text: str, shacl_graph: Graph):
    """Parses the SHACL validation graph into a flat dictionary.

    Requires 'shacl_graph' to resolve Blank Node names for accurate shape identification.

    Args:
        conforms: Whether the validation passed.
        results_graph: The RDF graph containing validation results.
        results_text: The textual report from validation.
        shacl_graph: The SHACL shapes graph for resolving blank nodes.

    Returns:
        A dictionary with violation count, failed shapes, messages and full report.
    """
    if conforms:
        return {
            "violation_count": 0,
            "failed_shapes": "None",
            "messages": "None",
            "full_report": "Conforms: True"
        }

    failed_shapes = set()
    messages = []

    for result_node in results_graph.subjects(RDF.type, SH.ValidationResult):
        # Get the source shape that failed
        source_shape = results_graph.value(result_node, SH.sourceShape)
        shape_name = "Unknown"

        if source_shape:
            # Handle blank nodes by finding parent shapes
            if isinstance(source_shape, BNode):
                # Check for sh:property or sh:sparql parents
                parent = list(shacl_graph.subjects(SH.property, source_shape))
                if not parent:
                    parent = list(shacl_graph.subjects(SH.sparql, source_shape))

                if parent:
                    # Extract shape name from URI
                    shape_name = str(parent[0]).split("#")[-1].split("/")[-1]
            else:
                # Direct URI extraction
                shape_name = str(source_shape).split("#")[-1].split("/")[-1]

            failed_shapes.add(shape_name)

        # Get the validation message
        message = results_graph.value(result_node, SH.resultMessage)
        if message:
            messages.append(str(message))

    return {
        "violation_count": len(messages),
        "failed_shapes": "; ".join(sorted(list(failed_shapes))),
        "messages": " | ".join(messages),
        "full_report": results_text
    }
    

# Prefix header for Turtle mutations
PREFIX_HEADER = """
@prefix : <http://example.org/schema#> .
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

def apply_mutations(base_graph: Graph, actions) -> Graph:
    """Applies text-based Turtle mutations to a graph.

    Supports patching nodes by updating properties while preserving other data.

    Args:
        base_graph: The original RDF graph to mutate.
        actions: List of action dictionaries, each with 'type' and 'turtle' keys.

    Returns:
        A new graph with mutations applied.
    """
    # Clone the base graph
    new_graph = Graph()
    for ns in base_graph.namespaces():
        new_graph.bind(ns[0], ns[1])
    for triple in base_graph:
        new_graph.add(triple)

    for action in actions:
        # Handle patch_node actions
        if action['type'] == 'patch_node':
            snippet = action['turtle']

            # Parse the partial Turtle snippet
            full_turtle = PREFIX_HEADER + "\n" + snippet
            snippet_graph = Graph()
            try:
                snippet_graph.parse(data=full_turtle, format="turtle")
            except Exception as e:
                raise ValueError(f"Invalid Turtle in patch: {e}")

            # Apply updates: remove old values and add new ones
            for subject, predicate, obj in snippet_graph:
                # Remove old value for this property
                new_graph.remove((subject, predicate, None))
                # Add new value
                new_graph.add((subject, predicate, obj))

        elif action['type'] == 'no_action':
            pass

    return new_graph


# Write to csv
def flush_context_to_csv(context_dict, csv_file: str) -> None:
    """Appends a context dictionary as a row to an existing CSV file.

    Reads the CSV headers to ensure data alignment and appends the row.

    Args:
        context_dict: Dictionary of context data to write.
        csv_file: Path to the CSV file.
    """
    # Read headers from the existing file
    with open(csv_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Get the first row as headers

    # Align data to headers
    row_data = []
    for header in headers:
        value = context_dict.get(header, "N/A")  # Safe get with default
        row_data.append(value)

    # Append the row to the CSV
    with open(csv_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)
        
