"""Utilities for RDF graph operations, including visualization, hashing, validation and path resolution."""

# Standard library imports
import hashlib
import os
import webbrowser

# Third-party imports
import pyparsing
from pyvis.network import Network
from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, SH
from rdflib.compare import to_isomorphic
from rdflib.plugins.parsers.notation3 import BadSyntax
from rdflib.plugins.sparql import prepareQuery

# Namespaces
CCCEV = Namespace("http://data.europa.eu/m8g/")
CPSV = Namespace("http://purl.org/vocab/cpsv#")
SC = Namespace("http://example.org/schema#")

# Pyvis graph visualization
def visualize_graph(ttl_file: str, open_in_browser: bool = False) -> None:
    """Visualizes an RDF graph from a TTL file using PyVis.

    Loads the TTL file, creates an interactive network graph and saves it as HTML.
    Optionally opens the HTML file in the default web browser.

    Args:
        ttl_file: Path to the Turtle (.ttl) file containing the RDF graph.
        open_in_browser: If True, opens the generated HTML file in the browser.
    """
    # Load TTL file
    graph = Graph()
    graph.parse(ttl_file, format="turtle")  

    network = Network(height="1440px", width="100%", notebook=True, directed=True, cdn_resources='remote')
    network.force_atlas_2based()

    # Determine node color based on RDF type for visual distinction
    def node_color(uri):
        if (uri, RDF.type, CPSV.PublicService) in graph:
            return "gold"
        if (uri, RDF.type, CCCEV.Constraint) in graph:
            return "maroon"
        if (uri, RDF.type, CCCEV.InformationConcept) in graph:
            return "darkturquoise"
        if (uri, RDF.type, SC.Applicant) in graph:
            return "yellowgreen"
        return "lightgrey"

    # Generate readable label for nodes
    def node_label(uri):
        if isinstance(uri, Literal):
            return str(uri)

        for lbl in graph.objects(uri, RDFS.label):
            return str(lbl)
        for lbl in graph.objects(uri, CCCEV.name):
            return str(lbl)

        uri_str = str(uri).rstrip('/')  # Safety fix for trailing slashes
        if "#" in uri_str:
            return uri_str.split("#")[-1]
        return uri_str.split("/")[-1]

    # Add nodes and edges, skipping rdf:type for better readability
    for subject, predicate, obj in graph:
        if predicate == RDF.type:
            continue

        # Add subject node
        network.add_node(str(subject), label=node_label(subject), color=node_color(subject))

        # Add object node
        if isinstance(obj, Literal):
            network.add_node(str(obj), label=node_label(obj), color="beige", shape="box")
        else:
            network.add_node(str(obj), label=node_label(obj), color=node_color(obj))

        # Add edge
        network.add_edge(str(subject), str(obj), label=node_label(predicate), arrows="to")

    html_file = ttl_file.replace("ttl", "html")
    # Render and save graph
    network.save_graph(html_file)
    if open_in_browser:
        webbrowser.open("file://" + os.path.abspath(html_file))  # Use absolute path for browser


# Hashing Graphs
def get_semantic_hash(rdf_text: str) -> str:
    """Generates a hash of RDF text based on its logical structure.

    Parses the RDF, canonicalizes it to ignore formatting differences and computes
    an MD5 hash of the sorted triples. Useful for detecting equivalence.

    Args:
        rdf_text: The RDF content as a Turtle-formatted string.

    Returns:
        The MD5 hash as a hexadecimal string, or "INVALID_RDF" if parsing fails.
    """
    graph = Graph()
    try:
        graph.parse(data=rdf_text, format="turtle")
    except Exception:
        return "INVALID_RDF"

    iso_graph = to_isomorphic(graph)

    # Generate deterministic hash from sorted triples
    triples = sorted(list(iso_graph))
    triples_string = "".join([str(t) for t in triples])
    return hashlib.md5(triples_string.encode('utf-8')).hexdigest()


# SPARQL query to find constraint nodes with embedded queries
QUERY_FINDER = """
    PREFIX sh: <http://www.w3.org/ns/shacl#>
    SELECT ?constraintNode ?sparql
    WHERE {
        ?constraintNode sh:select ?sparql .
    }
"""

def validate_shacl_syntax(shacl_ttl_string: str) -> tuple[bool, str, str]:
    """Validates the syntactic correctness of a SHACL file at multiple levels.

    Performs two levels of validation:
    1. RDF syntax parsing.
    2. Embedded SPARQL syntax checking.
    Resolves blank nodes to parent shape names for error readability.

    Args:
        shacl_ttl_string: The SHACL constraints as a Turtle-formatted string.

    Returns:
        Tuple:
        - is_valid: True if valid, False otherwise.
        - error_type: "VALID", "RDF_SYNTAX", "SPARQL_SYNTAX", or "QUERY_EXTRACTION".
        - details: Error message or "OK" if valid.
    """
    graph = Graph()

    # RDF Syntax Validation
    try:
        graph.parse(data=shacl_ttl_string, format="turtle")
    except (BadSyntax, Exception) as e:
        return False, "RDF_SYNTAX", str(e).replace("\n", " ")

    namespaces = dict(graph.namespaces())

    # Embedded SPARQL Syntax Validation
    collected_errors = []

    try:
        results = graph.query(QUERY_FINDER)

        for row in results:
            constraint_node = row.constraintNode
            sparql_string = str(row.sparql)

            # Resolve shape name for better error messages
            shape_name = "Unknown_Shape"

            if isinstance(constraint_node, BNode):
                # Blank node: find parent via sh:sparql or sh:property
                parents = list(graph.subjects(SH.sparql, constraint_node))
                if parents:
                    shape_name = str(parents[0]).split("#")[-1].split("/")[-1]
                else:
                    parents = list(graph.subjects(SH.property, constraint_node))
                    if parents:
                        shape_name = str(parents[0]).split("#")[-1].split("/")[-1] + "_Prop"
            else:
                # Named URI
                shape_name = str(constraint_node).split("#")[-1].split("/")[-1]

            # Check SPARQL compilation
            try:
                prepareQuery(sparql_string, initNs=namespaces)
            except pyparsing.ParseException as pe:
                msg = str(pe).replace("\n", " ").split("(at char")[0].strip()
                collected_errors.append(f"{shape_name}: {msg}")
            except Exception as e:
                msg = str(e).replace("\n", " ")
                collected_errors.append(f"{shape_name}: {msg}")

    except Exception as e:
        return False, "QUERY_EXTRACTION", str(e)

    if collected_errors:
        full_report = " | ".join(collected_errors)
        return False, "SPARQL_SYNTAX", full_report

    return True, "VALID", "OK"


def resolve_node_path(citizen_graph: Graph, root_uri: str, path_list: list[str], datatype: str) -> set[str]:
    """Resolves a path of predicates from a root URI to find related nodes.

    Traverses the graph along the given path, handling URI and literal datatypes differently.
    For URIs, traverses the full path. For literals, stops one step early to get the node holding the value.

    Args:
        citizen_graph: The RDF graph to traverse.
        root_uri: The starting URI node.
        path_list: List of predicate names (without namespace) to follow.
        datatype: "URI" for identity logic (e.g., City, Person), "literal" for value logic (e.g., Income).

    Returns:
        A set of URIs reached at the end of the path.
    """
    # Determine traversal depth based on datatype
    if datatype == "URI":
        # For identity logic, the value IS the node
        traversal_parts = path_list
    else:
        # For value logic, stop before the literal to get the holding node
        traversal_parts = path_list[:-1]

    # Traverse the graph
    current_nodes = {root_uri}

    for part in traversal_parts:
        next_nodes = set()
        predicate = SC[part]  # Assumes schema matches the namespace

        for node in current_nodes:
            # Find objects connected by this predicate
            for obj in citizen_graph.objects(node, predicate):
                # Skip literals if expecting URIs to avoid data errors
                if isinstance(obj, Literal) and datatype == "URI":
                    continue
                next_nodes.add(obj)

        current_nodes = next_nodes

        # Early exit if no nodes found
        if not current_nodes:
            break

    return current_nodes