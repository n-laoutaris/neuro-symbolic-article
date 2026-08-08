import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage
from langchain_core.messages import SystemMessage
from rdflib import Graph, BNode, SH, Namespace
from rdflib.plugins.sparql import prepareQuery
# import pyparsing
import pypdf
from dotenv import load_dotenv
import json
from pydantic import BaseModel
from typing import List
import yaml
from pyshacl import validate

from src.parsing_utils import read_txt
from src.graph_utils import visualize_graph, resolve_node_path
from src.testing_utils import apply_mutations, parse_validation_report

load_dotenv()
# llm = ChatGoogleGenerativeAI(
#     model = "gemini-3.1-flash-lite",
#     thinking_level = "minimal",
#     include_thoughts = True,
#     temperature = 0,
#     max_retries = 2
# )    

llm = ChatOpenAI(
    model="gpt-5.6-terra",
    temperature = 0,
    reasoning_effort = "low", 
    max_retries = 2
)

# ==========================================
# STATE
# ==========================================
# This defines the variables that get passed from node to node.
# We use the built-in message state to track the conversation and tool calls
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    file_name: str
    citizen_schema: str  # Holds the TTL string
    golden_citizen: str     # Holds the TTL string of the "golden citizen"
    mutation_scenarios: str  # Holds the YAML string of the mutation scenarios
    raw_greek_text: str
    preconditions: str
    json_model: str      # Holds the JSON information model as a string
    service_graph: str   # Holds the service graph in TTL format
    citizen_service_graph: Graph  # Holds the citizen-service graph as a Graph object
    shacl_shapes: str    # Holds the SHACL shapes in TTL format
    shacl_validation_status: str 
    shacl_validation_retries: int
    
# ==========================================
# PYDANTIC SCHEMA
# ==========================================

class Paths(BaseModel):
    path: List[str]
    datatype: str

class InformationConcept(BaseModel):
    name: str
    related_paths: List[Paths]  # Links the concept to citizen data available

class Constraint(BaseModel):
    name: str
    desc: str
    constrains: List[InformationConcept]
    
class ConstraintsList(BaseModel):
    constraints: List[Constraint]

# ==========================================
# NODES
# ==========================================

def ingestion_node(state: AgentState):
    """Deterministic PDF Parsing"""
    file_path = f"Precondition documents/{state['file_name']}.pdf"
    print(f"📄 [Ingestion] Reading {file_path}...")
    text_content = []
    try:
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content.append(extracted + "\n")
        return{"raw_greek_text": "".join(text_content)}
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extraction_node(state: AgentState):
    """LLM for Extraction & Translation"""
    print("🧠 [Extraction Agent] Isolating and translating preconditions...")
    
    system_prompt = read_txt(f'Prompts/extraction.txt')
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here is the raw document text:\n\n{state['raw_greek_text']}")
    ]
    
    response = llm.invoke(messages)
    return {"preconditions": response.text}

def json_structuring_node(state: AgentState):
    """LLM for JSON Information Model Structuring"""
    print("🏗️ [JSON Architect] Mapping preconditions to TTL ontology...")      

    system_prompt = read_txt(f'Prompts/JSON_structuring.txt')
    preconditions = state["preconditions"]
    citizen_schema = state["citizen_schema"]
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Preconditions:\n{preconditions}\n\nOntology:\n{citizen_schema}")
    ]
    
    # Force the LLM to output a list of Pydantic Constraints
    structured_llm = llm.with_structured_output(ConstraintsList)
    
    # The response is now a single ConstraintsList object
    response = structured_llm.invoke(messages)
    
    # Access the actual list using .constraints, then dump it to JSON
    json_string = json.dumps([obj.model_dump() for obj in response.constraints], indent=2, ensure_ascii=False)
    return {"json_model": json_string}

def service_graph_node(state: AgentState):
    """LLM for Service Graph Generation"""
    print("🕸️ [Graph Builder] Creating service graph from JSON model...")
    # Parse JSON string
    info_model = json.loads(state["json_model"])
    service_name = state["file_name"]
    PREFIXES = """@prefix ex: <http://example.org/> .
    @prefix cccev: <http://data.europa.eu/m8g/> .
    @prefix cpsv: <http://purl.org/vocab/cpsv#> .
    @prefix dct: <http://purl.org/dc/terms/> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    """
    triples = [PREFIXES]
    triples.append(f"ex:{service_name} a cpsv:PublicService .\n\n")

    # Convert constraints + concepts into triples
    for constraint in info_model:
        constraint_name = constraint["name"]
        constraint_desc = constraint["desc"].replace('"', '\\"')

        # Public service -> holdsRequirement -> constraint
        triples.append(f"ex:{service_name} cpsv:holdsRequirement ex:{constraint_name} .\n")

        # Constraint node
        triples.append(f'ex:{constraint_name} a cccev:Constraint ; dct:description "{constraint_desc}" .\n')

        # InformationConcept nodes
        for concept in constraint.get("constrains", []):
            concept_name = concept["name"]

            # Link constraint to concept
            triples.append(f"ex:{constraint_name} cccev:constrains ex:{concept_name} .\n")

            # Declare information concept
            triples.append(f'ex:{concept_name} a cccev:InformationConcept .\n')

        triples.append("\n")  # Spacing for readability

    service_graph_ttl = "".join(triples)
    return {"service_graph": service_graph_ttl}

def citizen_service_graph_node(state: AgentState):
    """LLM for Citizen Service Graph Generation"""
    print("🕸️ [Graph Builder] Creating citizen service graph...")
    EX = Namespace("http://example.org/")
    SC = Namespace("http://example.org/schema#")

    # Load service and citizen TTLs and info model
    citizen_ttl = state["golden_citizen"]
    service_graph_ttl = state["service_graph"]
    info_model = json.loads(state["json_model"])

    # Realize them into graphs
    unified_graph = Graph()
    unified_graph.parse(data=service_graph_ttl, format="turtle")
    citizen_graph = Graph()
    citizen_graph.parse(data=citizen_ttl, format="turtle")

    # Merge citizen triples into main graph
    for triple in citizen_graph:
        unified_graph.add(triple)

    # Automatically determine the root citizen node
    root_candidates = list(citizen_graph.subjects(predicate=None, object=SC.Applicant))
    citizen_root = root_candidates[0]

    # Add mapsTo edges
    for constraint in info_model:
        for concept in constraint["constrains"]:
            concept_uri = EX[concept["name"]]

            for path_obj in concept["related_paths"]:
                path_list = path_obj["path"]
                dtype = path_obj["datatype"]

                # Pass the datatype to the resolver
                subject_nodes = resolve_node_path(citizen_graph, citizen_root, path_list, dtype)

                for subj in subject_nodes:
                    # Connect the Information Concept to the Data Node
                    unified_graph.add((concept_uri, EX.mapsTo, subj))

    return {"citizen_service_graph": unified_graph}

def shacl_generator_node(state: AgentState):
    """LLM for SHACL Generation"""
    print("🔍 [SHACL Generator] Creating SHACL shapes from JSON model...")         

    system_prompt = read_txt(f'Prompts/SHACL_generation.txt')    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"JSON information model:\n\n{state['json_model']}\n\nOntology:\n{state['citizen_schema']}")
    ] + state.get("messages", [])  # Include any messages from previous validation attempts
    # Reminder: state messages only carry the conversation history after the system prompt and human context. So we append them at the end of the conversation start.
    
    if state.get("shacl_validation_retries", 0) >= 2:
        print("⚠️ 2 SHACL generation retries reached. Enabling Gemini 'low' thinking mode.")
        #   TODO: Implement escalation here.
        active_llm = llm
    else:
        active_llm = llm
    
    response = active_llm.invoke(messages)
    
    return {"messages": [response], "shacl_shapes": response.text}

def shacl_validator_node(state: AgentState):
    """
    Validates the syntactic correctness of a SHACL file in Turtle format.
    """
    print("⚙️ [Validator] Checking SHACL syntax and logic...")
    shacl_ttl_string = state["shacl_shapes"]

    ### RDF Syntax Validation
    graph = Graph()
    try:
        graph.parse(data=shacl_ttl_string, format="turtle")
    except Exception as e:
        # We catch the error and return it as a string for the LLM to read
        error_msg = str(e)
        print("❌ [Validator] RDF_SYNTAX_ERROR.")
        return {"shacl_validation_status": "RDF_SYNTAX_ERROR", 
                "shacl_validation_retries": state.get("shacl_validation_retries", 0) + 1,
                "messages": [HumanMessage(content=f"RDF Syntax Error found. Fix it and output the full corrected Turtle code again. Error details: \n\n {error_msg}.")]}
    
    # Embedded SPARQL Syntax Validation
    namespaces = dict(graph.namespaces())
    collected_errors = []
    QUERY_FINDER = """
    PREFIX sh: <http://www.w3.org/ns/shacl#>
    SELECT ?constraintNode ?sparql
    WHERE {
        ?constraintNode sh:select ?sparql .
    }
    """
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

            # Check if SPARQL compiles
            try:
                prepareQuery(sparql_string, initNs=namespaces)
            except Exception as e:
                msg = str(e)
                collected_errors.append(f"Shape: {shape_name}. Error details: {msg}.")

    except Exception as e:
        error_msg = str(e)       
        print("❌ [Validator] QUERY_EXTRACTION_ERROR.")
        return {"shacl_validation_status": "QUERY_EXTRACTION_ERROR", 
                "shacl_validation_retries": state.get("shacl_validation_retries", 0) + 1,
                "messages": [HumanMessage(content=f"Error found while extracting SPARQL queries. Error details: \n\n {error_msg}.")]}

    if collected_errors:
        n_errors = len(collected_errors)
        full_report = " \n\n ".join(collected_errors)
        print("❌ [Validator] SPARQL_SYNTAX_ERROR.")
        return {"shacl_validation_status": "SPARQL_SYNTAX_ERROR", 
                "shacl_validation_retries": state.get("shacl_validation_retries", 0) + 1,
                "messages": [HumanMessage(content=f"{n_errors} SPARQL syntax error(s) found. Fix them and output the full corrected Turtle code again. Full report: \n\n {full_report}.")]}
            
    ### SHACL Logic Validation
    # Load the Golden Citizen (Baseline) graph
    golden_ttl = state["golden_citizen"]
    golden_graph = Graph()
    golden_graph.parse(data=golden_ttl, format="turtle")
    golden_graph.bind("", Namespace("http://example.org/schema#"))
    
    # Load SHACL Shapes Graph
    shacl_ttl = state["shacl_shapes"]
    shacl_graph = Graph()
    shacl_graph.parse(data=shacl_ttl, format="turtle")
    shacl_graph.bind("", Namespace("http://example.org/schema#"))

    # Load the Scenarios from YAML
    # with open(f"Citizens/{state['file_name']} scenarios.yaml", "r") as f:
    scenarios = yaml.safe_load(state["mutation_scenarios"])

    # Iterate through each scenario
    for scn in scenarios:
        scenario_description = scn['description']
        expected_violation_count = scn['expected_violation_count']
        
        # Apply mutations to create a new mutated graph (leaving golden_graph untouched)
        mutated_graph = apply_mutations(golden_graph, scn['actions'])

        # Validate the mutated graph against SHACL shapes
        conforms, results_graph, results_text = validate(
            data_graph=mutated_graph,
            shacl_graph=shacl_graph,    
            inference='rdfs',
        )
        
        # Parse the validation report
        parse_result = parse_validation_report(conforms, results_graph, results_text, shacl_graph)
        actual_violation_count = parse_result["violation_count"]
        violated_shapes = parse_result["failed_shapes"]
        violation_messages = parse_result["messages"]
        
        if actual_violation_count != expected_violation_count:
            collected_errors.append(f"Scenario: {scenario_description}. Expected {expected_violation_count} violations but found {actual_violation_count}. Violated shapes: {violated_shapes}. Details: {violation_messages}")
            
    if collected_errors:
        n_errors = len(collected_errors)
        full_report = " \n\n ".join(collected_errors)
        print("❌ [Validator] LOGIC_VALIDATION_ERROR.")
        return {"shacl_validation_status": "LOGIC_VALIDATION_ERROR", 
                "shacl_validation_retries": state.get("shacl_validation_retries", 0) + 1,
                "messages": [HumanMessage(content=f"{n_errors} logic error(s) found. Fix them and output the full corrected Turtle code again. Full report: \n\n {full_report}.")]}
    
    print("✅ [Validator] No errors found.")
    return {"shacl_validation_status": "Valid"} 

def shacl_valid_condition(state: AgentState) -> str:
    """Check if the SHACL shapes are valid by looking at the validation status."""
    status = state.get("shacl_validation_status", "Unknown")
    if status == "Valid":
        return "Valid"
    else:
        return "Error"

def artifact_logger_node(state: AgentState):
    """Deterministic File Persistence"""
    # Check if both required parallel variables exist in the state yet
    if not state.get("citizen_service_graph") or not state.get("shacl_validation_status") == "Valid":
        # It's too early. One branch arrived before the other. Do nothing and exit.
        return
    
    artifact_path = "Testing_Artifacts_Test"
    file_name = state["file_name"]
    print(f"💾 [Artifact Logger] Saving Artifacts to {artifact_path}")    

    preconditions_path = f"{artifact_path}/{file_name}_preconditions.txt"    
    with open(preconditions_path, "w", encoding="utf-8") as f:
        f.write(state["preconditions"])
        
    json_model_path = f"{artifact_path}/{file_name}_information_model.json"
    with open(json_model_path, "w", encoding="utf-8") as f:
        f.write(state["json_model"])
        
    service_graph_path = f"{artifact_path}/{file_name}_service_graph.ttl"
    with open(service_graph_path, "w", encoding="utf-8") as f:
        f.write(state["service_graph"])
    visualize_graph(service_graph_path)  # Also saves its own HTML artifact
        
    citizen_service_graph_path = f"{artifact_path}/{file_name}_citizen_service_graph.ttl"
    state["citizen_service_graph"].serialize(citizen_service_graph_path, format="turtle")
    visualize_graph(citizen_service_graph_path) # Also saves its own HTML artifact
        
    shacl_path = f"{artifact_path}/{file_name}_shacl_shapes.ttl"
    with open(shacl_path, "w", encoding="utf-8") as f:
        f.write(state["shacl_shapes"])

    print(f"   [Success] Artifacts saved to: {artifact_path}")
    

# ==========================================
# THE GRAPH 
# ==========================================
print("\n--- Compiling LangGraph ---")
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("Ingest", ingestion_node)
workflow.add_node("Extract", extraction_node)
workflow.add_node("Structure_JSON", json_structuring_node)
workflow.add_node("Generate_Service_Graph", service_graph_node)
workflow.add_node("Generate_Citizen_Service_Graph", citizen_service_graph_node)
workflow.add_node("Generate_SHACL", shacl_generator_node)
workflow.add_node("Validate_SHACL", shacl_validator_node)
workflow.add_node("Log_Artifacts", artifact_logger_node)
# Edges
workflow.add_edge(START, "Ingest")
workflow.add_edge("Ingest", "Extract")
workflow.add_edge("Extract", "Structure_JSON")
workflow.add_edge("Structure_JSON", "Generate_Service_Graph")
workflow.add_edge("Generate_Service_Graph", "Generate_Citizen_Service_Graph")
workflow.add_edge("Generate_Citizen_Service_Graph", "Log_Artifacts")
workflow.add_edge("Structure_JSON", "Generate_SHACL")
workflow.add_edge("Generate_SHACL", "Validate_SHACL")
workflow.add_conditional_edges("Validate_SHACL", shacl_valid_condition, {"Valid": "Log_Artifacts", "Error": "Generate_SHACL"})
workflow.add_edge("Log_Artifacts", END)

# Compile into an executable application
app = workflow.compile()
# Save the graph visualization directly to project folder
with open("langgraph_architecture.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())

# ==========================================
# EXECUTE THE WORKFLOW
# ==========================================
print("\n--- Starting Execution ---")
document = "student_housing"
# document = "parental_leave"
initial_state = {"file_name": document,
                 "citizen_schema": read_txt(f"Citizens/{document} schema.ttl"),
                 "golden_citizen": read_txt(f"Citizens/{document} eligible.ttl"),
                 "mutation_scenarios": read_txt(f"Citizens/{document} scenarios.yaml")
                }
final_state = app.invoke(initial_state)

