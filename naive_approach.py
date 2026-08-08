import os
from dotenv import load_dotenv
import pypdf
import yaml
from rdflib import Graph, BNode, SH, Namespace
from langchain_openai import ChatOpenAI
from rdflib.plugins.sparql import prepareQuery
from pyshacl import validate
from langchain_core.messages import SystemMessage, HumanMessage

# Import your custom tools exactly as they are in your main script
from src.parsing_utils import read_txt
from src.testing_utils import apply_mutations, parse_validation_report

load_dotenv()


# 1. Initialize the LLM (Giving the baseline a fair fight with High thinking)
llm = ChatOpenAI(
    model="gpt-5.6-terra",
    temperature = 0,
    reasoning_effort = "high", 
    max_retries = 2
)

def extract_pdf_text(file_path: str) -> str:
    text_content = []
    with open(file_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_content.append(extracted + "\n")
    return "".join(text_content)

def run_zero_shot_baseline(document_name: str):
    print(f"\n🚀 Starting Zero-Shot Baseline Test for: {document_name}")

    # 2. Load your exact files
    pdf_path = f"Precondition documents/{document_name}.pdf"
    schema_path = f"Citizens/{document_name} schema.ttl"
    golden_path = f"Citizens/{document_name} eligible.ttl"
    yaml_path = f"Citizens/{document_name} scenarios.yaml"

    raw_greek_text = extract_pdf_text(pdf_path)
    citizen_schema = read_txt(schema_path)

    # 3. Formulate the "One Big Prompt"
    system_prompt = (
        "You are an expert in Semantic Web technologies, specifically RDF, SPARQL, and SHACL. "
        "Your task is to translate natural language legal documents into strict, executable graph logic."
    )
    
    human_prompt = f"""
Below is the official legal text outlining the eligibility requirements for a Greek public administration allowance. 
Please read the text, identify all the eligibility criteria, and write a complete SHACL shapes graph that can validate whether an applicant is eligible. 

CRITICAL INSTRUCTIONS:
1. You MUST use the exact classes and properties defined in the provided Ontology Schema below. Do not invent your own URIs.
2. You must express these rules using SHACL NodeShapes, PropertyShapes, and SPARQL-based constraints where necessary. 
3. Output the final result in valid Turtle (.ttl) format. Output only the text of the ttl, nothing else.

--- LEGAL TEXT ---
{raw_greek_text}

--- ONTOLOGY SCHEMA ---
{citizen_schema}
"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    # 4. Generate the SHACL in one shot
    print("🧠 Generating SHACL in one shot (Zero-Shot)...")
    response = llm.invoke(messages)
    shacl_output = response.text 
    
    # Save the output artifact
    os.makedirs("Testing_Artifacts_Naive", exist_ok=True)
    out_path = f"Testing_Artifacts_Naive/{document_name}_baseline_shacl.ttl"
    with open(out_path, "w", encoding='utf-8') as f:
        f.write(shacl_output)
    print(f"✅ Baseline SHACL saved to {out_path}")

    # ==========================================
    # 5. THE CRUCIBLE (Your exact Validator Logic)
    # ==========================================
    print("\n🧪 Validating against YAML Scenarios...")
    
    # Check RDF Syntax
    shacl_graph = Graph()
    try:
        shacl_graph.parse(data=shacl_output, format="turtle")
    except Exception as e:
        print(f"💥 CRITICAL FAILURE: The LLM generated invalid Turtle syntax. Error: {e}")
        return

    # Check SPARQL Syntax
    namespaces = dict(shacl_graph.namespaces())
    QUERY_FINDER = """PREFIX sh: <http://www.w3.org/ns/shacl#> SELECT ?sparql WHERE { ?node sh:select ?sparql . }"""
    for row in shacl_graph.query(QUERY_FINDER):
        try:
            prepareQuery(str(row.sparql), initNs=namespaces)
        except Exception as e:
            print(f"💥 CRITICAL FAILURE: SPARQL Syntax Error in generation: {e}")
            return

    # Load Golden Graph & YAML
    golden_graph = Graph()
    golden_graph.parse(data=read_txt(golden_path), format="turtle")
    golden_graph.bind("", Namespace("http://example.org/schema#"))

    with open(yaml_path, "r") as f:
        scenarios = yaml.safe_load(f)

    # Run the Mutations
    passed_scenarios = 0
    failed_scenarios = []

    for scn in scenarios:
        scenario_desc = scn['description']
        expected_count = scn['expected_violation_count']
        
        mutated_graph = apply_mutations(golden_graph, scn['actions'])

        conforms, results_graph, results_text = validate(
            data_graph=mutated_graph,
            shacl_graph=shacl_graph,    
            inference='rdfs',
        )
        
        parse_result = parse_validation_report(conforms, results_graph, results_text, shacl_graph)
        actual_count = parse_result["violation_count"]
        
        if actual_count == expected_count:
            passed_scenarios += 1
        else:
            failed_scenarios.append(f"- {scenario_desc} (Expected {expected_count}, got {actual_count})")

    # 6. Report the Carnage
    print("\n==========================================")
    print("📊 BASELINE ABLATION STUDY RESULTS")
    print("==========================================")
    print(f"Total Scenarios Tested: {len(scenarios)}")
    print(f"Scenarios Passed: {passed_scenarios}")
    print(f"Scenarios Failed: {len(failed_scenarios)}")
    
    if failed_scenarios:
        print("\n❌ Failed Scenarios Breakdown:")
        for fail in failed_scenarios:
            print(fail)
    print("==========================================\n")

if __name__ == "__main__":
    run_zero_shot_baseline("long_term_unemployment")