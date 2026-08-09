import os
from dotenv import load_dotenv
import pypdf
import yaml
from rdflib import Graph, Namespace
from langchain_openai import ChatOpenAI
from rdflib.plugins.sparql import prepareQuery
from pyshacl import validate
from langchain_core.messages import SystemMessage, HumanMessage

from src.parsing_utils import read_txt, setup_run_log, append_run_log
from src.testing_utils import apply_mutations, parse_validation_report

load_dotenv()

# Initialize the LLM (Giving the baseline a fair fight with a strong model and High thinking)
llm = ChatOpenAI(
    model="gpt-5.6-sol",
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
    artifact_dir = "Testing_Artifacts_Naive"
    setup_run_log(document_name, artifact_dir=artifact_dir)

    message = f"\nStarting Zero-Shot Baseline Test for: {document_name}"
    print(message)
    append_run_log(message, document_name, artifact_dir)

    # Load files
    pdf_path = f"Precondition documents/{document_name}.pdf"
    schema_path = f"Citizens/{document_name} schema.ttl"
    golden_path = f"Citizens/{document_name} eligible.ttl"
    yaml_path = f"Citizens/{document_name} scenarios.yaml"

    message = f"📄 [Ingestion] Reading {pdf_path}..."
    print(message)
    append_run_log(message, document_name, artifact_dir)
    raw_greek_text = extract_pdf_text(pdf_path)

    citizen_schema = read_txt(schema_path)

    # Load the one-pass SHACL generation prompt from disk
    prompt_template = read_txt("Prompts/SHACL_generation_one_pass.txt")
    human_prompt = prompt_template.replace("{raw_greek_text}", raw_greek_text)
    human_prompt = human_prompt.replace("{citizen_schema}", citizen_schema)

    # Keep the model role stable while the prompt itself carries the domain-specific instructions.
    system_prompt = (
        "You are an expert in Semantic Web technologies, specifically RDF, SPARQL, and SHACL. "
        "Your task is to translate natural language legal documents into strict, executable graph logic."
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    # Generate the SHACL in one shot
    message = "🧠 [Baseline Generator] Generating SHACL in one shot (Zero-Shot)..."
    print(message)
    append_run_log(message, document_name, artifact_dir)
    response = llm.invoke(messages)
    shacl_output = response.text 
    
    # Save the output artifact
    os.makedirs(artifact_dir, exist_ok=True)
    out_path = f"{artifact_dir}/{document_name}_baseline_shacl.ttl"
    with open(out_path, "w", encoding='utf-8') as f:
        f.write(shacl_output)
    message = f"✅ [Baseline] SHACL saved to {out_path}"
    print(message)
    append_run_log(message, document_name, artifact_dir)

    # ==========================================
    # Validation Logic (same as in the original code)
    # ==========================================
    message = "\n🧪 [Validator] Validating against YAML Scenarios..."
    print(message)
    append_run_log(message, document_name, artifact_dir)
    
    # Check RDF Syntax
    shacl_graph = Graph()
    try:
        shacl_graph.parse(data=shacl_output, format="turtle")
    except Exception as e:
        message = f"💥 CRITICAL FAILURE: The LLM generated invalid Turtle syntax. Error: {e}"
        print(message)
        append_run_log(message, document_name, artifact_dir)
        return

    # Check SPARQL Syntax
    namespaces = dict(shacl_graph.namespaces())
    QUERY_FINDER = """PREFIX sh: <http://www.w3.org/ns/shacl#> SELECT ?sparql WHERE { ?node sh:select ?sparql . }"""
    for row in shacl_graph.query(QUERY_FINDER):
        try:
            prepareQuery(str(row.sparql), initNs=namespaces)
        except Exception as e:
            message = f"💥 CRITICAL FAILURE: SPARQL Syntax Error in generation: {e}"
            print(message)
            append_run_log(message, document_name, artifact_dir)
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
    failed_scenario_ids = []

    for scn in scenarios:
        scenario_desc = scn['description']
        scenario_id = scn.get('id', 'UNKNOWN')
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
            failed_scenario_ids.append(scenario_id)
            failed_scenarios.append(f"- {scenario_desc} (Expected {expected_count}, got {actual_count})")

    # Report

    message = "📊 BASELINE ABLATION RESULTS"
    print(message)
    append_run_log(message, document_name, artifact_dir)
    
    message = f"Total Scenarios Tested: {len(scenarios)}"
    print(message)
    append_run_log(message, document_name, artifact_dir)
    message = f"Scenarios Passed: {passed_scenarios}"
    print(message)
    append_run_log(message, document_name, artifact_dir)
    message = f"Scenarios Failed: {len(failed_scenarios)}"
    print(message)
    append_run_log(message, document_name, artifact_dir)

    if failed_scenario_ids:
        message = f"❌ [Validator] LOGIC_VALIDATION_ERROR. ({len(failed_scenario_ids)} errors: {', '.join(failed_scenario_ids)})"
        print(message)
        append_run_log(message, document_name, artifact_dir)
    else:
        message = "✅ [Validator] No errors found."
        print(message)
        append_run_log(message, document_name, artifact_dir)


if __name__ == "__main__":
    run_zero_shot_baseline("parental_leave")
    run_zero_shot_baseline("long_term_unemployment")
    run_zero_shot_baseline("student_housing")