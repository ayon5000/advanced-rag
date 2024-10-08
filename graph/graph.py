from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.state import GraphState
from graph.nodes import retrieve, generate, grade_documents, web_search
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.chains.router import RouteQuery, question_router

_ = load_dotenv()


def decide_to_generate(state: GraphState):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    print("---CHECK GENERATION VS DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallicination_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := hallicination_score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        answer_score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if answer_grade := answer_score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not_useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not_supported"


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


workflow: StateGraph = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GENERATE, generate)

# workflow.set_entry_point(RETRIEVE)
workflow.set_conditional_entry_point(
    route_question,
    path_map={
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={
        "useful": END,
        "not_useful": WEBSEARCH,
        "not_supported": GENERATE,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="adaptive_rag_graph.png")
