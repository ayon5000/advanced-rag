from dotenv import load_dotenv
from pprint import pprint

_ = load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.hallucination_grader import hallucination_grader, GradeHallunications
from graph.chains.generation import generation_chain
from graph.chains.router import question_router, RouteQuery
from ingestion import retriever


def test__retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test__retrieval_grader_answer_no() -> None:
    question = "how to make pizza"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "no"


def test__generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test__hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})

    res: GradeHallunications = hallucination_grader.invoke(
        {"generation": generation, "documents": docs}
    )

    assert res.binary_score == "yes"


def test__hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallunications = hallucination_grader.invoke(
        {"generation": "In order to make pizza we need dough.", "documents": docs}
    )

    assert res.binary_score == "no"


def test__router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})

    assert res.datasource == "vectorstore"


def test__router_to_websearch() -> None:
    question = "How to cook pizza"

    res: RouteQuery = question_router.invoke({"question": question})

    assert res.datasource == "websearch"
