from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from graph.consts import GENERATION_MODEL

llm = ChatOpenAI(
    model=GENERATION_MODEL,
    temperature=0,
)

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
