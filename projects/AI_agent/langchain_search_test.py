from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import pprint
#https://rfriend.tistory.com/838

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
# os.environ["OPENAI_API_KEY"] = "sk-xxxx..." # set with yours

template = """You are an AI assistant. Answer the question.
If you don't know the answer, just say you don't know.

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
# HuggingFace Repository ID
repo_id = 'microsoft/Phi-3-mini-4k-instruct'

#https://wikidocs.net/233804
gpu_model = HuggingFacePipeline.from_model_id(
    model_id=repo_id,  # 사용할 모델의 ID를 지정합니다.
    task="text-generation",  # 수행할 작업을 설정합니다. 여기서는 텍스트 생성입니다.
    # 사용할 GPU 디바이스 번호를 지정합니다. "auto"로 설정하면 accelerate 라이브러리를 사용합니다.
    device=0,
    # 파이프라인에 전달할 추가 인자를 설정합니다. 여기서는 생성할 최대 토큰 수를 10으로 제한합니다.
    pipeline_kwargs={"temperature": 0.2, "max_new_tokens": 64},
)

parser = StrOutputParser()

chain = prompt | gpu_model | parser
print("BEFORE\n",chain.invoke({"question": "Which BTS member was the last to go to the military?"}))

######################################################
## Adding Web Search Tools
from langchain.tools import DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSerperAPIWrapper
os.environ["SERPER_API_KEY"] = ""
search_ = DuckDuckGoSearchRun() #따로 API 키 필요 없음
results = search_.run("Which BTS member was the last to go to the military?")
pprint.pp(results)
print("-----------------------------------------\n\n\n")

search = GoogleSerperAPIWrapper() # 키 필요함

## Provide "the latest context information" from web search
template = """Answer the question based on context.

Question: {question}
Context: {context}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
# model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

chain = (
    {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
    | prompt
    | gpu_model
    | parser
)
search_result = search.run("Which BTS member was the last to go to the military?")

print("AFTER#####################\n", search_result)

###############################
question = "Which BTS member was the last to go to the military?"

print("THIRD#####################\n", chain.invoke({"question": question, "context": search.run(question)}))

###################################
#https://bcho.tistory.com/1426
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

google_search = GoogleSerperAPIWrapper(k=20)
# google_search.k = 20
results = search.results("Which BTS member was the last to go to the military?")# kwargs= {"num":20})
pprint.pp(results)
print("====================================================\n\n\n")

# tools = [Tool(
# name="Intermediate Answer",
# func=google_search.run,
# description="useful for when you need to ask with search")
# ]
#
# agent = initialize_agent(tools = tools,
# llm = gpu_model,
# agent=AgentType.SELF_ASK_WITH_SEARCH,
# verbose=True)
# agent.run("Which BTS member was the last to go to the military?")
