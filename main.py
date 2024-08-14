from dotenv import load_dotenv
import pandas as pd
import os
from llama_index.experimental.query_engine.pandas import PandasQueryEngine
from prompt import instruction_str, new_prompt,context
from note_angine import note_engine
from llama_index.core.tools import QueryEngineTool,ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI



load_dotenv()

laptop = os.path.join("data", "laptop_cleaned.csv")
df = pd.read_csv(laptop)

query_engine = PandasQueryEngine(df = df,verbose=True,instruction_str=instruction_str)
query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name = "laptop_data",
            description = "give information about laptop data",
    ))
]

llm = OpenAI(model="gpt-3.5-turbo")
agent= ReActAgent.from_tools(tools,llm=llm,verbose=True,context = context)


while(prompt :=input("Enter a prompt(q or quit): ")) != "q":
    response = agent.query(prompt)
    print(response)
