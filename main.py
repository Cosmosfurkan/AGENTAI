from dotenv import load_dotenv
import pandas as pd
import os
from llama_index.experimental.query_engine.pandas import PandasQueryEngine
from prompt import instruction_str, new_prompt,context
from note_angine import note_engine,save_note
from llama_index.core.tools import QueryEngineTool,ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import germany_engine


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
        ),
    ),
    QueryEngineTool(
        query_engine=germany_engine,
        metadata=ToolMetadata(
            name = "germiny_Data",
            description = "the detail information germin country",
        ),
    ),
]

llm = OpenAI(model="gpt-3.5-turbo")
agent= ReActAgent.from_tools(tools,llm=llm,verbose=True,context = context)


while (prompt := input("Enter a prompt (q or quit): ")) != "q":
    if "germany" in prompt.lower():
        response = germany_engine.query(prompt)
    else:
        response = query_engine.query(prompt)
    
    print(response)

    # Ask the user if they want to save the response as a note
    save_prompt = input("Do you want to save this response as a note? (y/n): ")
    if save_prompt.lower() == "y":
        save_note(str(response))  # Call the save_note function to save the note