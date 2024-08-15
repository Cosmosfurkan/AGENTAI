import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("Creating new index and saving to", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        print("Loading index from storage", index_name)
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    return index

current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "data", "Germany.pdf")

germany_pdf = PDFReader().load_data(file=pdf_path)
germany_index = get_index(germany_pdf, "germany")
germany_engine = germany_index.as_query_engine()


# # Terminalden sorgu almak ve çalıştırmak
# while True:
#     query = input("Enter your query about Germany (or type 'q' to quit): ")
#     if query.lower() in ['q', 'quit']:
#         break
#     response = germany_engine.query(query)
#     print(response)
