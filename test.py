from src.utils.embed_store_data import load_embed_model
from langchain_community.vectorstores import FAISS
embed_model = "BAAI/bge-small-en"
DEFAULT_VECTOR_DB_PATH = "vector_store/faiss_db"
from src.utils.embed_store_data import load_embed_model
embeddings = load_embed_model(model_name=embed_model)
print("Loading vector store")
    
db = FAISS.load_local(DEFAULT_VECTOR_DB_PATH, embeddings,
                              allow_dangerous_deserialization=True)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2, "fetch_k": 2},
)

bg = retriever.invoke("Mức phạt khi không thực hiện đúng quy trình tuyển sinh theo quy định của pháp luật là bao nhiêu?")
print(bg)