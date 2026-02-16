import chromadb
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


#1 Load the text
with open('Tesla2.txt', 'r') as current_file:
    text = current_file.read()

client = chromadb.Client()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=60)
chunks = splitter.split_text(text)

coll_c = client.create_collection(
    name="pro_strategy",
)

coll_c.add(ids=[f"c{i}" for i in range(len(chunks))], documents=chunks)



user_query = "User_Query: What is the name of this company?"

print(f"Query: {user_query}\n" + "="*30)
res = coll_c.query(query_texts=[user_query], n_results=3)
print(f"[2]: {res['documents'][0]} \n")
print(f"[3]: {res['documents'][0][0]} \n")

