
!pip install transformers
!pip install sentence_transformers
!pip install "pinecone-client[grpc]"
!pip install pandas

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from sentence_transformers import SentenceTransformer
import pinecone
from transformers import BertTokenizer, BertModel
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
embedding = BertModel.from_pretrained("bert-base-uncased")

query = "What is the book all about?"
query_tokens = tokenizer(query, padding=True, truncation=True, return_tensors="pt")

query_tokens

with torch.no_grad():
    outputs = embedding(**query_tokens)

with torch.no_grad():
    qa_output = embedding(**query_tokens)

query_embeddings = qa_output.last_hidden_state.mean(dim=1)

query_embeddings[0].shape

query_embeddings = torch.randn(768)

query_embeddings_np = query_embeddings.squeeze().cpu().numpy()

q_e_list = query_embeddings_np.tolist()

q_e_list[767]

query_embed_list = [{'id': str(i), 'values': query_embeddings_np[i].tolist()} for i in range(query_embeddings_np.shape[0])]

query_embed_list[4]

query_embed_list

df_query =  pd.DataFrame(data = query_embed_list)

df_query

document_embeddings = pinecone.query("document-embeddings", queries=query_embedding, top_k=10)

api_key = ''
pinecone.init(api_key=api_key,environment='gcp-starter'   )
index = pinecone.Index('bert')

vector=df_query['values']

type(vector)

results = index.query(vector = q_e_list, top_k=10)

results

answers = []
for result in results:
    document_id = result["id"]
    document_data = get_document_data(document_id)
    answers.append({"document_id": document_id, "text": document_data["text"]})

# Return the answers to the user
return answers