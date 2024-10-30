from dotenv import load_dotenv
from haystack_integrations.components.embedders.cohere import CohereTextEmbedder
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.builders import PromptBuilder
# from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.rankers.cohere import CohereRanker
from haystack import Pipeline

load_dotenv()

USE_RANKER = True
USE_CHAT = False

document_store = PineconeDocumentStore(
  index="prototype",
  metric="cosine",
  dimension=384,
  spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
  )

template = """
En tenant compte des INFORMATIONS suivantes, répond à la QUESTION.

<INFORMATIONS>
{% for document in documents %}
    {{ document.content }}
{% endfor %}
</INFORMATIONS>

<QUESTION> 
{{question}} 
</QUESTION>

Ne jamais mentionner les INFORMATIONS fournies dans la RÉPONSE s'ils ne sont pas lié à la QUESTION. 
Si la question n'est pas compréhensible, ou l'INFORMATIONS n'est pas lié à la question, indiquez que vous n'avez pas l'information pour répondre. 
Répond dans la langue que la QUESTION est posé.
"""

text_embedder = CohereTextEmbedder(model="embed-multilingual-light-v3.0")
retriever = PineconeEmbeddingRetriever(document_store=document_store)

if USE_CHAT:
    system_message = ChatMessage.from_system("You are a prompt expert who answers questions based on the given documents.")
    messages = [system_message, ChatMessage.from_user(template)]
    prompt_builder = ChatPromptBuilder(variables=["documents","question"])
    # generator =  AnthropicChatGenerator(model="claude-3-haiku-20240307")
    generator = OpenAIChatGenerator(model="gpt-4o-mini")
else:
    prompt_builder = PromptBuilder(template=template)
    generator = OpenAIGenerator(model="gpt-4o-mini")

if USE_RANKER:
    ranker = CohereRanker(model="rerank-multilingual-v3.0")

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
if USE_RANKER:
    rag_pipeline.add_component(instance=ranker, name="ranker")
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", generator)

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
if USE_RANKER:
    rag_pipeline.connect("retriever.documents", "ranker.documents")
    rag_pipeline.connect("ranker", "prompt_builder.documents")
else:
    rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")


def run_pipeline(question):
    
    if USE_RANKER:
        response = rag_pipeline.run({"text_embedder": {"text": question}, 
                                # "prompt_builder": {"template_variables": {"query": question}, "template": messages},
                                "prompt_builder": {"question": question},
                                "retriever": {"top_k": 5},
                                "ranker": {"query": question, "top_k": 3},
                                })
    else:
        response = rag_pipeline.run({"text_embedder": {"text": question}, 
                                # "prompt_builder": {"template_variables": {"query": question}, "template": messages},
                                "prompt_builder": {"question": question},
                                "retriever": {"top_k": 5},
                                })

    # reply_content = response["llm"]["replies"][0].content
    reply_content = response["llm"]["replies"][0]

    return reply_content
