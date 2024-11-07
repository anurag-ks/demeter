import os
import warnings


from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from termcolor import colored

warnings.filterwarnings("ignore")

BOT_NAME = ""
CHROMA_PATH = "vector_store"
OPENAI_API_KEY=""

PROMPT_TEMPLATE = """
Assume the user is asking questions related to agriculture and
Answer the question based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

def main():
    while True:
        # Create CLI.
        query_text = input(colored("You: ", "red")) # User input

        if query_text == "exit":
            break

        # Prepare the DB.
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = ChatOpenAI()
        response_text = model.predict(prompt)
        formatted_response = f"{colored(BOT_NAME, 'green')}: {response_text}"
        print(formatted_response)
        print()


if __name__ == "__main__":
    main()