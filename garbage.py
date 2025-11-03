from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the following question only based on the provided context.
Context:
{context}
Question:
{question}

"""

prompt = ChatPromptTemplate.from_template(template)
print(prompt)
