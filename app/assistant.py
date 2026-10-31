from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.retriever import Retriever


class BankingAssistant:
    def __init__(self):
        self.retriever = Retriever()
        self.client = None
        if OPENAI_API_KEY:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)

    def answer(self, question: str, k: int = 3):
        documents = self.retriever.search(question, k=k)
        context = "\n".join(
            f"[{doc['id']}] {doc['text']}" for doc in documents
        )

        if self.client:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the supplied banking context. "
                            "If it does not contain the answer, say you do not "
                            "have enough information. Do not invent fees, rates, "
                            "eligibility rules, or legal requirements."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
            )
            answer = response.choices[0].message.content or ""
            mode = "llm"
        else:
            answer = "Retrieved information:\n" + context
            mode = "retrieval_only"

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": documents,
            "generation_mode": mode,
        }
