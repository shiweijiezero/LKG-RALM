from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

system_prompt_1 = (
    "You are a helpful assistant for question-answering tasks. "
    # "Use the following pieces of retrieved context to answer "
    # "the question. If you don't know the answer, say that you "
    # "don't know. "
    "The references contain noise, so you "
    "may need to use your judgement to determine"
)

user_prompt_1 = (
    "References: {context}\n"
    "Question: {question}\n"
    "Please do not response irrelevant information or noise about the question."
    # "Helpful Answer:"
)

system_prompt_2 = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
    "You are a helpful assistant for question-answering tasks. The references contain noise, so you need to use your judgement.<|eot_id|>"
)

user_prompt_2 = (
    "<|start_header_id|>user<|end_header_id|>"
    "References: {context}\n"
    "Question: {question}\n"
    "Please do not response irrelevant information or noise about the question."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)

openai_user_prompt = (
    "References: {context}\n"
    "Question: {question}\n"
    "Please do not response irrelevant information or noise about the question."
)

template = \
"""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.
Contexts: {context}

Question: {question}

Helpful Answer:"""

template_1 = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
    "You are a helpful assistant for question-answering tasks. The references contain noise, so you need to use your judgement.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>"
    "References: {context}\n"
    "Question: {question}"
    "Please do not response irrelevant information or noise about the question."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)

template_2 = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
    "You are a helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>"
    "Context: {context}\n"
    "<Question>: {question}\n"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)

few_shot_template = \
"""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.

Example 1: 
genre of diary of a wimpy kid the getaway
Children's novel

Example 2:
what is billy last name in where the red fern grows
Colman

Example 3:
who plays heather in beauty and the beast
Nicole Gale Anderson

Contexts: {context}

Question: {question}

Helpful Answer:"""


rewrite_template = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
    "You are a helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>"
    "Provide a better search query for"
    "web search engine to answer the given question, end"
    "the queries with ’**’. Question: {x} Answer:"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)

openai_rewrite_template = (
    "Provide a better search query for "
    "web search engine to answer the given question, end "
    "the queries with ’**’. Question: {x} Answer:"
)

class MyPrompt:
    @staticmethod
    def get_chat_prompt():
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{question}"),
            ]
        )
        return prompt

    @staticmethod
    def get_chat_prompt_1():
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt_1),
                HumanMessagePromptTemplate.from_template(user_prompt_1)
            ]
        )
        return prompt

    @staticmethod
    def get_chat_prompt_2():
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt_2),
                HumanMessagePromptTemplate.from_template(user_prompt_2)
            ]
        )
        return prompt

    @staticmethod
    def get_completion_prompt():
        prompt = PromptTemplate.from_template(template)
        return prompt

    @staticmethod
    def get_completion_prompt_2():
        prompt = PromptTemplate.from_template(template_2)
        return prompt


    @staticmethod
    def get_few_shot_prompt():
        prompt = PromptTemplate.from_template(few_shot_template)
        return prompt

    @staticmethod
    def get_rewrite_prompt():
        prompt = PromptTemplate.from_template(rewrite_template)
        return prompt

    @staticmethod
    def get_openai_prompt():
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_1),
                ("human", openai_user_prompt),
            ]
        )
        return prompt

    @staticmethod
    def get_openai_rewrite_prompt():
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", openai_rewrite_template),
            ]
        )
        return prompt