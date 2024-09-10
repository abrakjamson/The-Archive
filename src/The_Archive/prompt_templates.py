class Prompt_Templates:
    def __init__(self, system_token="<|system|>", user_token="<|user|>", assistant_token="<|assistant|>", end_token="<|end|>"):
        self.system_token = system_token
        self.user_token = user_token
        self.assistant_token = assistant_token
        self.end_token = end_token

    def search_prompt(self):
        # newlines for phi go after the speaker and after the end token
        return f"""{self.system_token}
Give a comma-separated list of search keywords most likely to find an answer to the user's question. Do NOT answer the question.{self.end_token}
{self.user_token}
When was Obama born?{self.end_token}
{self.assistant_token}
Barack Obama,United States Presidents,Family of Barack Obama{self.end_token}
{self.user_token}
How can I make charcoal?{self.end_token}
{self.assistant_token}
Charcoal,Charcoal Kiln,Retort (Chemistry){self.end_token}
{self.user_token}
{{user_question}}{self.end_token}
{self.assistant_token}
"""

    def context_prompt(self):
        return f"""{self.system_token}
Directly answer the user's question.
Use only the following context information for the user's question:
{{context}}{self.end_token}
{self.user_token}
{{user_question}}{self.end_token}
{self.assistant_token}
"""

    def answer_prompt(self):
        return f"""{self.system_token}
Directly answer the user's question based on two other assistants' attempted answers.
Answer only based on the two assistants' answers. First assistant's answer:
{{lexical_context}}

Second assistant's answer:
{{semantic_context}}{self.end_token}
{self.user_token}
{{user_question}}{self.end_token}
{self.assistant_token}
"""
