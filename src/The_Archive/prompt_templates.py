""" 
Copyright Abram Jackson 2024
All rights reserved
"""

class Prompt_Templates:
    def __init__(self, system_token="<|system|>", user_token="<|user|>", assistant_token="<|assistant|>", end_token="<|end|>"):
        """
        Sets the format of a conversation and special tokens.
        Default is Phi-style.
        """
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
    
    def retrieval_generation_prompt(self):
        return f"""{self.system_token}
Your task is to write JSONL questions and answers to evaluate a search system.
You will give five factual questions from the specified article with answers.
The questions should be obscure but easy to identify from the article text.
The answers should be a single short and unambiguous string (one to three words) that the search engine should include in its result.
The entire response should be formatted as JSONL without any additional commentary or markdown. Here is the format to use:

["question": "What month was Barack Obama born?", "answer": "august"]

If the article is unsuitable for essay questions, return only a newline.{self.end_token}
{self.user_token}
article_id: 12
title: "Anarchy"
text: "Anarchy is a form of society without... (full text omitted from conversation history)"{self.end_token}
{self.assistant_token}
["question": "When did Donatien Alphonse François write Juliette?", "answer": "1797"]
["question": "What Greek word is 'anarchy' derived from?'", "answer": "αναρχία"]
["question": "Who wrote 'The Limits of State Action' about anarchy?", "answer": "Wilhelm von Humboldt"]
["question": "When was the International Workingmen's Association founded?", "answer": "1864"]
["question": "Which of Thomas Hobbes or John Locke disapproved of anarchy?", "answer": "Thomas Hobbes"]{self.end_token}
{self.user_token}
article_id: {{article_id}}
title: {{article_title}}
text: {{article_text}}{self.end_token}
{self.assistant_token}"""
        
    def essay_generation_prompt(self):
        return f"""{self.system_token}
{{article_identifier}} {{article_title}} {{article_text}}
Give five JSON Lines of essay questions based on the article text.
The entire response should be formatted as JSONL without any additional commentary.
If the article is unsuitable for essay questions, return only a newline.{self.end_token}
{self.user_token}
{{"article_id": "12", "text": "Anarchy is a form of society without... (full text omitted from conversation history)" }}{self.end_token}
{self.assistant_token}
{{"article_id": "12", "question": "Describe three major criticisms of Anarchy" }}
{{"article_id": "12", "question": "What are the origins of the word 'anarchy'?" }}
{{"article_id": "12", "question": "Why is anarchy considered a social disorder?" }}
{{"article_id": "12", "question": "Describe how John Calvin compared anarchy to tyranny and why." }}
{{"article_id": "12", "question": "What is the 'state of nature' according to John Locke?" }}{self.end_token}
{self.user_token}
{{"article_id": {{article_identifier}}, "title": {{article_title}}, "text": {{article_text}} }}{self.end_token}
{self.assistant_token}"""