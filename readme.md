Copyright Abram Jackson 2024
All rights reserved

# The Archive: Civilization Rebuilder

[Development blog](https://www.abramjackson.com/tag/the-archive/)

My new project is an entirely offline AI to answer any question you have. You can think of it like The Hitchhiker’s Guide to the Galaxy, or a Rabbit R1 that doesn’t have connection issues. Or perhaps it is more like Foundation’s Encyclopedia Galactica: an archive of human knowledge in case of the collapse of civilization. It should be a chatbot to answer any question but fit in your hand and not use an internet connection. It could teach you to build a ceramic kiln or help you remember that movie you saw once.

But I’m not making it to make money. Perhaps there is some value in this idea, but I doubt there is a lot of money. The primary goal is like Wikipedia’s:

```
to benefit readers by acting as a widely accessible and free encyclopedia. 
```
[Wikipedia:Purpose – Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Purpose)

And my secondary goal is to expand my own knowledge. While I know Python well enough to get around a notebook file, I have never set up and run a Python project myself. And while I’m an expert on productizing AI, I need a playground to try out new ideas and get a deeper understanding of the technology beyond a user and product manager. And finally, this is a good excuse to use AI as a programmer.

## Spec
### LLM Chain
1. have LLM identify the subjects of the query
2. search for those subjects in the index
3. load those pages into context with the query
4. have LLM generate response
5. in parallel, perform semantic search of the query
6. have another LLM use the results of the two search types to synthesize a response
7. (future) check for groundedness and generate citations
8. display response and (future) citations

### Orchestrator
1. Get user question
2. Execute primary LLM processing chain
3. Display response
4. (future) Manage conversation state
5. (future) Store memories

## Design
src/The_Archive/app.py
 * Starts program
 * Takes input
 * Displays results (text and voice)
 * (future) Adds citations (queries DB)
 * (future) Manages conversation and memory

src/The_Archive/language_model.py
 * Executes LLM chain
 * Queries DB with RAG
 * Generates responses
 
src/The_Archive/local_wikipedia.py
* Queries Elasticsearch with lexical
* Queries Elasticsearch with semantic

[Read more about how search is set up](https://www.abramjackson.com/artificial-intelligence/the-archive-pt-2-giant-vector-databases-with-big-gpus/)

src/The_Archive/prompt_templates.py
* Contains all the prompt templates used to answer questions

src/Test/app_test.py
* Runs The Archive's processing on the specified prompt
* Allows for changing the three system prompts or model

src/Test/promptfooconfig.yaml
* Evaluates 113 fact retrieval questions

[Read more about running LLM evals](https://www.abramjackson.com/artificial-intelligence/the-archive-pt-3-dont-hack-away-on-vibes-alone/)