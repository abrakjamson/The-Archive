Copyright Abram Jackson 2024
All rights reserved

# The Archive: Civilization Rebuilder

(Development blog)[https://www.abramjackson.com/tag/the-archive/]

## Spec
## LLM Chain
1. have LLM identify the subjects of the query
2. search for those subjects in the index
3. load those pages into context with the query
4. have LLM generate response
5. check for groundedness and generate citations
6. display response and citations


## Design
 app.py
 * Starts program
 * Takes input (text or voice)
 * Displays results (text and voice)
 * Adds citations (queries DB)
 * Manages conversation and memory

 language_model.py
 * Executes LLM chain
 * Queries DB with RAG
 * Generates responses
 
local_wikipedia.py
* Indexes Wikipedia
* Queries Elasticsearch

 (future) Speech.py 
 * User's voice to text
 * Text to AI voice