Copyright Abram Jackson 2024
All rights reserved

# The Archive: Civilization Rebuilder

## Spec
## LLM Chain
1. have LLM identify the subjects of the query
2. search for those subjects in the index
3. load those pages into context with the query
4. have LLM generate response
5. check for groundedness and generate citations
6. display response and citations


## Design
 Application.py
 * Starts program
 * Takes input (text or voice)
 * Displays results (text and voice)
 * Adds citations (queries DB)
 * Manages conversation and memory

 LLM.py
 * Executes LLM
 * Queries DB with RAG
 * Generates responses
 
 (future) Speech.py 
 * User's voice to text
 * Text to AI voice