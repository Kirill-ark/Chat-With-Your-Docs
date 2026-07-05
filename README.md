# Chat-With-Your-Docs

A Retrieval-Augmented Generation (RAG) app that answers questions strictly from your own
PDFs and cites the exact source (file and page). Local embeddings + vector database + LLM,
deployed to a live public URL.

---

## What is RAG and why it is needed

A large language model answers from its training memory. That creates two problems: it has
never seen your documents, and when it does not know an answer it tends to invent a
plausible one instead of admitting it does not know (hallucination).

RAG fixes this. Before the model answers, we search the user's own documents for the
passages relevant to the question, paste those passages into the prompt, and instruct the
model to answer using only that text and to say "I don't know" when the answer is not there.
The model stops relying on memory and answers from real, citable text — which is exactly why
every answer can point back to a source like "notes.pdf, p. 4".

Put simply: instead of a student answering an exam from memory, we let the same student open
the textbook to the right page first.

---

## How it works (the 6-step flow)

1. **Chunk** — split each PDF into small, overlapping pieces (~800 characters). Small pieces
   let retrieval return one focused passage instead of a whole chapter. The overlap makes
   neighbouring pieces share a bit of text so a sentence's meaning is never cut in half at a
   boundary.
2. **Embed** — turn each chunk into a vector: a list of numbers that captures its *meaning*.
   Texts with similar meaning end up close together, so we can later search by meaning rather
   than by exact words.
3. **Store** — save the vectors together with their metadata (filename, page number) in a
   vector database (ChromaDB), persisted locally.
4. **Retrieve** — embed the question with the *same* model, then ask the database for the
   top-k nearest chunks by similarity. This finds passages that match the *meaning* of the
   question, even when they do not share its exact words.
5. **Augment** — build the prompt from the retrieved chunks plus the question, so the model
   has the relevant source text in front of it.
6. **Generate** — the LLM answers using only that context and returns the sources (file,
   page) it relied on. If the answer is not in the context, it declines instead of inventing.

Two things this design depends on:

- **Same embedding model for steps 2 and 4.** If documents and questions are embedded by
  different models, their vectors live in different spaces and retrieval returns garbage.
- **The grounding prompt in step 6 is essential.** Without an explicit "use only the context /
  say you don't know" instruction, the model still hallucinates. This is tested on purpose.

---

## Status

Work in progress. This README grows phase by phase; evaluation results, the live demo link,
and setup instructions are added as the project reaches those phases.
