# AskAround demo (20/10)

AskAround is a Teams-style pop-up. A colleague types a question, and the tool answers in one of three ways depending on how confident it is. A knowledge graph (NetworkX) with personalized PageRank sits underneath.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
.venv/bin/python app.py                                             # then open http://localhost:5050
.venv/bin/python -m pytest tests                                    # 12 tests
```

## Demo script

The example chips under the input box cover all three levels:

| Question | Result | Why |
|---|---|---|
| How long must we keep payslips after an employee leaves? | **High**: ready-made answer | Answered before, verified on 15 Sep 2026, confirmed by 2 colleagues, 14 👍 |
| Is the end-of-year bonus subject to social security contributions? | **Medium**: verify with Chloé | The payroll guide changed after the last check, and only Bram gave this answer |
| What are the employer social security contributions for a first hire? | **Medium**: draft from the document | No earlier answer, but a strong matching document |
| Can we dismiss an employee who is on sick leave? | **Low**: route to Lotte | Legal topic without a verified answer |
| What price do we charge a retail customer with 500 employees…? | **Low**: route to Ines / Felix | Pricing topic without a verified answer |

Feedback loops you can show:
- **Medium, then "✓ Still correct", then "Ask again":** the answer becomes **High**, because it is now verified and confirmed by a second colleague.
- **Low, then paste the expert's answer and "Save", then "Ask again":** the answer is now reused at **Medium** level, because only one colleague has given it so far.
- **👍 / 👎 on a High answer:** nudges the answerers' scores in that area of expertise.
- **Reset demo:** restores the starting data.

## How it works

```
askaround/
  models.py       nodes (Employee, Document, Customer, Expertise), raw signals, StoredAnswer
  weights.py      one formula per edge type (same as SDworkx20_10)
  graph.py        builds the weighted, undirected nx.Graph
  text.py         TF-IDF "embedding" of questions and documents
  engine.py       personalized PageRank → confidence → high / medium / low
  sample_data.py  fictional people, documents, customers and earlier answers
app.py            Flask API + templates/index.html (the pop-up)
```

1. **Embed the question:** TF-IDF, compared with every document and with every earlier question.
2. **Personalized PageRank:** `nx.pagerank(G, alpha=0.85, personalization=…)`. The walk starts from the closest documents, weighted by similarity, plus the matching expertise nodes. It jumps back to that start 15% of the time.
   - Each edge type gets a *walk factor*. Employee–document links carry the most; customer links carry little. Without this, well-connected people and customers soak up the score.
   - A person's score is half PageRank and half their direct link to the starting documents, so the author of the best document stays on top.
3. **Answer memory:** the best earlier answer is rated on helpfulness (👍/👎), the age of its last verification, whether its sources changed since then, how many colleagues gave or confirmed it, and how reliable the answerer is for this question.
4. **Decision:**
   - **High:** confidence ≥ 0.70, verified within the last year, sources unchanged, and ≥ 2 colleagues gave or confirmed it.
   - **Low:** a high-risk topic (legal, compliance, pricing) without a verified answer, or nothing good enough was found.
   - **Medium:** everything in between. That is an answer with confidence ≥ 0.40, or a strong matching document that no one has answered from yet.
5. **Feedback:** thumbs and expert verifications update the person–document edges and a score per (person, area of expertise), which raises or lowers how confident AskAround is about that person.

All thresholds and factors are at the top of `engine.py` and in `weights.py`.

## Limitations

- The data is fictional and kept in memory, so a restart resets it. "Send in Teams" and the expert replies are simulated.
- TF-IDF only matches words, not meaning: "fire someone" won't find "dismissal" unless the words overlap. Text embeddings are the next step, and only `TfidfIndex` would need replacing.
- High-risk topics and expertise areas are detected with keyword lists.
- The concept says *directed* edges; this demo uses an undirected graph, as in SDworkx20_10.
