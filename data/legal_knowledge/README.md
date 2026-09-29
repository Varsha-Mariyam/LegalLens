# Dataset D — Legal Knowledge Base (RAG corpus)

Controlled reference collection used by the RAG retriever. Organised by document type and topic,
as specified in the implementation guide (section 4, Dataset D).

* Every file starts with a small header (`Title`, `Jurisdiction`, `Topic`, `Sources`, `Note`).
* Content is a short **educational summary** written for this academic project. Each file names the
  statute sections or reported judgments it summarises so that every statement can be checked.
* Files whose sources say "General drafting practice" describe common contract-drafting conventions,
  not legal requirements, and say so.
* Nothing here is legal advice. Verify statutes at https://www.indiacode.nic.in (official portal).
* Several older labour statutes cited here (Payment of Wages Act, Industrial Disputes Act, Payment of
  Gratuity Act) are consolidated into India's Labour Codes. Always check which regime applies on the
  date of the document before relying on a section number.

Add a new topic by creating a `.txt` file with the same header format and running
`python scripts/build_vector_db.py --force`.
