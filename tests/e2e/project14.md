**Input Format:** Verbal Description via Chat

**User:** Need to index the Enron dump equivalent.
**Context:** We just got a hard drive with 5 million emails and PDF attachments for the upcoming trial. I need a local search engine.

**Request:**
You need to write a script that:
1.  Crawls through every folder recursively.
2.  Extracts text from PDFs, Word docs, and Emails.
3.  Splits the text into words (tokens).
4.  **The Important Part:** Builds a "Master Index" file.
    * If I search for "Project X" AND "Fraud", I need the list of matching filenames in under 1 second.
    * Do *not* just use `grep` (linear search). That is too slow. You need to build a reverse index (Word -> List of Files).
