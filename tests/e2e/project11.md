**Input Format:** Verbal Description via Email

**Subject:** We need to find the printing errors in the new edition

We have the "Master Text" of *War and Peace*. We just got back 5,000 scanned pages from the print shop. The problem is, the pages are all jumbled, and the OCR (optical character recognition) isn't perfect—it sometimes reads 'e' as 'c'.

I need a tool where I feed in the "Master Text" and the big folder of "Scanned Snippets."

1.  **Placement:** Figure out where each snippet belongs in the book (they aren't in order).
2.  **Comparison:** Compare the snippet to the master text.
3.  **Noise Filtering:** Ignore the random OCR noise (like 'c' vs 'e' happening randomly).
4.  **Error Detection:** But if *every* snippet that covers Page 50, Line 10 shows the word "Doge" instead of "Dog", that's a real misprint. Flag it.
