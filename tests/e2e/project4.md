**Input Format:** Verbal Description via Chat Log

**User:** Storage is getting full, need a thumbnailer.
**Context:** We have terabytes of 4K bodycam screenshots sitting in the 'Pending' folder.
**Request:** I need a background worker that watches that folder. Whenever a new high-res image lands there, I need you to automatically make a small version (max 200px wide) and save it to the 'Previews' folder so the dashboard loads faster.

Oh, and keep the aspect ratio correct—don't stretch the evidence. If the image is corrupted, just log an error and move on, don't crash the worker. The file names should remain the same but with `_thumb` appended.
