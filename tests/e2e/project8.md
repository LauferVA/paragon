**Input Format:** Verbal Description via Email

**Subject:** Need to capture the weather station data streams

We've got about a thousand sensors out in the field. They all send a simple HTTP POST every minute with a JSON body like `{"temp": 22.4, "humidity": 60, "station_id": "ST-99", "timestamp": 1701234567}`.

I need a server that can catch all these requests and stick them in a database. But here's the catch: I need to be able to query it later to ask things like "Give me the average temperature for Station ST-99 over the last 24 hours" or "Show me the max humidity for all stations in Region A."

A standard SQL table gets too slow/clogged with this much raw write volume (1,000 writes/minute forever). Maybe think about how we store these numbers efficiently so the dashboard doesn't time out when asking for a month of data.
