# Building

To build from source, you will need the following packages installing (xx is PostgreSQL server version, e.g. 14):

- gcc
- make
- postgresql-server-dev-xx
- libnlopt-dev

# Running/Debugging

To view debug messages from psql:

```sql
SET client_min_messages = 'debug5';
```

To build and install the plugin from source:

```bash
sudo make install && sudo make clean && sudo -u postgres psql -d DB_NAME -c "DROP EXTENSION IF EXISTS pg_forecast CASCADE; CREATE EXTENSION pg_forecast;"
```
