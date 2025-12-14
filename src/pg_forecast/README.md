# Building

To build from source, you will need the following packages installing (xx is PostgreSQL server version, e.g. 14):

- postgresql-server-dev-xx
- libnlopt-dev

# Running/Debugging

To view debug messages from psql:

```sql
SET client_min_messages = 'debug5';
```
