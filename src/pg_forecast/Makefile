EXTENSION = pg_forecast
EXTVERSION = $(shell grep default_version $(EXTENSION).control | \
               sed -e "s/default_version[[:space:]]*=[[:space:]]*'\([^']*\)'/\1/")
               
DATA = $(filter-out $(wildcard sql/*--*.sql),$(wildcard sql/*.sql))
PG_CONFIG = pg_config

# Depends on 1 output SQL file
all: sql/$(EXTENSION)--$(EXTVERSION).sql

# Reconcatenate every time
.PHONY: sql/$(EXTENSION)--$(EXTVERSION).sql

# Concatenate all SQL files in tables/ and functions/
sql/$(EXTENSION)--$(EXTVERSION).sql: sql/tables/*.sql sql/functions/*.sql
	cat $^ > $@

DATA = $(wildcard updates/*--*.sql) sql/$(EXTENSION)--$(EXTVERSION).sql
EXTRA_CLEAN = sql/$(EXTENSION)--$(EXTVERSION).sql

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)