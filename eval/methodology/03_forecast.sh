#!/bin/bash

DB="diss"
USER="postgres"

HORIZONS=(7 14 30 365 1000)
REPEATS=3

RAW_OUTPUT=$(mktemp)
CSV_OUTPUT="timings.csv"

echo "Running benchmark..."

########################################
# 1. Setup table
########################################
psql -d "$DB" -U "$USER" <<'SQL' > /dev/null
DROP TABLE IF EXISTS pg_forecast_fcast_eval;

CREATE TABLE pg_forecast_fcast_eval(
    t TIMESTAMP PRIMARY KEY,
    x DOUBLE PRECISION
);

SELECT setseed(0.42);

INSERT INTO pg_forecast_fcast_eval(t, x)
SELECT
    ('2026-01-01'::date + s.i) AS t,
    random() AS x
FROM generate_series(1, 1000000) AS s(i);
SQL

########################################
# 2. Run queries and capture output
########################################
for ((r=1; r<=REPEATS; r++)); do
    echo "========== RUN $r ==========" >> "$RAW_OUTPUT"

    for h in "${HORIZONS[@]}"; do
        echo "--- Horizon h=$h ---" >> "$RAW_OUTPUT"

        psql -d "$DB" -U "$USER" <<SQL >> "$RAW_OUTPUT" 2>&1
\timing on

SELECT 'C version';
SELECT arima_forecast(
    ARRAY_AGG(x ORDER BY t DESC),
    '{0.25, 0.5}'::DOUBLE PRECISION[],
    2,
    2,
    0.0,
    '{0.25, 0.5}'::DOUBLE PRECISION[],
    '{0.3, 0.5}'::DOUBLE PRECISION[],
    $h
)
FROM pg_forecast_fcast_eval;

SELECT 'SQL version';
SELECT arima_forecast_sql(
    ARRAY_AGG(x ORDER BY t ASC),
    '{0.25, 0.5}',
    2,
    2,
    0.0,
    '{0.25, 0.5}',
    '{0.3, 0.5}',
    $h
)
FROM pg_forecast_fcast_eval
LIMIT 2;

\timing off
SQL

    done
done

########################################
# 3. Parse into CSV
########################################
echo "Horizon,C Run 1,C Run 2,C Run 3,SQL Run 1,SQL Run 2,SQL Run 3" > "$CSV_OUTPUT"

awk '
BEGIN { run=0 }

/^========== RUN/ { run++ }

/--- Horizon h=/ {
    match($0, /h=([0-9]+)/, a);
    h=a[1];
}

/C version/ { mode="C" }
/SQL version/ { mode="SQL" }

/Time:/ {
    match($0, /Time: ([0-9.]+)/, a);
    t=a[1];

    if (mode=="C") {
        c[h,run]=t;
    } else if (mode=="SQL") {
        s[h,run]=t;
    }
}

END {
    horizons[1]=7; horizons[2]=14; horizons[3]=30; horizons[4]=365; horizons[5]=1000;

    for (i=1; i<=5; i++) {
        h=horizons[i];
        printf "%s,%s,%s,%s,%s,%s,%s\n",
            h,
            c[h,1], c[h,2], c[h,3],
            s[h,1], s[h,2], s[h,3];
    }
}
' "$RAW_OUTPUT" >> "$CSV_OUTPUT"

########################################
# 4. Done
########################################
echo "CSV written to $CSV_OUTPUT"
echo "Raw output saved to $RAW_OUTPUT"