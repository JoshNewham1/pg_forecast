/*
 * SQL tests - persist to table
 */

DROP TABLE IF EXISTS pg_forecast_kpss_eval;
CREATE TABLE pg_forecast_kpss_eval(
    t TIMESTAMP PRIMARY KEY,
    x DOUBLE PRECISION
);
-- Run 1
TRUNCATE TABLE pg_forecast_kpss_eval;
SELECT setseed(0.42);
INSERT INTO pg_forecast_kpss_eval(t, x)
SELECT
    ('2026-01-01'::date + s.i) AS t,
    random() AS x
FROM generate_series(1, 1000) AS s(i);
-- Run 2
TRUNCATE TABLE pg_forecast_kpss_eval;
INSERT INTO pg_forecast_kpss_eval(t, x)
SELECT
    ('2026-01-01'::date + s.i) AS t,
    random() AS x
FROM generate_series(1, 1000) AS s(i);
-- Run 3
TRUNCATE TABLE pg_forecast_kpss_eval;
INSERT INTO pg_forecast_kpss_eval(t, x)
SELECT
    ('2026-01-01'::date + s.i) AS t,
    random() AS x
FROM generate_series(1, 1000) AS s(i);

-- C version
SELECT  
    kpss(
        ARRAY_AGG(x ORDER BY t)
    )
FROM
    pg_forecast_kpss_eval;

-- SQL version
WITH
base AS (
    SELECT
        t,
        x,
        ROW_NUMBER() OVER (ORDER BY t) AS rn
    FROM pg_forecast_kpss_eval
),

stats AS (
    SELECT
        COUNT(*)::float8 AS n,
        AVG(x) AS mean_x
    FROM base
),

resid AS (
    SELECT
        b.rn,
        b.x - s.mean_x AS resid,
        s.n
    FROM base b
    CROSS JOIN stats s
),

cum AS (
    SELECT
        rn,
        resid,
        n,
        SUM(resid) OVER (ORDER BY rn) AS S
    FROM resid
),

eta_calc AS (
    SELECT
        n,
        SUM(S*S) AS eta
    FROM cum
    GROUP BY n
),

params AS (
    SELECT
        n,
        GREATEST(1, FLOOR(POWER(n, 2.0/9.0)))::int AS L
    FROM stats
),

gamma0 AS (
    SELECT SUM(resid*resid) AS g0 FROM resid
),

autocov AS (
    SELECT
        lag,
        SUM(r1.resid * r2.resid) AS g
    FROM params p
    CROSS JOIN generate_series(1, (SELECT L FROM params)) lag
    JOIN resid r1 ON r1.rn > lag
    JOIN resid r2 ON r2.rn = r1.rn - lag
    GROUP BY lag
),

lrvar AS (
    SELECT
        (
            g0 +
            COALESCE(SUM(
                2.0 *
                (1.0 - ac.lag::float8/(p.L+1)) *
                ac.g
            ), 0)
        ) / p.n AS lr_var
    FROM gamma0
    CROSS JOIN params p
    LEFT JOIN autocov ac ON TRUE
    GROUP BY g0, p.n, p.L
)

SELECT
    eta / (stats.n * stats.n * lrvar.lr_var) AS kpss_stat
FROM eta_calc
CROSS JOIN lrvar
CROSS JOIN stats;