/*
 * SQL tests - persist to table
 */

DROP TABLE IF EXISTS pg_forecast_fcast_eval;
CREATE TABLE pg_forecast_fcast_eval(
    t TIMESTAMP PRIMARY KEY,
    x DOUBLE PRECISION
);
TRUNCATE TABLE pg_forecast_fcast_eval;
SELECT setseed(0.42);
INSERT INTO pg_forecast_fcast_eval(t, x)
SELECT
    ('2026-01-01'::date + s.i) AS t,
    random() AS x
FROM generate_series(1, 1000) AS s(i);

-- C version
SELECT  
    arima_forecast(
        ARRAY_AGG(x ORDER BY t DESC),
        '{0.25, 0.5}'::DOUBLE PRECISION[],
        2,
        2,
        0.0,
        '{0.25, 0.5}'::DOUBLE PRECISION[],
        '{0.3, 0.5}'::DOUBLE PRECISION[],
        7
    )
FROM
    pg_forecast_fcast_eval

-- SQL version
SELECT  
    arima_forecast_sql(
        ARRAY_AGG(x ORDER BY t ASC),
        '{0.25, 0.5}',
        2,
        2,
        0.0,
        '{0.25, 0.5}',
        '{0.3, 0.5}',
        7
    )
FROM
    pg_forecast_fcast_eval
LIMIT 2;