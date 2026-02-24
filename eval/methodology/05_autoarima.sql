DROP TABLE IF EXISTS pg_forecast_autoarima_eval;
CREATE TABLE pg_forecast_autoarima_eval(
    t TIMESTAMP PRIMARY KEY,
    x DOUBLE PRECISION
);
TRUNCATE TABLE pg_forecast_autoarima_eval;
SELECT setseed(0.42);
INSERT INTO pg_forecast_autoarima_eval(t, x)
SELECT
    ('2026-01-01'::date + s.i) AS t,
    random() AS x
FROM generate_series(1, 10000000) AS s(i);

-- AutoARIMA
DO $$
DECLARE
    v_start TIMESTAMP;
    v_end   TIMESTAMP;
    v_elapsed DOUBLE PRECISION;
    r INT;
BEGIN
    FOR r IN 1..3 LOOP
        v_start := clock_timestamp();
        PERFORM autoarima_train('pg_forecast_autoarima_eval', 't', 'x');
        v_end := clock_timestamp();
        v_elapsed := EXTRACT(EPOCH FROM (v_end - v_start)) * 1000;
        RAISE NOTICE 'AutoARIMA run % completed in % ms', r, v_elapsed;
    END LOOP;
END $$;

-- Individual models
DO $$
DECLARE
    v_start TIMESTAMP;
    v_end   TIMESTAMP;
    v_elapsed DOUBLE PRECISION;
    v_css DOUBLE PRECISION;
    r INT;

    -- model parameters
    rec RECORD;
BEGIN
    -- List of models to test
    FOR rec IN
        SELECT *
        FROM (VALUES
            ('ARIMA(0,2,0)',0,2,0),
            ('ARIMA(2,2,2)',2,2,2),
            ('ARIMA(1,2,0)',1,2,0),
            ('ARIMA(0,2,1)',0,2,1),
            ('ARIMA(1,2,1)',1,2,1),
            ('ARIMA(0,2,2)',0,2,2),
            ('ARIMA(1,2,2)',1,2,2),
            ('ARIMA(0,2,1)',0,2,1)  -- duplicate kept intentionally
        ) AS t(test_name, p, d, q)
    LOOP

        -- run each model 3 times
        FOR r IN 1..3 LOOP
            v_start := clock_timestamp();

            SELECT (arima_train(
                rec.p,
                rec.d,
                rec.q,
                'pg_forecast_autoarima_eval',
                't',
                'x',
                TRUE,
                TRUE
            )).css INTO v_css;

            v_end := clock_timestamp();
            v_elapsed := EXTRACT(EPOCH FROM (v_end - v_start)) * 1000;

            RAISE NOTICE '% trained in % ms with CSS %', rec.test_name, v_elapsed, v_css;

        END LOOP;

    END LOOP;
END $$;