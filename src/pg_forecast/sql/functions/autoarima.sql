CREATE TYPE kpss_result AS (
    kpss_val DOUBLE PRECISION,
    crit_val DOUBLE PRECISION,
    test_passed BOOLEAN
);

CREATE FUNCTION kpss(
    vals DOUBLE PRECISION[], -- Ordered array of time-series values
    p_val DOUBLE PRECISION DEFAULT 0.05 -- (0.1, 0.05, 0.025, 0.01)
)
RETURNS kpss_result
AS 'MODULE_PATHNAME', 'kpss'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION aicc(
    css DOUBLE PRECISION,
    p INT,
    q INT,
    k INT, -- Number of parameters
    n_vals INT
)
RETURNS DOUBLE PRECISION
AS 'MODULE_PATHNAME', 'aicc'
LANGUAGE C STRICT STABLE;

CREATE TYPE autoarima_rec AS (
    p INT,
    d INT,
    q INT,
    c DOUBLE PRECISION,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    css DOUBLE PRECISION,
    aicc DOUBLE PRECISION
);

CREATE OR REPLACE FUNCTION autoarima_train(
    horizon INT,
    source_table TEXT, -- Table/view name
    date_col TEXT,     -- Timestamp column
    value_col TEXT     -- Numerical value column
)
RETURNS autoarima_rec AS $$
DECLARE
    vals DOUBLE PRECISION[];
    n_vals INT;
    diff_vals DOUBLE PRECISION[];
    v_p INT;
    v_d INT := 0;
    v_q INT;
    kpss_res kpss_result;
    include_c BOOLEAN := TRUE;
    to_train RECORD;
    trained arima_optimise_result;
    n_params INT;
    model_aicc DOUBLE PRECISION;
    current_model RECORD;
    best_aicc DOUBLE PRECISION := '-Infinity';
BEGIN
    -- Aggregate the series into an array
    EXECUTE format(
        'SELECT array_agg(%I ORDER BY %I) FROM %I',
        value_col,
        date_col,
        source_table
    )
    INTO vals;

    n_vals := array_length(vals, 1);

    -- Determine the number of differences with repeated KPSS tests
    diff_vals := vals;
    FOR i IN 0..3 LOOP
        kpss_res := kpss(diff_vals);
        EXIT WHEN kpss_res.test_passed;
        v_d := v_d + 1;
        -- If test failed, difference and try again
        diff_vals := arima_difference(diff_vals, v_d);
    END LOOP;

    IF v_d > 2 THEN
        RAISE EXCEPTION 'More than 2 differences required, ARIMA model not suitable';
    END IF;

    RAISE NOTICE '% differences required', v_d;

    -- Fit 4 initial models - ARIMA(0,d,0), ARIMA(2,d,2), ARIMA(1,d,0), ARIMA(0,d,1)
    IF v_d = 2 THEN
        include_c := FALSE;
    END IF;

    DROP TABLE IF EXISTS tmp_autoarima;
    CREATE TEMP TABLE tmp_autoarima
    OF autoarima_rec
    ON COMMIT DROP;

    FOR to_train IN
        SELECT * FROM (VALUES 
            (0, 0), (2, 2), (1, 0), (0, 1)
        ) AS m(p, q)
    LOOP
        trained := arima_train(to_train.p, v_d, to_train.q, source_table, date_col, value_col, include_c);
        n_params := to_train.p + to_train.q;
        IF include_c THEN
            n_params := n_params + 1;
        END IF;
        model_aicc := aicc(trained.css, to_train.p, to_train.q, n_params, n_vals);
        INSERT INTO tmp_autoarima
        VALUES (
            to_train.p,
            v_d,
            to_train.q,
            trained.c,
            trained.phi,
            trained.theta,
            trained.css,
            model_aicc
        );
    END LOOP;

    -- If d <= 1, ARIMA(0, d, 0) without a constant is also fitted
    IF v_d <= 1 THEN
        trained := arima_train(0, v_d, 0, source_table, date_col, value_col, FALSE);
        model_aicc := aicc(trained.css, to_train.p, to_train.q, 0, n_vals);
        INSERT INTO tmp_autoarima
        VALUES (0, v_d, 0, 0, trained.phi, trained.theta, trained.css, model_aicc);
    END IF;

    /* 
      The best model (lowest AICc) is the "current model" and variations on it
      are considered:
      * vary p and/or q by +/- 1
      * include/exclude c from current model
      Repeat until no lower AICc can be found
    */
    SELECT *
    INTO current_model
    FROM tmp_autoarima
    ORDER BY aicc
    LIMIT 1;

    WHILE current_model.aicc > best_aicc LOOP
        SELECT *
        INTO current_model
        FROM tmp_autoarima
        ORDER BY aicc
        LIMIT 1;

        -- Train p/q variations
        FOR to_train IN
            SELECT * FROM (VALUES 
                (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1)
            ) AS m(p, q)
        LOOP
            v_p := current_model.p + to_train.p;
            v_q := current_model.q + to_train.q;
            -- Skip if invalid or already trained
            CONTINUE WHEN v_p < 0 OR v_q < 0
            OR EXISTS (
                SELECT 1
                FROM tmp_autoarima t
                WHERE t.p = v_p AND t.d = v_d AND t.q = v_q
            );
            trained := arima_train(v_p, v_d, v_q, source_table, date_col, value_col, include_c);
            n_params := to_train.p + current_model.p + to_train.q + current_model.q;
            IF include_c THEN
                n_params := n_params + 1;
            END IF;
            model_aicc := aicc(trained.css, to_train.p, to_train.q, n_params, n_vals);
            INSERT INTO tmp_autoarima
            VALUES (
                v_p,
                v_d,
                v_q,
                trained.c,
                trained.phi,
                trained.theta,
                trained.css,
                model_aicc
            );
            IF model_aicc < best_aicc THEN
                best_aicc := model_aicc;
            END IF;
        END LOOP;

        -- Train with/without constant
        v_p := current_model.p;
        v_q := current_model.q;
        trained := arima_train(v_p, v_d, v_q, source_table, date_col, value_col, NOT include_c);
        n_params := current_model.p + to_train.q;
        IF include_c THEN
            n_params := n_params + 1;
        END IF;
        model_aicc := aicc(trained.css, to_train.p, to_train.q, n_params, n_vals);
        INSERT INTO tmp_autoarima
        VALUES (
            v_p,
            v_d,
            v_q,
            trained.c,
            trained.phi,
            trained.theta,
            trained.css,
            model_aicc
        );
        IF model_aicc < best_aicc THEN
            best_aicc := model_aicc;
        END IF;
    END LOOP;

    SELECT *
    INTO current_model
    FROM tmp_autoarima
    ORDER BY aicc
    LIMIT 1;

    RETURN current_model;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION autoarima_train_and_forecast(
    horizon INT,
    source_table TEXT, -- Table/view name
    date_col TEXT,     -- Timestamp column
    value_col TEXT     -- Numerical value column
)
RETURNS TABLE(date TIMESTAMP, forecast_value DOUBLE PRECISION) AS $$
DECLARE
    best_model autoarima_rec;
    include_mean BOOL := TRUE;
BEGIN
    best_model := autoarima_train(horizon, source_table, date_col, value_col);
    IF best_model.c = 0 THEN
        include_mean := FALSE;
    END IF;

    RETURN QUERY
    SELECT * FROM 
    arima_train_and_forecast(best_model.p, best_model.d, best_model.q,
                            horizon, source_table, date_col, value_col,
                            include_mean);
END;
$$ LANGUAGE plpgsql;