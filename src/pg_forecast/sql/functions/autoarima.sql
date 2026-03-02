/* -------------------------------------------------------------------------
 * Types
 * ------------------------------------------------------------------------- */

CREATE TYPE kpss_rec AS (
    kpss_val DOUBLE PRECISION,
    crit_val DOUBLE PRECISION,
    test_passed BOOLEAN
);

CREATE TYPE autoarima_rec AS (
    p INT,
    d INT,
    q INT,
    c DOUBLE PRECISION,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    css DOUBLE PRECISION,
    aicc DOUBLE PRECISION,
    use_log_transform BOOLEAN -- To pass to other functions
);

/* -------------------------------------------------------------------------
 * C Functions
 * ------------------------------------------------------------------------- */

CREATE FUNCTION kpss(
    vals DOUBLE PRECISION[], -- Ordered array of time-series values
    p_val DOUBLE PRECISION DEFAULT 0.05 -- (0.1, 0.05, 0.025, 0.01)
)
RETURNS kpss_rec
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

/* -------------------------------------------------------------------------
 * PL/pgSQL Functions
 * ------------------------------------------------------------------------- */

-- Helper function to calculate number of parameters
CREATE OR REPLACE FUNCTION autoarima_param_count(p INT, q INT, include_c BOOLEAN)
RETURNS INT AS $$
BEGIN
    RETURN p + q + CASE WHEN include_c THEN 1 ELSE 0 END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION autoarima_train(
    source_table TEXT, -- Table/view name
    date_col TEXT,     -- Timestamp column
    value_col TEXT     -- Numerical value column
)
RETURNS autoarima_rec
SECURITY DEFINER
AS $$
DECLARE
    AUTO_LOG_CV_THRESHOLD DOUBLE PRECISION := 0.3;
    MAX_DIFF CONSTANT INT := 2;
    MAX_KPSS_TRIES CONSTANT INT := 3;

    v_min_val DOUBLE PRECISION;
    v_std_dev DOUBLE PRECISION;
    v_mean DOUBLE PRECISION;
    n_vals INT;
    v_cv DOUBLE PRECISION;
    v_is_log BOOLEAN := FALSE;
    v_log_tmp_col TEXT;

    arr_vals DOUBLE PRECISION[];
    arr_diff_vals DOUBLE PRECISION[];
    v_p INT;
    v_d INT := 0;
    v_q INT;
    kpss_res kpss_rec;
    include_c BOOLEAN := TRUE;
    rec_candidate RECORD;
    trained arima_optimise_result;
    n_params INT;
    model_aicc DOUBLE PRECISION;
    rec_current autoarima_rec;
    best_aicc DOUBLE PRECISION := 'Infinity';
    v_model_id BIGINT;
    v_arima_id BIGINT;
    v_incremental_state css_incremental_state;
BEGIN
    -- Safety precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);

    -- Log transform if appropriate
    EXECUTE FORMAT(
        'SELECT MIN(%I) AS min_val, STDDEV_POP(%I) AS std_dev, AVG(%I) AS mean, COUNT(1) AS n
        FROM %I',
        value_col, value_col, value_col, source_table
    ) INTO v_min_val, v_std_dev, v_mean, n_vals;

    v_cv := COALESCE(v_std_dev / NULLIF(v_mean, 0.0), 0.0);
    IF v_min_val >= 0 AND v_cv > AUTO_LOG_CV_THRESHOLD THEN
        RAISE DEBUG 'Automatic log transform enabled';
        v_is_log := TRUE;
    END IF;

    -- Aggregate the series into an array, dropping NULLs
    arr_vals := series_to_array(source_table, date_col, value_col, TRUE, v_is_log);

    -- Determine the number of differences with repeated KPSS tests
    arr_diff_vals := arr_vals;
    FOR i IN 1..MAX_KPSS_TRIES LOOP
        kpss_res := kpss(arr_diff_vals);
        EXIT WHEN kpss_res.test_passed;

        v_d := i;
        arr_diff_vals := arima_difference(arr_vals, v_d);
    END LOOP;

    IF v_d > MAX_DIFF THEN
        RAISE WARNING 'More than 2 differences required, ARIMA model not suitable';
        v_d = 2;
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

    FOR rec_candidate IN
        SELECT * FROM (VALUES 
            (0, 0), (2, 2), (1, 0), (0, 1)
        ) AS m(p, q)
    LOOP
        trained := arima_train(rec_candidate.p, v_d, rec_candidate.q, source_table, date_col, value_col, include_c, TRUE, v_is_log);
        n_params := autoarima_param_count(rec_candidate.p, rec_candidate.q, include_c);
        model_aicc := aicc(trained.css, rec_candidate.p, rec_candidate.q, n_params, n_vals);

        INSERT INTO tmp_autoarima (
            p, d, q, c, phi, theta, css, aicc, use_log_transform
        ) VALUES (
            rec_candidate.p,
            v_d,
            rec_candidate.q,
            trained.c,
            trained.phi,
            trained.theta,
            trained.css,
            model_aicc,
            v_is_log
        );
    END LOOP;

    -- If d <= 1, ARIMA(0, d, 0) without a constant is also fitted
    IF v_d <= 1 THEN
        trained := arima_train(0, v_d, 0, source_table, date_col, value_col, FALSE, TRUE, v_is_log);
        n_params := 0;
        model_aicc := aicc(trained.css, 0, 0, n_params, n_vals);

        INSERT INTO tmp_autoarima (
            p, d, q, c, phi, theta, css, aicc, use_log_transform
        ) VALUES (
            0, v_d, 0, 0, trained.phi, trained.theta, trained.css, model_aicc, v_is_log
        );
    END IF;

    /*
      The best model (lowest AICc) is the "current model" and variations on it
      are considered:
      * vary p and/or q by +/- 1
      * include/exclude c from current model
      Repeat until no lower AICc can be found
    */
    SELECT *
    INTO rec_current
    FROM tmp_autoarima
    ORDER BY aicc
    LIMIT 1;

    best_aicc := rec_current.aicc;

    LOOP
        -- Train p/q variations
        FOR rec_candidate IN
            SELECT * FROM (VALUES 
                (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1)
            ) AS m(p, q)
        LOOP
            v_p := rec_current.p + rec_candidate.p;
            v_q := rec_current.q + rec_candidate.q;

            -- Skip if invalid or already trained
            CONTINUE WHEN v_p < 0 OR v_q < 0
            OR EXISTS (
                SELECT 1
                FROM tmp_autoarima t
                WHERE t.p = v_p AND t.d = v_d AND t.q = v_q
            );

            RAISE DEBUG 'AutoARIMA: Training ARIMA(%, %, %)', v_p, v_d, v_q;
            trained := arima_train(v_p, v_d, v_q, source_table, date_col, value_col, include_c, TRUE, v_is_log);
            n_params := autoarima_param_count(v_p, v_q, include_c);
            model_aicc := aicc(trained.css, v_p, v_q, n_params, n_vals);
            RAISE DEBUG 'AutoARIMA: Trained ARIMA(%, %, %) with CSS % and AICc %', v_p, v_d, v_q, trained.css, model_aicc;

            INSERT INTO tmp_autoarima (
                p, d, q, c, phi, theta, css, aicc, use_log_transform
            ) VALUES (
                v_p, v_d, v_q, trained.c, trained.phi, trained.theta, trained.css, model_aicc, v_is_log
            );

            IF model_aicc < best_aicc THEN
                best_aicc := model_aicc;
                RAISE DEBUG 'AutoARIMA: New best AICc found, ARIMA(%, %, %)', v_p, v_d, v_q;
            END IF;
        END LOOP;

        -- Train with/without constant
        v_p := rec_current.p;
        v_q := rec_current.q;

        RAISE DEBUG 'AutoARIMA: Training ARIMA(%, %, %)', v_p, v_d, v_q;
        trained := arima_train(v_p, v_d, v_q, source_table, date_col, value_col, NOT include_c, TRUE, v_is_log);
        n_params := autoarima_param_count(v_p, v_q, NOT include_c);
        model_aicc := aicc(trained.css, v_p, v_q, n_params, n_vals);
        RAISE DEBUG 'AutoARIMA: Trained ARIMA(%, %, %) with phi = %, theta = %', v_p, v_d, v_q, trained.phi, trained.theta;

        INSERT INTO tmp_autoarima (
            p, d, q, c, phi, theta, css, aicc, use_log_transform
        ) VALUES (
            v_p, v_d, v_q, trained.c, trained.phi, trained.theta, trained.css, model_aicc, v_is_log
        );

        SELECT *
        INTO rec_current
        FROM tmp_autoarima
        ORDER BY aicc
        LIMIT 1;

        IF model_aicc < best_aicc THEN
            best_aicc := model_aicc;
        ELSE
            EXIT; -- No improvement, terminate loop
        END IF;
    END LOOP;

    -- Create entry in models & stats tables to incrementally update CSS
    INSERT INTO models("model_type", "input_table", "date_column", "value_column", "use_log_transform")
    VALUES (
        'autoarima'::model,
        source_table,
        date_col,
        value_col,
        v_is_log
    )
    ON CONFLICT (model_type, input_table, date_column, value_column)
    DO UPDATE SET model_type = models.model_type  -- No op
    RETURNING id
    INTO v_model_id;

    -- Get starting incremental state (from every value in the table)
    RAISE DEBUG 'Best model: ARIMA(%, %, %) with phi = %, theta = %, CSS = %', rec_current.p, rec_current.d, rec_current.q, rec_current.phi, rec_current.theta, rec_current.css;
    v_incremental_state := css_incremental_full_table(source_table, value_col, date_col, rec_current.phi, rec_current.theta, rec_current.c, rec_current.d, v_is_log);

    -- Deactivate any existing models
    UPDATE model_arima_stats
    SET is_active = FALSE
    WHERE model_id = v_model_id AND is_active = TRUE;
    
    -- Upsert
    INSERT INTO model_arima_stats(model_id, phi, theta, d, c, is_active, incremental_state)
    VALUES (v_model_id, rec_current.phi, rec_current.theta, rec_current.d, rec_current.c, TRUE, v_incremental_state)
    ON CONFLICT (model_id, phi, theta, d, c)
    DO UPDATE SET incremental_state = v_incremental_state, is_active = TRUE
    RETURNING id
    INTO v_arima_id;

    -- Generate simplex vertices for bounds checking
    PERFORM arima_create_simplex_vertices(v_arima_id);

    -- Register trigger on table
    EXECUTE format('
        CREATE OR REPLACE TRIGGER autoarima_on_insert_%I_%I
        AFTER INSERT ON %I
        REFERENCING NEW TABLE AS new
        FOR EACH STATEMENT
        EXECUTE FUNCTION trg_autoarima_on_insert();', 
        source_table, value_col, source_table
    );

    RAISE NOTICE 'AutoARIMA: Best model found ARIMA(%,%,%) with CSS % and AICc %',
    rec_current.p, rec_current.d, rec_current.q, rec_current.css, rec_current.aicc;

    RETURN rec_current;
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
    best_model := autoarima_train(source_table, date_col, value_col);
    IF best_model.c = 0 THEN
        include_mean := FALSE;
    END IF;

    RETURN QUERY
    SELECT * FROM 
    arima_train_and_forecast(best_model.p, best_model.d, best_model.q,
                            horizon, source_table, date_col, value_col,
                            include_mean, best_model.use_log_transform);
END;
$$ LANGUAGE plpgsql;