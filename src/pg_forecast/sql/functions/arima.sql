/* -------------------------------------------------------------------------
 * Types
 * ------------------------------------------------------------------------- */

CREATE TYPE arima_optimise_result AS (
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    c DOUBLE PRECISION, -- Intercept (0 if not including mean)
    residuals DOUBLE PRECISION[],
    css DOUBLE PRECISION
);

CREATE TYPE css_incremental_state AS (
    t INT,
    p INT,
    q INT,
    y_lags DOUBLE PRECISION[],
    e_lags DOUBLE PRECISION[],
    css DOUBLE PRECISION,

    d INT,
    n_diff INT,
    diff_buf DOUBLE PRECISION[]
);

/* -------------------------------------------------------------------------
 * Tables
 * ------------------------------------------------------------------------- */

CREATE TABLE model_arima_stats(
    id BIGSERIAL PRIMARY KEY,
    model_id INT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    phi DOUBLE PRECISION[] NOT NULL,
    theta DOUBLE PRECISION[] NOT NULL,
    d INT NOT NULL,
    c DOUBLE PRECISION DEFAULT 0.0,
    is_active BOOLEAN NOT NULL,
    incremental_state css_incremental_state NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_id, phi, theta, d, c)
);

CREATE UNIQUE INDEX idx_active_model
ON model_arima_stats (model_id)
WHERE is_active = TRUE;

CREATE TABLE arima_vertices (
    arima_id BIGINT NOT NULL REFERENCES model_arima_stats(id) ON DELETE CASCADE,
    vertex_id INT NOT NULL,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    incremental_state css_incremental_state NOT NULL,
    PRIMARY KEY (arima_id, vertex_id)
);

/* -------------------------------------------------------------------------
 * C Functions
 * ------------------------------------------------------------------------- */

CREATE FUNCTION css_loss(
    vals DOUBLE PRECISION[],
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    p INT,
    q INT,
    c DOUBLE PRECISION DEFAULT 0.0 -- Constant term, leave as 0 if not required
)
RETURNS DOUBLE PRECISION
AS 'MODULE_PATHNAME', 'css_loss'
LANGUAGE C STRICT VOLATILE;

CREATE FUNCTION arima_difference(
    vals DOUBLE PRECISION[],
    d INT
)
RETURNS DOUBLE PRECISION[]
AS 'MODULE_PATHNAME', 'arima_difference'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION arima_integrate(
    differences DOUBLE PRECISION[],
    d INT,
    initial_vals DOUBLE PRECISION[]
)
RETURNS DOUBLE PRECISION[]
AS 'MODULE_PATHNAME', 'arima_integrate'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION arima_optimise(
    vals DOUBLE PRECISION[],
    p INT,
    q INT,
    include_c BOOLEAN DEFAULT TRUE,
    method TEXT DEFAULT 'Nelder-Mead' -- 'Nelder-Mead', 'L-BFGS'
)
RETURNS arima_optimise_result
AS 'MODULE_PATHNAME', 'arima_optimise'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION arima_forecast(
    last_vals DOUBLE PRECISION[],      -- Last max(p, q) values
    last_residuals DOUBLE PRECISION[], -- Last max(p, q) residuals
    p INT,
    q INT,
    c DOUBLE PRECISION, -- Intercept (0 if not including mean)
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    horizon INT
)
RETURNS DOUBLE PRECISION[]
AS 'MODULE_PATHNAME', 'arima_forecast'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION css_incremental_transition(
    state css_incremental_state,
    y DOUBLE PRECISION,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    c DOUBLE PRECISION DEFAULT 0.0, -- 0 if no constant term required
    d INT DEFAULT 0
)
RETURNS css_incremental_state
AS 'MODULE_PATHNAME', 'css_incremental_transition'
LANGUAGE C IMMUTABLE;

CREATE FUNCTION _css_incremental_transition(
    state INTERNAL,
    y DOUBLE PRECISION,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    c DOUBLE PRECISION DEFAULT 0.0, -- 0 if no constant term required
    d INT DEFAULT 0
)
RETURNS INTERNAL
AS 'MODULE_PATHNAME', '_css_incremental_transition'
LANGUAGE C IMMUTABLE;

CREATE FUNCTION _css_incremental_final(state INTERNAL)
RETURNS css_incremental_state
AS 'MODULE_PATHNAME', '_css_incremental_final'
LANGUAGE C IMMUTABLE;

/* -------------------------------------------------------------------------
 * Aggregates
 * ------------------------------------------------------------------------- */

CREATE AGGREGATE css_incremental(
    y DOUBLE PRECISION,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    c DOUBLE PRECISION,
    d INT
) (
    SFUNC = _css_incremental_transition,
    STYPE = INTERNAL, -- Initialisation handled by C
    FINALFUNC = _css_incremental_final
);

/* -------------------------------------------------------------------------
 * PL/pgSQL Functions
 * ------------------------------------------------------------------------- */

CREATE OR REPLACE FUNCTION arima_train(
    p INT, -- Number of lagged y_t
    d INT, -- Number of times to difference
    q INT, -- Number of lagged residuals
    source_table TEXT, -- Table/view name
    date_col TEXT,     -- Timestamp column
    value_col TEXT,    -- Numerical value column
    include_mean BOOLEAN DEFAULT TRUE,
    silent_error BOOLEAN DEFAULT FALSE,
    optimiser TEXT DEFAULT 'Nelder-Mead'
)
RETURNS arima_optimise_result AS $$
DECLARE
    v_ncond INT;
    arr_vals DOUBLE PRECISION[];
    opt_result arima_optimise_result;
BEGIN
    v_ncond := GREATEST(p, q);

    -- Aggregate the series into an array, dropping NULLs
    arr_vals := series_to_array(source_table, date_col, value_col, TRUE);

    IF d > 0 THEN
        arr_vals := arima_difference(arr_vals, d);
    END IF;

    -- Fit ARIMA model
    BEGIN
        opt_result := arima_optimise(arr_vals, p, q, include_mean, optimiser);
    EXCEPTION
        WHEN OTHERS THEN
        IF silent_error THEN
            RAISE DEBUG 'arima_train: Failed to train ARMA(%, %) due to error: %',
                p, q, SQLERRM;
            opt_result := ('{}', '{}', 0.0, '{}', 'Infinity'::double precision)::arima_optimise_result;
        ELSE
            RAISE EXCEPTION 'arima_train: Failed to train ARMA(%, %) due to error: %', p, q, SQLERRM;
        END IF;
    END;

    RETURN opt_result;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION arima_train_and_forecast(
    p INT, -- Number of lagged y_t
    d INT, -- Number of times to difference
    q INT, -- Number of lagged residuals
    horizon INT,
    source_table TEXT, -- Table/view name
    date_col TEXT,     -- Timestamp column
    value_col TEXT,    -- Numerical value column
    include_mean BOOLEAN DEFAULT TRUE,
    optimiser TEXT DEFAULT 'Nelder-Mead',
    forecast_step INTERVAL DEFAULT '1 day'
)
RETURNS TABLE(date TIMESTAMP, forecast_value DOUBLE PRECISION) AS $$
DECLARE
    v_ncond INT;
    arr_vals DOUBLE PRECISION[];
    opt_result arima_optimise_result;
    arr_initial_vals DOUBLE PRECISION[];
    v_last_idx INT;
    v_n_vals INT;
    arr_last_vals DOUBLE PRECISION[];
    arr_forecasts DOUBLE PRECISION[];
    last_date TIMESTAMP;
BEGIN
    v_ncond := GREATEST(p, q);

    -- Aggregate the series into an array, dropping NULLs
    arr_vals := series_to_array(source_table, date_col, value_col, TRUE);

    IF d > 0 THEN
        arr_initial_vals := arr_vals[1:d];
        arr_vals := arima_difference(arr_vals, d);
    END IF;

    -- Fit ARIMA model
    opt_result := arima_optimise(arr_vals, p, q, include_mean, optimiser);
    RAISE DEBUG 'ARIMA model optimised with phi = %, theta = %, c = %', opt_result.phi, opt_result.theta, opt_result.c;

    -- Warn for risky bounds on phi/theta coefficients
    FOR v_i IN 1..p LOOP
        IF opt_result.phi[v_i] <= -1.0 OR opt_result.phi[v_i] >= 1.0 THEN
            RAISE WARNING 'AR model not invertible - phi[%] = %', v_i, opt_result.phi[v_i];
        END IF;
    END LOOP;
    FOR v_i IN 1..q LOOP
        IF opt_result.theta[v_i] <= -1.0 OR opt_result.theta[v_i] >= 1.0 THEN
            RAISE WARNING 'AR model not invertible - theta[%] = %', v_i, opt_result.theta[v_i];
        END IF;
    END LOOP;

    -- Determine number of values/residuals needed for forecast
    v_n_vals := array_length(arr_vals, 1);
    v_last_idx := v_n_vals - v_ncond + 1;
    arr_last_vals := arr_vals[v_last_idx : v_n_vals];

    -- Generate forecasts
    arr_forecasts := arima_forecast(arr_last_vals, opt_result.residuals, p, q, opt_result.c, opt_result.phi, opt_result.theta, horizon);
    RAISE DEBUG 'ARIMA forecasted with last_vals: %, residuals: %, forecast: %', arr_last_vals, opt_result.residuals, arr_forecasts;

    -- Get the last timestamp to build forecast dates
    EXECUTE format(
        'SELECT MAX(%I) FROM %I',
        date_col,
        source_table
    )
    INTO last_date;

    IF d > 0 THEN
        arr_forecasts := arima_integrate(arr_vals || arr_forecasts, d, arr_initial_vals);
        arr_forecasts := arr_forecasts[v_n_vals + d + 1 : v_n_vals + d + horizon];
    END IF;

    -- Return table of dates and forecast values
    RETURN QUERY
        SELECT
            last_date + (i * forecast_step) AS forecast_date,
            arr_forecasts[i] AS forecast_value
        FROM generate_series(1, horizon) AS i;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION arima_run_forecast(
    p INT, -- Number of lagged y_t
    d INT, -- Number of times to difference
    q INT, -- Number of lagged residuals
    opt_result arima_optimise_result,
    source_table TEXT, -- Table/view name
    date_col TEXT,     -- Timestamp column
    value_col TEXT,    -- Numerical value column
    horizon INT,
    forecast_step INTERVAL
)
RETURNS TABLE(forecast_date TIMESTAMP, forecast_value DOUBLE PRECISION) AS $$
DECLARE
    v_ncond INT;
    arr_vals DOUBLE PRECISION[];
    arr_initial_vals DOUBLE PRECISION[];
    v_last_idx INT;
    v_n_vals INT;
    arr_last_vals DOUBLE PRECISION[];
    arr_forecasts DOUBLE PRECISION[];
    last_date TIMESTAMP;
BEGIN
    v_ncond := GREATEST(p, q);

    -- Aggregate the series into an array, dropping NULLs
    arr_vals := series_to_array(source_table, date_col, value_col, TRUE);

    IF d > 0 THEN
        arr_initial_vals := arr_vals[1:d];
        arr_vals := arima_difference(arr_vals, d);
    END IF;

    -- Determine number of values/residuals needed for forecast
    v_n_vals := array_length(arr_vals, 1);
    v_last_idx := v_n_vals - v_ncond + 1;
    arr_last_vals := arr_vals[v_last_idx : v_n_vals];

    -- Generate forecasts
    arr_forecasts := arima_forecast(arr_last_vals, opt_result.residuals, p, q, opt_result.c, opt_result.phi, opt_result.theta, horizon);
    RAISE DEBUG 'ARIMA forecasted with last_vals: %, residuals: %, forecast: %', arr_last_vals, opt_result.residuals, arr_forecasts;

    -- Get the last timestamp to build forecast dates
    EXECUTE format(
        'SELECT MAX(%I) FROM %I',
        date_col,
        source_table
    )
    INTO last_date;

    IF d > 0 THEN
        arr_forecasts := arima_integrate(arr_vals || arr_forecasts, d, arr_initial_vals);
        arr_forecasts := arr_forecasts[v_n_vals + d + 1 : v_n_vals + d + horizon];
    END IF;

    -- Return table of dates and forecast values
    RETURN QUERY
        SELECT
            last_date + (i * forecast_step) AS forecast_date,
            arr_forecasts[i] AS forecast_value
        FROM generate_series(1, horizon) AS i;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION arima_get_active_id(
    model_id BIGINT,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    d INT,
    c DOUBLE PRECISION DEFAULT 0.0
)
RETURNS BIGINT
SECURITY DEFINER
AS $$
DECLARE
    rec_result RECORD;
BEGIN
    -- Safety precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);

    EXECUTE format(
        'SELECT id
        FROM model_arima_stats
        WHERE model_id = %L AND phi = %L AND theta = %L AND d = %L AND c = %L AND is_active = TRUE
        LIMIT 1',
        
        model_id, phi, theta, d, c
    ) INTO rec_result;

    RETURN rec_result.id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION css_incremental_full_table(
    source_table TEXT,
    value_col TEXT,
    date_col TEXT,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    c DOUBLE PRECISION DEFAULT 0, -- 0 if no constant term required
    d INT DEFAULT 0
)
RETURNS css_incremental_state AS $$
DECLARE
    v_query TEXT;
    rec_state RECORD;
BEGIN
    EXECUTE format(
        'SELECT (css_incremental(%I, %L, %L, %L, %L) OVER (ORDER BY %I)) AS s
         FROM %I
         ORDER BY %I DESC
         LIMIT 1',

        value_col, phi, theta, c, d, date_col,
        source_table, date_col, date_col
    ) INTO rec_state;
    RAISE DEBUG 'Incremental CSS: %', (rec_state.s).css;
    RETURN rec_state.s;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION arima_has_vertices(
    arima_id BIGINT
)
RETURNS BOOLEAN
SECURITY DEFINER
AS $$
DECLARE
    rec_result RECORD;
BEGIN
    -- Safety precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);

    EXECUTE format(
        'SELECT 1 AS result
        FROM arima_vertices
        WHERE arima_id = %L
        LIMIT 1',
        
        arima_id
    ) INTO rec_result;

    RETURN rec_result.result IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION arima_create_simplex_vertices(
    centre_id BIGINT, -- ID of model to target (centre vector) in model_arima_stats
    tolerance DOUBLE PRECISION DEFAULT 0.05 -- 0.05 empirically found by Rosenthal & Lehner
)
RETURNS VOID
SECURITY DEFINER
AS $$
DECLARE
    rec_centre RECORD;
    rec_vertex RECORD;
    v_d INT;
    i INT;
    v_phi DOUBLE PRECISION[];
    v_theta DOUBLE PRECISION[];
    v_phi_new DOUBLE PRECISION[];
    v_theta_new DOUBLE PRECISION[];
    rec_state RECORD;
    v_state_new css_incremental_state;
    v_dist_from_origin DOUBLE PRECISION;
    v_scale_factor DOUBLE PRECISION;
BEGIN
    -- Safety precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);

    -- Find centre point
    SELECT
        m.input_table,
        m.value_column,
        m.date_column,
        s.*
    INTO rec_centre
    FROM model_arima_stats s
    INNER JOIN models m ON m.id = s.model_id
    WHERE s.id = centre_id AND s.is_active = TRUE;

    IF rec_centre IS NULL THEN
        RAISE EXCEPTION 'Could not create vertices for ARIMA ID %', centre_id;
        RETURN;
    END IF;

    -- Clear old vertices
    DELETE FROM arima_vertices
    WHERE arima_id = rec_centre.id;

    v_d := cardinality(rec_centre.phi) + cardinality(rec_centre.theta);
    v_phi = rec_centre.phi;
    v_theta := rec_centre.theta;

    -- Create the first d vertices (add +/- tolerance to all dimensions)
    FOR i IN 1..v_d LOOP
        v_phi_new := rec_centre.phi;
        v_theta_new := rec_centre.theta;

        IF i <= cardinality(rec_centre.phi) THEN
            v_phi_new[i] := v_phi[i] + (CASE WHEN v_phi[i] >= 0 THEN tolerance ELSE -tolerance END);
        ELSE
            v_theta_new[i - cardinality(v_phi)] := v_theta[i - cardinality(v_phi)] + 
                (CASE WHEN v_theta[i - cardinality(v_phi)] >= 0 THEN tolerance ELSE -tolerance END);
        END IF;

        v_state_new := css_incremental_full_table(rec_centre.input_table, rec_centre.value_column, rec_centre.date_column, v_phi_new, v_theta_new, rec_centre.c, rec_centre.d);

        INSERT INTO arima_vertices(arima_id, vertex_id, phi, theta, incremental_state)
        VALUES (centre_id, i, v_phi_new, v_theta_new, v_state_new);
    END LOOP;

    -- Add (d+1)th vertex
    -- Norm of the centre vector
    SELECT sqrt(SUM(val^2)) INTO v_dist_from_origin 
    FROM (SELECT unnest(v_phi || v_theta) as val) s;

    -- Scale toward origin so the distance from centre is tolerance
    -- new_vector = centre * (1 - tol/dist)
    IF v_dist_from_origin > tolerance THEN
        v_scale_factor := 1.0 - (tolerance / v_dist_from_origin);
    ELSE
        v_scale_factor := 0.5; -- Fallback for near-zero vectors
    END IF;

    SELECT COALESCE(array_agg(val * v_scale_factor), '{}') INTO v_phi_new 
    FROM unnest(v_phi) val;
    
    SELECT COALESCE(array_agg(val * v_scale_factor), '{}') INTO v_theta_new 
    FROM unnest(v_theta) val;

    v_state_new := css_incremental_full_table(rec_centre.input_table, rec_centre.value_column, rec_centre.date_column, v_phi_new, v_theta_new, rec_centre.c, rec_centre.d);

    INSERT INTO arima_vertices(arima_id, vertex_id, phi, theta, incremental_state)
    VALUES (centre_id, v_d+1, v_phi_new, v_theta_new, v_state_new);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION arima_updated_states(
    target_model_id BIGINT,
    new_ys DOUBLE PRECISION[],
    arima_id BIGINT,
    OUT vertex_id BIGINT,
    OUT new_state css_incremental_state
)
RETURNS SETOF RECORD
LANGUAGE SQL
AS $$
    -- TODO: Rewrite this as a loop in C
    WITH RECURSIVE vertices AS (
        SELECT NULL AS vertex_id, s.phi, s.theta, s.incremental_state AS state, s.c, s.d
        FROM model_arima_stats s
        WHERE s.id = arima_id

        UNION ALL

        SELECT v.vertex_id, v.phi, v.theta, v.incremental_state AS state, s.c, s.d
        FROM arima_vertices v
        INNER JOIN model_arima_stats s ON s.id = v.arima_id
        WHERE v.arima_id = arima_id
    ),
    ys AS (
        SELECT y, ord
        FROM unnest(new_ys) WITH ORDINALITY AS u(y, ord)
    ),
    recurse AS (
        SELECT
            base.vertex_id,
            y.ord,
            css_incremental_transition(base.state, y.y, base.phi, base.theta, base.c, base.d) AS state,
            base.phi,
            base.theta,
            base.c,
            base.d
        FROM vertices base
        INNER JOIN ys y ON y.ord = 1

        UNION ALL

        SELECT
            r.vertex_id,
            y.ord,
            css_incremental_transition(r.state, y.y, r.phi, r.theta, r.c, r.d) AS state,
            r.phi,
            r.theta,
            r.c,
            r.d
        FROM recurse r
        INNER JOIN ys y ON y.ord = r.ord + 1
    )
    SELECT
        vertex_id,
        state AS new_state
    FROM recurse
    WHERE ord = (SELECT max(ord) FROM ys);
$$;

CREATE OR REPLACE FUNCTION arima_update_model(
    target_model_id BIGINT,
    new_ys DOUBLE PRECISION[]
)
RETURNS VOID
SECURITY DEFINER
AS $$
DECLARE
    v_breach_detected BOOLEAN;
    rec_model RECORD;
    v_centre_css DOUBLE PRECISION;
    v_min_vertex_css DOUBLE PRECISION;
BEGIN
    -- Safety precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);

    SELECT
        m.*,
        a.id AS arima_id,
        (a.incremental_state).p AS p,
        (a.incremental_state).q AS q
    INTO rec_model 
    FROM models m 
    INNER JOIN model_arima_stats a ON m.id = a.model_id AND a.is_active = TRUE
    WHERE m.id = target_model_id;

    -- Create a temp table to hold the updated states so we don't calculate them twice
    -- and we don't mistakenly calculate t+1 ahead when querying again (further down)
    CREATE TEMP TABLE tmp_arima_states ON COMMIT DROP AS
    SELECT vertex_id, new_state
    FROM arima_updated_states(target_model_id, new_ys, rec_model.arima_id);

    SELECT (new_state).css INTO v_centre_css FROM tmp_arima_states WHERE vertex_id IS NULL;
    SELECT MIN((new_state).css) INTO v_min_vertex_css FROM tmp_arima_states WHERE vertex_id IS NOT NULL;

    -- Check for divergence (Infinity/NaN) or improvement
    IF v_centre_css = 'Infinity'::DOUBLE PRECISION OR v_centre_css = 'NaN'::DOUBLE PRECISION THEN
        v_breach_detected := TRUE;
        RAISE WARNING 'arima_update_model: Model diverged (CSS=%). Forcing retrain.', v_centre_css;
    ELSIF v_min_vertex_css < v_centre_css THEN
        v_breach_detected := TRUE;
    ELSE
        v_breach_detected := FALSE;
    END IF;

    -- If no simplex can be performed, retrain
    IF rec_model.p + rec_model.q = 0 THEN
        v_breach_detected := TRUE;
    END IF;

    IF v_breach_detected THEN
        -- Mark current model inactive and retrain
        UPDATE model_arima_stats
        SET is_active = FALSE
        WHERE id = rec_model.arima_id;

        DELETE FROM arima_vertices
        WHERE arima_id = rec_model.arima_id;

        RAISE NOTICE 'Retraining AutoARIMA model as loss dropped below threshold';
        PERFORM autoarima_train(rec_model.input_table, rec_model.date_column, rec_model.value_column);
    ELSE
        -- Update the stored states with the new values calculated in the temp table
        -- Centre vector
        UPDATE model_arima_stats m
        SET incremental_state = u.new_state
        FROM tmp_arima_states u
        WHERE m.model_id = target_model_id AND u.vertex_id IS NULL;

        -- Simplex vertices
        UPDATE arima_vertices v
        SET incremental_state = u.new_state
        FROM tmp_arima_states u
        WHERE v.arima_id = rec_model.arima_id AND v.vertex_id = u.vertex_id;
    END IF;

    DROP TABLE IF EXISTS tmp_arima_states;
END;
$$ LANGUAGE plpgsql;

/* -------------------------------------------------------------------------
 * Triggers
 * ------------------------------------------------------------------------- */

CREATE OR REPLACE FUNCTION trg_autoarima_on_insert()
RETURNS TRIGGER 
SECURITY DEFINER
AS $$
DECLARE
    rec_model RECORD;
    v_new_values DOUBLE PRECISION[];
BEGIN
    -- Safety precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);

    SELECT id, value_column FROM models
    INTO rec_model
    WHERE input_table = TG_TABLE_NAME AND model_type = 'autoarima';

    IF rec_model IS NULL THEN
        RAISE WARNING 'trg_autoarima_on_insert: Model not found for table %. This should not happen.', TG_TABLE_NAME;
        RETURN NEW;
    END IF;

    -- Compute updated states once for all rows in the table of NEW records
    EXECUTE format(
        'SELECT array_agg((n).%I::double precision) FROM new n',
        rec_model.value_column
    ) INTO v_new_values;
   
    IF v_new_values IS NULL OR array_length(v_new_values, 1) IS NULL THEN
        RETURN NULL;
    END IF;

    PERFORM arima_update_model(rec_model.id, v_new_values);

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;