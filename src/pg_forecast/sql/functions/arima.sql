CREATE FUNCTION css_loss(
    vals DOUBLE PRECISION[],
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    p INT,
    q INT
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

CREATE TYPE arima_optimise_result AS (
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    c DOUBLE PRECISION, -- Intercept (0 if not including mean)
    residuals DOUBLE PRECISION[],
    css DOUBLE PRECISION
);

CREATE FUNCTION arima_optimise(
    vals DOUBLE PRECISION[],
    p INT,
    q INT,
    include_c BOOLEAN DEFAULT TRUE,
    method TEXT DEFAULT 'L-BFGS' -- 'Nelder-Mead', 'L-BFGS'
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

CREATE OR REPLACE FUNCTION arima_train(
    p INT, -- Number of lagged y_t
    d INT, -- Number of times to difference
    q INT, -- Number of lagged residuals
    source_table TEXT, -- Table/view name
    date_col TEXT,     -- Timestamp column
    value_col TEXT,    -- Numerical value column
    include_mean BOOLEAN DEFAULT TRUE,
    optimiser TEXT DEFAULT 'L-BFGS'
)
RETURNS arima_optimise_result AS $$
DECLARE
    v_ncond INT;
    arr_vals DOUBLE PRECISION[];
    opt_result arima_optimise_result;
BEGIN
    v_ncond := GREATEST(p, q);

    -- Aggregate the series into an array
    EXECUTE format(
        'SELECT array_agg(%I ORDER BY %I) FROM %I',
        value_col,
        date_col,
        source_table
    )
    INTO arr_vals;

    IF arr_vals IS NULL OR array_length(arr_vals, 1) = 0 THEN
        RAISE EXCEPTION
            'ARIMA: no data found in % for columns %, %',
            source_table, date_col, value_col;
    END IF;

    IF d > 0 THEN
        arr_vals := arima_difference(arr_vals, d);
    END IF;

    -- Fit ARIMA model
    opt_result := arima_optimise(arr_vals, p, q, include_mean, optimiser);

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
    optimiser TEXT DEFAULT 'L-BFGS'
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
    v_i INT;
BEGIN
    v_ncond := GREATEST(p, q);

    -- Aggregate the series into an array
    EXECUTE format(
        'SELECT array_agg(%I ORDER BY %I) FROM %I',
        value_col,
        date_col,
        source_table
    )
    INTO arr_vals;

    IF arr_vals IS NULL OR array_length(arr_vals, 1) = 0 THEN
        RAISE EXCEPTION
            'ARIMA: no data found in % for columns %, %',
            source_table, date_col, value_col;
    END IF;

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
    FOR v_i IN 1..horizon LOOP
        date := last_date + (v_i * interval '1 day');  -- adjust interval if your data is not daily
        forecast_value := arr_forecasts[v_i];
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Incremental update logic
CREATE TYPE css_incremental_state AS (
    t INT,
    p INT,
    q INT,
    y_lags DOUBLE PRECISION[],
    e_lags DOUBLE PRECISION[],
    css DOUBLE PRECISION
);

CREATE FUNCTION css_incremental_transition(
    state css_incremental_state,
    y DOUBLE PRECISION,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[]
)
RETURNS css_incremental_state
AS 'MODULE_PATHNAME', 'css_incremental_transition'
LANGUAGE C IMMUTABLE;

CREATE FUNCTION _css_incremental_transition(
    state INTERNAL,
    y DOUBLE PRECISION,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[]
)
RETURNS INTERNAL
AS 'MODULE_PATHNAME', '_css_incremental_transition'
LANGUAGE C IMMUTABLE;

CREATE FUNCTION _css_incremental_final(state INTERNAL)
RETURNS css_incremental_state
AS 'MODULE_PATHNAME', '_css_incremental_final'
LANGUAGE C IMMUTABLE;

CREATE AGGREGATE css_incremental(
    y DOUBLE PRECISION,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[]
) (
    SFUNC = _css_incremental_transition,
    STYPE = INTERNAL, -- Initialisation handled by C
    FINALFUNC = _css_incremental_final
);

CREATE TABLE model_arima_stats(
    id BIGSERIAL PRIMARY KEY,
    model_id INT NOT NULL REFERENCES models(id),
    phi DOUBLE PRECISION[] NOT NULL,
    theta DOUBLE PRECISION[] NOT NULL,
    d INT NOT NULL,
    is_active BOOLEAN NOT NULL,
    incremental_state css_incremental_state NOT NULL,
    UNIQUE (model_id, phi, theta)
);

CREATE OR REPLACE FUNCTION css_incremental_full_table(
    source_table TEXT,
    value_col TEXT,
    date_col TEXT,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[]
)
RETURNS css_incremental_state AS $$
DECLARE
    rec_state RECORD;
BEGIN
    EXECUTE format(
        'SELECT (css_incremental(%I, %L, %L) OVER (ORDER BY %I)) AS s
         FROM %I
         GROUP BY %I
         ORDER BY %I DESC
         LIMIT 1',
        value_col,
        phi,
        theta,
        date_col,
        source_table,
        date_col,
        date_col
    ) INTO rec_state;
    RETURN rec_state.s;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE arima_vertices (
    arima_id INT NOT NULL REFERENCES model_arima_stats(id),
    vertex_id INT NOT NULL,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    incremental_state css_incremental_state NOT NULL,
    PRIMARY KEY (arima_id, vertex_id)
);

-- Create Simplex vertices for incremental bound checking
CREATE OR REPLACE FUNCTION arima_create_simplex_vertices(
    centre_id INT, -- ID of model to target (centre vector) in model_arima_stats
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

    -- Clear old vertices
    DELETE FROM arima_vertices
    WHERE arima_id = centre_id;

    IF rec_centre IS NULL THEN
        RAISE EXCEPTION 'Could not create vertices for ARIMA ID %', centre_id;
        RETURN;
    END IF;

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

        v_state_new := css_incremental_full_table(rec_centre.input_table, rec_centre.value_column, rec_centre.date_column, v_phi_new, v_theta_new);

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

    SELECT array_agg(val * v_scale_factor) INTO v_phi_new 
    FROM unnest(v_phi) val;
    
    SELECT array_agg(val * v_scale_factor) INTO v_theta_new 
    FROM unnest(v_theta) val;

    v_state_new := css_incremental_full_table(rec_centre.input_table, rec_centre.value_column, rec_centre.date_column, v_phi_new, v_theta_new);

    INSERT INTO arima_vertices(arima_id, vertex_id, phi, theta, incremental_state)
    VALUES (centre_id, v_d+1, v_phi_new, v_theta_new, v_state_new);
END;
$$ LANGUAGE plpgsql;