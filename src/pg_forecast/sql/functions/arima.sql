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