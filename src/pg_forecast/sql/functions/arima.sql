/**
 * Calculates the Conditional Sum of Squares (CSS) for an ARIMA(p,0,q) model
 * using a recursive CTE, based on the formula from Rosenthal & Lehner 2011.
 *
 * This function is intended for demonstration or for use with small data sets.
 *
 * @param source_table The properly-escaped name of the source data table.
 * @param date_col The name of the timestamp column (used for ordering).
 * @param value_col The name of the time series value column.
 * @param p The AR order
 * @param q The MA order
 * @param phi_params The array of AR parameters (phi_1, phi_2, ... phi_p).
 * @param theta_params The array of MA parameters (theta_1, theta_2, ... theta_q).
 * @returns The total Conditional Sum of Squares (CSS).
 */
CREATE OR REPLACE FUNCTION css_loss_sql(
    source_table TEXT,
    date_col TEXT,
    value_col TEXT,
    p INT,
    q INT,
    phi_params DOUBLE PRECISION[],
    theta_params DOUBLE PRECISION[]
)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    ar_sum_sql TEXT := '0.0';
    ma_sum_sql TEXT := '0.0';
    lag_cols_sql TEXT := '';
    j INT;
    k INT;
    css_query TEXT;
    result_css DOUBLE PRECISION;
    ncond INT;
BEGIN
    ncond := GREATEST(p, q);

    -- Build AR lag expressions
    FOR j IN 1..p LOOP
        lag_cols_sql := lag_cols_sql || format(
            ', LAG(%I, %s) OVER (ORDER BY %I) AS x_lag_%s',
            value_col, j, date_col, j
        );
        ar_sum_sql := ar_sum_sql || format(' + COALESCE(d.x_lag_%s * %L, 0.0)', j, phi_params[j]);
    END LOOP;

    -- Build MA sum respecting min(i - ncond, q)
    ma_sum_sql := '0.0';
    FOR k IN 1..q LOOP
        ma_sum_sql := ma_sum_sql || format(
            ' + CASE WHEN r.i - %s >= %s THEN COALESCE(r.past_errors[%s] * %L, 0.0) ELSE 0.0 END',
            ncond, k, k, theta_params[k]
        );
    END LOOP;

    -- Construct recursive CSS query starting at i = ncond
    css_query := format(
        $QUERY$
        WITH RECURSIVE
        ordered_data AS (
            SELECT
                %I AS t,
                %I AS x_i,
                ROW_NUMBER() OVER (ORDER BY %I) AS i
                %s -- date_col, value_col, date_col, lag_cols_sql
            FROM %s
        ),

        -- Recursively calculate errors (a_i)
        -- a_i = x_i - (AR part) - (MA part) 
        error_calculator(i, x_i, ar_part, ma_part, a_i, past_errors) AS (
            -- Base Case: i = 1, no errors
            SELECT
                d.i,
                d.x_i,
                0.0::double precision AS a_i,
                ARRAY_FILL(0.0::double precision, ARRAY[%s]) AS past_errors
            FROM ordered_data d
            WHERE d.i = %s

            UNION ALL

            -- Recursive step: i > ncond
            SELECT
                d.i,
                d.x_i,
                (d.x_i - (%s) - (%s)) AS a_i,
                ARRAY[(d.x_i - (%s) - (%s))] || r.past_errors[1:%s] AS past_errors
            FROM error_calculator r
            JOIN ordered_data d ON d.i = r.i + 1
        )
        SELECT sum(a_i * a_i)
        FROM error_calculator;
        $QUERY$,
        date_col, value_col, date_col, lag_cols_sql, source_table, -- ordered_data
        q, ncond,                        -- base case past_errors length, start at i = ncond
        ar_sum_sql, ma_sum_sql,          -- residual a_i
        ar_sum_sql, ma_sum_sql,          -- past_errors array
        q                                -- slice past_errors
    );

    RAISE DEBUG 'Generated CSS Query: %', css_query;
    EXECUTE css_query INTO result_css;
    RETURN result_css;
END;
$$ LANGUAGE plpgsql;


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

/*
SELECT
    css_loss(
        vals := array_agg(value ORDER BY t)::double precision[],
        phi := ARRAY[0.5]::double precision[],
        theta := ARRAY[0.3]::double precision[],
        p := 1,
        q := 1
    )
FROM
    time_series_data
WHERE
    series_id = 'TestSeries';
*/

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
    residuals DOUBLE PRECISION[]
);

CREATE FUNCTION arima_optimise(
    vals DOUBLE PRECISION[],
    p INT,
    q INT,
    include_c BOOLEAN DEFAULT FALSE,
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

CREATE OR REPLACE FUNCTION arima(
    p INT, -- Number of lagged y_t
    d INT, -- Number of times to difference
    q INT, -- Number of lagged residuals
    horizon INT,
    source_table TEXT, -- Table/view name
    date_col TEXT,     -- Timestamp column
    value_col TEXT,    -- Numerical value column
    include_mean BOOLEAN DEFAULT TRUE
)
RETURNS TABLE(date TIMESTAMP, forecast_value DOUBLE PRECISION) AS $$
DECLARE
    ncond INT;
    vals DOUBLE PRECISION[];
    opt_result arima_optimise_result;
    initial_vals DOUBLE PRECISION[];
    last_idx INT;
    n_vals INT;
    last_vals DOUBLE PRECISION[];
    forecasts DOUBLE PRECISION[];
    last_date TIMESTAMP;
    i INT;
BEGIN
    ncond := GREATEST(p, q);

    -- Aggregate the series into an array
    EXECUTE format(
        'SELECT array_agg(%I ORDER BY %I) FROM %I',
        value_col,
        date_col,
        source_table
    )
    INTO vals;

    IF vals IS NULL OR array_length(vals, 1) = 0 THEN
        RAISE EXCEPTION
            'ARIMA: no data found in % for columns %, %',
            source_table, date_col, value_col;
    END IF;

    IF d > 0 THEN
        initial_vals := vals[1:d];
        vals := arima_difference(vals, d);
    END IF;

    -- Fit ARIMA model
    opt_result := arima_optimise(vals, p, q, include_mean, 'Nelder-Mead');
    RAISE DEBUG 'ARIMA model optimised with phi = %, theta = %, c = %', opt_result.phi, opt_result.theta, opt_result.c;

    -- Determine number of values/residuals needed for forecast
    n_vals := array_length(vals, 1);
    last_idx := n_vals - ncond + 1;
    last_vals := vals[last_idx : n_vals];

    -- Generate forecasts
    forecasts := arima_forecast(last_vals, opt_result.residuals, p, q, opt_result.c, opt_result.phi, opt_result.theta, horizon);
    RAISE DEBUG 'ARIMA forecasted with last_vals: %, residuals: %, forecast: %', last_vals, opt_result.residuals, forecasts;

    -- Get the last timestamp to build forecast dates
    EXECUTE format(
        'SELECT MAX(%I) FROM %I',
        date_col,
        source_table
    )
    INTO last_date;

    IF d > 0 THEN
        forecasts := arima_integrate(vals || forecasts, d, initial_vals); -- add in original vals for integration
        forecasts := forecasts[n_vals+d+1 : n_vals+d+horizon]; -- trim back to forecasted values
    END IF;

    -- Return table of dates and forecast values
    FOR i IN 1..horizon LOOP
        date := last_date + (i * interval '1 day');  -- adjust interval if your data is not daily
        forecast_value := forecasts[i];
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;