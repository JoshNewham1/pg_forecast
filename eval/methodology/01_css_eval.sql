/*
 * C tests - create array
 */
SELECT
    css_loss(
        ARRAY_AGG(val),
        '{0.25, 0.5}',
        '{0.3, 0.5}',
        2,
        2
    )
FROM pg_forecast_css_eval;

/*
 * SQL tests - persist to table
 */

DROP TABLE IF EXISTS pg_forecast_css_eval;
CREATE UNLOGGED TABLE pg_forecast_css_eval(
    t TIMESTAMP PRIMARY KEY,
    val DOUBLE PRECISION
);
-- Run 1
TRUNCATE TABLE pg_forecast_css_eval;
SELECT setseed(0.42);
INSERT INTO pg_forecast_css_eval(t, val)
SELECT
    ('2026-01-01'::date + s.i) AS t,
    random() AS val
FROM generate_series(1, 1000) AS s(i);
SELECT css_loss_sql(
    'pg_forecast_css_eval',
    't',
    'val',
    2,
    2,
    '{0.25, 0.5}',
    '{0.3, 0.5}'
);
TRUNCATE TABLE pg_forecast_css_eval;
-- Run 2
INSERT INTO pg_forecast_css_eval(t, val)
SELECT
    ('2026-01-01'::date + s.i) AS t,
    random() AS val
FROM generate_series(1, 1000) AS s(i);
SELECT css_loss_sql(
    'pg_forecast_css_eval',
    't',
    'val',
    2,
    2,
    '{0.25, 0.5}',
    '{0.3, 0.5}'
);
TRUNCATE TABLE pg_forecast_css_eval;
-- Run 3
INSERT INTO pg_forecast_css_eval(t, val)
SELECT
    ('2026-01-01'::date + s.i) AS t,
    random() AS val
FROM generate_series(1, 1000) AS s(i);
SELECT css_loss_sql(
    'pg_forecast_css_eval',
    't',
    'val',
    2,
    2,
    '{0.25, 0.5}',
    '{0.3, 0.5}'
);
TRUNCATE TABLE pg_forecast_css_eval;

/* SQL to benchmark */
CREATE OR REPLACE FUNCTION css_loss_sql(
    source_table TEXT,
    date_col TEXT,
    value_col TEXT,
    p INT,
    q INT,
    phi_params DOUBLE PRECISION[],
    theta_params DOUBLE PRECISION[]
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
AS $$
DECLARE
    v_ncond INT := GREATEST(p, q);
    rec RECORD;
    css DOUBLE PRECISION := 0.0;
    a_i DOUBLE PRECISION;
    ma_sum DOUBLE PRECISION;
    err_buf DOUBLE PRECISION[];
    sql_query TEXT;
    ar_expr TEXT := '0.0';
    lag_cols TEXT := '';
    j INT;
BEGIN
    err_buf := ARRAY_FILL(0.0::double precision, ARRAY[q]);

    ------------------------------------------------------------
    -- Build AR expression and lag columns ONCE
    ------------------------------------------------------------
    FOR j IN 1..p LOOP
        lag_cols := lag_cols || format(
            ', LAG(%I,%s) OVER (ORDER BY %I) AS x_lag_%s',
            value_col, j, date_col, j
        );

        ar_expr := ar_expr || format(
            ' + COALESCE(x_lag_%s,0.0) * %s',
            j, phi_params[j]
        );
    END LOOP;

    ------------------------------------------------------------
    -- Query precomputes AR prediction (vectorized!)
    ------------------------------------------------------------
    sql_query := format(
        $Q$
        SELECT
            x_i,
            (%s) AS ar_part
        FROM (
            SELECT
                %I AS x_i,
                ROW_NUMBER() OVER (ORDER BY %I) AS i
                %s
            FROM %s
        ) s
        WHERE i >= %s
        ORDER BY i
        $Q$,
        ar_expr,
        value_col,
        date_col,
        lag_cols,
        source_table,
        v_ncond
    );

    ------------------------------------------------------------
    -- Single fast streaming pass
    ------------------------------------------------------------
    FOR rec IN EXECUTE sql_query LOOP

        -- MA component (sequential dependency)
        ma_sum := 0.0;
        FOR j IN 1..q LOOP
            ma_sum := ma_sum + err_buf[j] * theta_params[j];
        END LOOP;

        -- residual
        a_i := rec.x_i - rec.ar_part - ma_sum;

        css := css + a_i * a_i;

        -- rotate MA buffer
        IF q > 0 THEN
            err_buf := ARRAY[a_i] || err_buf[1:q-1];
        END IF;

    END LOOP;

    RETURN css;
END;
$$;