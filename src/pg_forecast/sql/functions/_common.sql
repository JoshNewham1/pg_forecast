-- Generic
CREATE OR REPLACE FUNCTION create_forecast(
    model_name TEXT,
    input_table TEXT,
    date_column TEXT,
    value_column TEXT,
    args JSONB DEFAULT '{}'
)
RETURNS BOOLEAN -- Was forecast creation successful
LANGUAGE plpgsql
AS $$
DECLARE
    v_result BOOLEAN;
BEGIN
    v_result :=
        CASE model_name
            WHEN 'autoarima' THEN
                (SELECT autoarima_train(
                    input_table,
                    date_column,
                    value_column
                )) IS NOT NULL
            ELSE
                NULL
        END;

    IF v_result IS NULL THEN
        RAISE EXCEPTION '% not supported, please try again', model_name;
    END IF;

    RETURN v_result;
END;
$$;

CREATE OR REPLACE FUNCTION run_forecast(
    model_name TEXT,
    input_table TEXT,
    date_column TEXT,
    value_column TEXT,
    horizon INT,
    forecast_step INTERVAL DEFAULT '1 day'
)
RETURNS TABLE(forecast_date TIMESTAMP, forecast_value DOUBLE PRECISION)
SECURITY DEFINER
AS $$
DECLARE
    rec_model RECORD;
    v_opt_result arima_optimise_result;
    v_func_name TEXT;
    v_func_exists BOOLEAN;
    v_last_date TIMESTAMP;
BEGIN
    -- Safety precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);
    
    IF forecast_step IS NULL THEN
        -- TODO: Estimate step from data
        forecast_step := '1 day'::interval;
    END IF;

    IF model_name = 'autoarima' THEN
        -- Get params and hyperparams for forecasting
        EXECUTE format(
            'SELECT 
                m.id,
                m.use_log_transform,
                a.d, a.c, 
                a.phi, a.theta,
                (a.incremental_state).p,
                (a.incremental_state).q,
                (a.incremental_state).e_lags AS residuals,
                (a.incremental_state).y_lags,
                (a.incremental_state).diff_buf,
                (a.incremental_state).css
            FROM models m
            INNER JOIN model_arima_stats a ON m.id = a.model_id AND a.is_active = TRUE
            WHERE m.model_type = %L AND m.input_table = %L AND m.date_column = %L AND m.value_column = %L',
            
            model_name, input_table, date_column, value_column
        ) INTO rec_model;

        IF rec_model IS NULL THEN
            RAISE WARNING 'run_forecast: no autoarima model found, please run create_forecast first';
            RETURN;
        END IF;

        -- Get the last timestamp to build forecast dates
        EXECUTE format(
            'SELECT MAX(%I) FROM %I',
            date_column,
            input_table
        )
        INTO v_last_date;

        RETURN QUERY
            SELECT 
                *
            FROM
                arima_run_forecast_incremental(
                    rec_model.p,
                    rec_model.d,
                    rec_model.q,
                    rec_model.phi,
                    rec_model.theta,
                    rec_model.c,
                    rec_model.y_lags,
                    rec_model.residuals,
                    rec_model.diff_buf,
                    v_last_date,
                    horizon,
                    forecast_step,
                    rec_model.use_log_transform
                );
        RETURN;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION remove_forecast(
    model_type model,
    input_table TEXT,
    date_column TEXT,
    value_column TEXT
)
RETURNS BOOLEAN
SECURITY DEFINER
AS $$
DECLARE
    v_deleted_id BIGINT;
BEGIN
    -- Safety precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);

    EXECUTE format(
        'DELETE FROM models
        WHERE model_type = %L AND input_table = %L AND date_column = %L AND value_column = %L
        RETURNING id;',
        
        model_type, input_table, date_column, value_column
    ) INTO v_deleted_id;

    EXECUTE format(
        'DROP TRIGGER IF EXISTS %I_on_insert_%I_%I ON %I;',
        
        model_type, input_table, value_column, input_table
    );

    IF v_deleted_id IS NOT NULL THEN
        RETURN TRUE;
    ELSE
        RETURN FALSE;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Used in unit tests to verify that a model has retrained
CREATE OR REPLACE FUNCTION model_get_id(
    model_type model,
    input_table TEXT,
    date_column TEXT,
    value_column TEXT
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
        FROM models
        WHERE model_type = %L AND input_table = %L AND date_column = %L AND value_column = %L
        LIMIT 1',
        
        model_type, input_table, date_column, value_column
    ) INTO rec_result;

    RETURN rec_result.id;
END;
$$ LANGUAGE plpgsql;

-- Used to validate performance test
CREATE OR REPLACE FUNCTION model_get_loss(
    model_type model,
    input_table TEXT,
    date_column TEXT,
    value_column TEXT
)
RETURNS DOUBLE PRECISION
SECURITY DEFINER
AS $$
DECLARE
    rec_result RECORD;
    v_stats_table TEXT;
BEGIN
    -- Safety precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);

    IF model_type = 'autoarima' THEN
        v_stats_table := 'model_arima_stats';
    ELSE
        v_stats_table := 'model_' || model_type || '_stats';
    END IF;

    EXECUTE format(
        'SELECT
            (s.incremental_state).css 
        FROM models m 
        LEFT JOIN %I s 
            ON s.model_id = m.id 
        WHERE 
            m.model_type = %L AND 
            m.input_table = %L AND 
            m.date_column = %L AND 
            m.value_column = %L AND
            s.is_active = TRUE
        LIMIT 1;',
        
        v_stats_table, model_type, input_table, date_column, value_column
    ) INTO rec_result;

    RETURN rec_result.css;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION series_to_array(
    source_table TEXT,
    date_col TEXT,
    value_col TEXT,
    drop_nulls BOOLEAN DEFAULT FALSE,
    log_transform BOOLEAN DEFAULT FALSE
) RETURNS DOUBLE PRECISION[] AS $$
DECLARE
    arr DOUBLE PRECISION[];
    value_expr TEXT := format('%I', value_col);
    where_clause TEXT := 'WHERE 1=1';
BEGIN
    IF log_transform THEN
        -- Add 1 to avoid log(0)
        value_expr := format('LOG(%I + 1)', value_col);
    END IF;

    IF drop_nulls THEN
        where_clause := format('WHERE %I IS NOT NULL', value_col);
    END IF;

    EXECUTE format(
        'SELECT array_agg(%s ORDER BY %I) FROM %I %s',
        value_expr, date_col, source_table, value_col, where_clause
    ) INTO arr;

    -- Handle empty result
    IF arr IS NULL OR cardinality(arr) = 0 THEN
        RAISE EXCEPTION 'ARIMA: no data found in % for columns %, %',
            source_table, date_col, value_col;
    END IF;

    RETURN arr;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION reverse_array(arr anyarray)
RETURNS anyarray AS $$
SELECT COALESCE(array_agg(x ORDER BY i DESC), '{}')
FROM unnest(arr) WITH ORDINALITY AS t(x, i);
$$ LANGUAGE sql IMMUTABLE;