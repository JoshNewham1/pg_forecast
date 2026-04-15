/*
    pyautoarima assumes you have the following packages installed
    in the system's Python instance or in a virtualenv that has
    been referenced in `pyautoarima_config` below

    * pmdarima
    * numpy
*/

BEGIN;

-- =========================================================
--  Register extensions
-- =========================================================
CREATE EXTENSION IF NOT EXISTS plpython3u;

-- =========================================================
--  Result types
-- =========================================================

CREATE TYPE pyautoarima_fit_result AS (
    p INT,
    d INT,
    q INT,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    c DOUBLE PRECISION,
    css DOUBLE PRECISION,
    model_pickle BYTEA
);

CREATE TYPE pyautoarima_incremental_state AS (
    p INT,
    q INT,
    css DOUBLE PRECISION,
    y_lags DOUBLE PRECISION[],
    e_lags DOUBLE PRECISION[],
    diff_lags DOUBLE PRECISION[]
);

-- =========================================================
--  Tables
-- =========================================================

DROP TABLE IF EXISTS pyautoarima_vertices CASCADE;
DROP TABLE IF EXISTS model_pyautoarima_stats CASCADE;
DROP TABLE IF EXISTS pyautoarima_config CASCADE;

CREATE TABLE model_pyautoarima_stats (
    id BIGSERIAL PRIMARY KEY,
    model_id INT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    d INT,
    c DOUBLE PRECISION,
    incremental_state pyautoarima_incremental_state,
    model_pickle BYTEA,
    is_incremental BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN,
    CONSTRAINT unique_model_pyautoarima_stats UNIQUE (model_id, phi, theta, d, c)
);

CREATE TABLE pyautoarima_vertices (
    id BIGSERIAL PRIMARY KEY,
    stats_id BIGINT NOT NULL REFERENCES model_pyautoarima_stats(id) ON DELETE CASCADE,
    vertex_index INT,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    incremental_state pyautoarima_incremental_state
);

CREATE TABLE IF NOT EXISTS pyautoarima_config (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- =========================================================
--  Python virtualenv & Helpers
-- =========================================================

INSERT INTO pyautoarima_config (key, value)
VALUES ('venv', '/home/josh/diss/.venv/lib/python3.12/site-packages')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

CREATE OR REPLACE FUNCTION activate_python_venv()
RETURNS VOID AS
$$
if 'venv_activated' not in GD:
    import sys
    import os
    venv = plpy.execute("SELECT value FROM pyautoarima_config WHERE key='venv'")[0]['value']
    if venv not in sys.path:
        sys.path.append(venv)
    
    import numpy as np
    import pmdarima as pm
    import pickle
    
    GD['np'] = np
    GD['pm'] = pm
    GD['pickle'] = pickle
    
    # Define the transition logic in GD to be reused efficiently
    def transition(state_dict, y_raw, phi, theta, c):
        np = GD['np']
        p = state_dict['p']
        q = state_dict['q']
        diff_lags = np.array(state_dict['diff_lags'], dtype=float)
        y_lags = np.array(state_dict['y_lags'], dtype=float)
        e_lags = np.array(state_dict['e_lags'], dtype=float)
        
        # Differencing (any d)
        x = y_raw
        for i in range(len(diff_lags)):
            delta = x - diff_lags[i]
            diff_lags[i] = x
            x = delta
        y = x
        
        # Prediction
        y_hat = c
        if len(phi):
            y_hat += phi @ y_lags
        if len(theta):
            y_hat += theta @ e_lags
        
        eps = y - y_hat
        
        # Lag updates
        if len(y_lags):
            y_lags[1:], y_lags[0] = y_lags[:-1], y
        if len(e_lags):
            e_lags[1:], e_lags[0] = e_lags[:-1], eps
            
        return {
            'p': p, 'q': q,
            'css': state_dict['css'] + eps * eps,
            'y_lags': y_lags.tolist(),
            'e_lags': e_lags.tolist(),
            'diff_lags': diff_lags.tolist()
        }
    
    GD['transition'] = transition
    GD['venv_activated'] = True
$$
LANGUAGE plpython3u VOLATILE;

-- =========================================================
-- Incremental ARIMA functions
-- =========================================================

CREATE OR REPLACE FUNCTION pyautoarima_full_table_state(
    ys DOUBLE PRECISION[],
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    c DOUBLE PRECISION,
    p INT,
    q INT,
    d INT
)
RETURNS pyautoarima_incremental_state
LANGUAGE plpython3u
AS $$
plpy.execute("SELECT activate_python_venv();")
np = GD['np']
transition = GD['transition']

state = {
    'p': p, 'q': q,
    'css': 0.0,
    'y_lags': [0.0] * p,
    'e_lags': [0.0] * q,
    'diff_lags': [0.0] * d
}

arr_phi = np.array(phi, dtype=float) if phi else np.array([], dtype=float)
arr_theta = np.array(theta, dtype=float) if theta else np.array([], dtype=float)

for y in ys:
    state = transition(state, y, arr_phi, arr_theta, c)

return state
$$;

CREATE OR REPLACE FUNCTION pyautoarima_generate_vertices(
    stats_id BIGINT,
    p INT,
    q INT,
    d_order INT,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    c DOUBLE PRECISION,
    history DOUBLE PRECISION[],
    tolerance DOUBLE PRECISION DEFAULT 0.05
)
RETURNS VOID
LANGUAGE plpython3u
AS $$
plpy.execute("SELECT activate_python_venv();")
np = GD['np']
transition = GD['transition']

arr_phi = np.array(phi if phi else [], dtype=float)
arr_theta = np.array(theta if theta else [], dtype=float)
params = np.concatenate([arr_phi, arr_theta])
dim = len(params)

def get_full_state(p_vals, t_vals):
    state = {
        'p': p, 'q': q,
        'css': 0.0,
        'y_lags': [0.0] * p,
        'e_lags': [0.0] * q,
        'diff_lags': [0.0] * d_order
    }
    for y in history:
        state = transition(state, y, p_vals, t_vals, c)
    return state

# Coordinate Perturbations
for i in range(dim):
    v = params.copy()
    v[i] += tolerance if v[i] >= 0 else -tolerance
    v_phi = v[:len(arr_phi)]
    v_theta = v[len(arr_phi):]
    
    v_state = get_full_state(v_phi, v_theta)
    
    plpy.execute(plpy.prepare(
        "INSERT INTO pyautoarima_vertices (stats_id, vertex_index, phi, theta, incremental_state) VALUES ($1, $2, $3, $4, $5)",
        ["bigint", "int", "double precision[]", "double precision[]", "pyautoarima_incremental_state"]
    ), [stats_id, i, v_phi.tolist(), v_theta.tolist(), v_state])

# Scaling towards origin Vertex
if dim > 0:
    dist = np.linalg.norm(params)
    scale = 1.0 - tolerance / dist if dist > tolerance else 0.5
    v = params * scale
    v_phi = v[:len(arr_phi)]
    v_theta = v[len(arr_phi):]
    
    v_state = get_full_state(v_phi, v_theta)
    
    plpy.execute(plpy.prepare(
        "INSERT INTO pyautoarima_vertices (stats_id, vertex_index, phi, theta, incremental_state) VALUES ($1, $2, $3, $4, $5)",
        ["bigint", "int", "double precision[]", "double precision[]", "pyautoarima_incremental_state"]
    ), [stats_id, dim, v_phi.tolist(), v_theta.tolist(), v_state])
$$;

CREATE OR REPLACE FUNCTION pyautoarima_batch_incremental_update(
    stats_id BIGINT,
    is_incremental BOOLEAN,
    new_ys DOUBLE PRECISION[]
)
RETURNS BOOLEAN
LANGUAGE plpython3u
AS $$
plpy.execute("SELECT activate_python_venv();")
np = GD['np']
transition = GD['transition']

res = plpy.execute(f"SELECT phi, theta, c, incremental_state FROM model_pyautoarima_stats WHERE id = {stats_id}")
if not res: 
    return True
row = res[0]
phi = np.array(row['phi'], dtype=float) if row['phi'] else np.array([], dtype=float)
theta = np.array(row['theta'], dtype=float) if row['theta'] else np.array([], dtype=float)
c = row['c']
c_state = row['incremental_state']

is_ok = True

if is_incremental:
    # Breach check for p+q=0 (matching main.py)
    if c_state['p'] + c_state['q'] == 0:
        is_ok = False
    
    v_rows = plpy.execute(f"SELECT id, incremental_state, phi, theta FROM pyautoarima_vertices WHERE stats_id = {stats_id}")
    v_data = []
    for vr in v_rows:
        v_data.append({
            'id': vr['id'],
            'state': vr['incremental_state'],
            'phi': np.array(vr['phi'], dtype=float) if vr['phi'] else np.array([], dtype=float),
            'theta': np.array(vr['theta'], dtype=float) if vr['theta'] else np.array([], dtype=float)
        })
    
    for y in new_ys:
        c_state = transition(c_state, y, phi, theta, c)
        for vd in v_data:
            vd['state'] = transition(vd['state'], y, vd['phi'], vd['theta'], c)

    # Breach check for CSS (matching main.py)
    if is_ok and v_data:
        min_css = min(vd['state']['css'] for vd in v_data)
        if min_css < c_state['css'] * (1 - 1e-12):
            is_ok = False

    if is_ok and v_data:
        # Update vertex states in DB
        for vd in v_data:
            s = vd['state']
            plpy.execute(plpy.prepare(
                "UPDATE pyautoarima_vertices SET incremental_state = $1 WHERE id = $2",
                ["pyautoarima_incremental_state", "bigint"]
            ), [s, vd['id']])
else:
    for y in new_ys:
        c_state = transition(c_state, y, phi, theta, c)

# Update centre state in DB
plpy.execute(plpy.prepare(
    "UPDATE model_pyautoarima_stats SET incremental_state = $1 WHERE id = $2",
    ["pyautoarima_incremental_state", "bigint"]
), [c_state, stats_id])

return is_ok
$$;

CREATE OR REPLACE FUNCTION pyautoarima_fit_series(
    y DOUBLE PRECISION[]
)
RETURNS pyautoarima_fit_result
LANGUAGE plpython3u
AS $$
plpy.execute("SELECT activate_python_venv();")
np = GD['np']
pm = GD['pm']
pickle = GD['pickle']

np.random.seed(42)

model = pm.auto_arima(
    y,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_p=5, max_q=5, max_d=2,
)

p, d, q = model.order
phi, theta, c = [], [], 0.0

for name, val in zip(model.arima_res_.param_names, model.arima_res_.params):
    if name in ("const", "intercept"):
        c = float(val)
    elif name.startswith("ar.L"):
        phi.append(float(val))
    elif name.startswith("ma.L"):
        theta.append(float(val))

# Initial state CSS
state = {
    'p': p, 'q': q,
    'css': 0.0,
    'y_lags': [0.0] * p,
    'e_lags': [0.0] * q,
    'diff_lags': [0.0] * d
}
arr_phi = np.array(phi)
arr_theta = np.array(theta)
transition = GD['transition']
for val in y:
    state = transition(state, val, arr_phi, arr_theta, c)

model_pickle = pickle.dumps(model)

return (p, d, q, phi, theta, c, state['css'], model_pickle)
$$;

CREATE OR REPLACE FUNCTION pyautoarima_train(
    input_tab TEXT,
    date_col TEXT,
    value_col TEXT,
    is_incremental BOOLEAN
)
RETURNS pyautoarima_fit_result
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_model_id BIGINT;
    v_series DOUBLE PRECISION[];
    v_fit pyautoarima_fit_result;
    v_inc_state pyautoarima_incremental_state;
    v_stats_id BIGINT;
BEGIN
    PERFORM set_config('search_path', 'public,pg_temp', true);
    
    EXECUTE format('SELECT array_agg(%I ORDER BY %I) FROM %I WHERE %I IS NOT NULL', 
                   value_col, date_col, input_tab, value_col) INTO v_series;
                    
    v_fit := pyautoarima_fit_series(v_series);

    SELECT m.id FROM models m INTO v_model_id WHERE m.model_type = 'pyautoarima'::model AND m.input_table = input_tab AND m.date_column = date_col AND m.value_column = value_col;

    v_inc_state := pyautoarima_full_table_state(v_series, v_fit.phi, v_fit.theta, v_fit.c, v_fit.p, v_fit.q, v_fit.d);

    UPDATE model_pyautoarima_stats SET is_active = FALSE WHERE model_id = v_model_id;
    
    INSERT INTO model_pyautoarima_stats (model_id, phi, theta, d, c, incremental_state, model_pickle, is_incremental, is_active)
    VALUES (v_model_id, v_fit.phi, v_fit.theta, v_fit.d, v_fit.c, v_inc_state, v_fit.model_pickle, is_incremental, TRUE)
    ON CONFLICT (model_id, phi, theta, d, c)
    DO UPDATE SET is_active = TRUE, model_pickle = EXCLUDED.model_pickle, incremental_state = EXCLUDED.incremental_state, is_incremental = EXCLUDED.is_incremental
    RETURNING id INTO v_stats_id;
    
    IF is_incremental THEN
        DELETE FROM pyautoarima_vertices WHERE stats_id = v_stats_id;
        PERFORM pyautoarima_generate_vertices(v_stats_id, v_fit.p, v_fit.q, v_fit.d, v_fit.phi, v_fit.theta, v_fit.c, v_series);
    END IF;

    RETURN v_fit;
END;
$$;

CREATE OR REPLACE FUNCTION pyautoarima_create(
    input_table TEXT,
    date_column TEXT,
    value_column TEXT,
    use_continuous_agg BOOLEAN DEFAULT FALSE,
    is_incremental BOOLEAN DEFAULT FALSE
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_model_id BIGINT;
    v_fit pyautoarima_fit_result;
BEGIN
    PERFORM set_config('search_path', 'public,pg_temp', true);

    INSERT INTO models (model_type, input_table, date_column, value_column)
    VALUES ('pyautoarima'::model, input_table, date_column, value_column)
    RETURNING id INTO v_model_id;

    v_fit := pyautoarima_train(input_table, date_column, value_column, is_incremental);

    EXECUTE format(
        'CREATE TRIGGER pyautoarima_on_insert_%I_%I AFTER INSERT ON %I REFERENCING NEW TABLE AS new_rows FOR EACH STATEMENT EXECUTE FUNCTION pyautoarima_on_insert_trigger(%L, %L, %L, %L);',
        input_table, value_column, input_table, input_table, date_column, value_column, is_incremental
    );

    RETURN TRUE;
END;
$$;

CREATE OR REPLACE FUNCTION pyautoarima_on_insert_trigger()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_input_table TEXT := TG_ARGV[0];
    v_date_column TEXT := TG_ARGV[1];
    v_value_column TEXT := TG_ARGV[2];
    v_is_incremental BOOLEAN := TG_ARGV[3];
    v_model RECORD;
    v_ok BOOLEAN;
    v_series DOUBLE PRECISION[];
BEGIN
    PERFORM set_config('search_path', 'public,pg_temp', true);

    SELECT s.id AS stats_id
    FROM models m
    INNER JOIN model_pyautoarima_stats s ON s.model_id = m.id
    WHERE m.model_type = 'pyautoarima'::model AND m.input_table = v_input_table AND m.date_column = v_date_column AND m.value_column = v_value_column AND s.is_active = TRUE
    LIMIT 1
    INTO v_model;

    IF NOT FOUND THEN RETURN NULL; END IF;
    
    EXECUTE format('SELECT array_agg(%I ORDER BY %I) FROM new_rows', v_value_column, v_date_column) INTO v_series;
    v_ok := pyautoarima_batch_incremental_update(v_model.stats_id, v_is_incremental, v_series);

    IF NOT v_ok THEN
        PERFORM pyautoarima_train(v_input_table, v_date_column, v_value_column, v_is_incremental);
    END IF;

    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION pyautoarima_forecast(
    input_table TEXT,
    date_column TEXT,
    value_column TEXT,
    horizon INT,
    forecast_step INTERVAL DEFAULT '1 day'
)
RETURNS TABLE(forecast_date TIMESTAMP, forecast_value DOUBLE PRECISION)
LANGUAGE plpython3u
SECURITY DEFINER
AS $$
plpy.execute("SELECT activate_python_venv();")
pickle = GD['pickle']

sql = f"""
SELECT s.model_pickle
FROM models m
JOIN model_pyautoarima_stats s ON s.model_id = m.id
WHERE m.model_type = 'pyautoarima'::model
  AND m.input_table = {input_table!r}
  AND m.date_column = {date_column!r}
  AND m.value_column = {value_column!r}
  AND s.is_active = TRUE
LIMIT 1
"""
res = plpy.execute(sql)
if not res:
    plpy.error("Model not found")

model = pickle.loads(res[0]['model_pickle'])
forecast_vals = model.predict(n_periods=horizon)

date_sql = f"SELECT MAX({date_column}) AS last_date FROM {input_table}"
last_date = plpy.execute(date_sql)[0]["last_date"]

arr_forecasts = "{" + ",".join(str(float(v)) for v in forecast_vals) + "}"
return plpy.execute(f"""
    SELECT
        '{last_date}'::timestamp + (i * '{forecast_step}'::interval) AS forecast_date,
        ('{arr_forecasts}'::double precision[])[i] AS forecast_value
    FROM generate_series(1, {horizon}) AS i
""")
$$;

COMMIT;

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
            WHEN 'pyautoarima' THEN
                (SELECT pyautoarima_create(
                    input_table,
                    date_column,
                    value_column,
                    COALESCE((args->>'use_continuous_agg')::boolean, FALSE),
                    COALESCE((args->>'is_incremental')::boolean, FALSE)
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
    ELSIF model_name = 'pyautoarima' THEN
        v_func_name := model_name || '_forecast';

        -- Check if the function exists in the current schema
        SELECT EXISTS (
            SELECT 1
            FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE p.proname = v_func_name
            AND n.nspname = 'public'
        ) INTO v_func_exists;

        IF v_func_exists THEN
            RETURN QUERY EXECUTE format(
                'SELECT * FROM %I(%L, %L, %L, %L, %L)',
                v_func_name,
                input_table,
                date_column,
                value_column,
                horizon,
                forecast_step
            );
        ELSE
            RAISE WARNING 'run_forecast: % is not a valid model', model_name;
            RETURN;
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql;