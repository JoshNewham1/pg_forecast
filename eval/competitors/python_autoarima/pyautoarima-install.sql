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
    css DOUBLE PRECISION
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

CREATE TABLE IF NOT EXISTS models (
    id BIGSERIAL PRIMARY KEY,
    model_type TEXT,
    input_table TEXT,
    date_column TEXT,
    value_column TEXT
);

CREATE TABLE model_pyautoarima_stats (
    id BIGSERIAL PRIMARY KEY,
    model_id INT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    d INT,
    c DOUBLE PRECISION,
    incremental_state pyautoarima_incremental_state,
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
--  Python virtualenv
-- =========================================================

INSERT INTO pyautoarima_config (key, value)
VALUES ('venv', '/home/josh/diss/.venv/lib/python3.12/site-packages')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

CREATE OR REPLACE FUNCTION activate_python_venv()
RETURNS VOID AS
$$
venv = plpy.execute("SELECT value FROM pyautoarima_config WHERE key='venv'")[0]['value']
from sys import path
if venv not in path:
    path.append(venv)
$$
LANGUAGE plpython3u VOLATILE;

-- =========================================================
-- Incremental ARIMA update functions
-- =========================================================

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
import numpy as np

arr_phi = np.array(phi if phi else [], dtype=float)
arr_theta = np.array(theta if theta else [], dtype=float)
params = np.concatenate([arr_phi, arr_theta])
dim = len(params)

def transition(state_dict, y_raw, phi_v, theta_v, c_val):
    diff_lags = np.array(state_dict['diff_lags'], dtype=float) if state_dict['diff_lags'] else np.array([])
    y_lags = np.array(state_dict['y_lags'], dtype=float) if state_dict['y_lags'] else np.array([])
    e_lags = np.array(state_dict['e_lags'], dtype=float) if state_dict['e_lags'] else np.array([])
    
    x = y_raw
    for i in range(len(diff_lags)):
        delta = x - diff_lags[i]
        diff_lags[i] = x
        x = delta
    
    y = x
    y_hat = c_val
    if len(phi_v) > 0 and len(y_lags) > 0:
        y_hat += np.dot(phi_v, y_lags)
    if len(theta_v) > 0 and len(e_lags) > 0:
        y_hat += np.dot(theta_v, e_lags)
        
    eps = y - y_hat
    
    if len(y_lags) > 0:
        y_lags = np.roll(y_lags, 1)
        y_lags[0] = y
    if len(e_lags) > 0:
        e_lags = np.roll(e_lags, 1)
        e_lags[0] = eps
        
    return {
        'p': state_dict['p'], 'q': state_dict['q'],
        'css': state_dict['css'] + eps*eps,
        'y_lags': list(y_lags),
        'e_lags': list(e_lags),
        'diff_lags': list(diff_lags)
    }

def get_full_state(ys, phi_v, theta_v, c_val, p_v, q_v, d_v):
    state = {
        'p': p_v, 'q': q_v,
        'css': 0.0,
        'y_lags': [0.0]*p_v,
        'e_lags': [0.0]*q_v,
        'diff_lags': [0.0]*d_v
    }
    for y in ys:
        state = transition(state, y, phi_v, theta_v, c_val)
    return state

# 1. Coordinate Perturbations
for i in range(dim):
    v_params = params.copy()
    v_params[i] += tolerance if v_params[i] >= 0 else -tolerance
    v_phi = v_params[:len(arr_phi)]
    v_theta = v_params[len(arr_phi):]
    v_state = get_full_state(history, v_phi, v_theta, c, p, q, d_order)
    
    plpy.execute(plpy.prepare(
        "INSERT INTO pyautoarima_vertices (stats_id, vertex_index, phi, theta, incremental_state) VALUES ($1, $2, $3, $4, $5)",
        ["bigint", "int", "double precision[]", "double precision[]", "pyautoarima_incremental_state"]
    ), [stats_id, i, list(v_phi), list(v_theta), v_state])

# (d+1)th Vertex
if dim > 0:
    dist = np.linalg.norm(params)
    scale = 1.0 - tolerance / dist if dist > tolerance else 0.5
    v_params = params * scale
    v_phi = v_params[:len(arr_phi)]
    v_theta = v_params[len(arr_phi):]
    v_state = get_full_state(history, v_phi, v_theta, c, p, q, d_order)

    plpy.execute(plpy.prepare(
        "INSERT INTO pyautoarima_vertices (stats_id, vertex_index, phi, theta, incremental_state) VALUES ($1, $2, $3, $4, $5)",
        ["bigint", "int", "double precision[]", "double precision[]", "pyautoarima_incremental_state"]
    ), [stats_id, dim, list(v_phi), list(v_theta), v_state])

    # with open("pyautoarima.log", "a") as f:
    #     f.writelines([f"pyautoarima: Created vertices with phi = {list(v_phi)} and theta = {list(v_theta)}\n"])
$$;

CREATE OR REPLACE FUNCTION pyautoarima_incremental_update(
    stats_id BIGINT,
    is_incremental BOOLEAN,
    new_y DOUBLE PRECISION
)
RETURNS BOOLEAN
LANGUAGE plpython3u
AS $$
plpy.execute("SELECT activate_python_venv();")
import numpy as np

def transition(state_dict, y_raw, phi_v, theta_v, c_val):
    diff_lags = np.array(state_dict['diff_lags'], dtype=float) if state_dict['diff_lags'] else np.array([])
    y_lags = np.array(state_dict['y_lags'], dtype=float) if state_dict['y_lags'] else np.array([])
    e_lags = np.array(state_dict['e_lags'], dtype=float) if state_dict['e_lags'] else np.array([])
    
    x = y_raw
    for i in range(len(diff_lags)):
        delta = x - diff_lags[i]
        diff_lags[i] = x
        x = delta
    
    y = x
    y_hat = c_val
    if len(phi_v) > 0 and len(y_lags) > 0:
        y_hat += np.dot(phi_v, y_lags)
    if len(theta_v) > 0 and len(e_lags) > 0:
        y_hat += np.dot(theta_v, e_lags)
        
    eps = y - y_hat
    
    if len(y_lags) > 0:
        y_lags = np.roll(y_lags, 1)
        y_lags[0] = y
    if len(e_lags) > 0:
        e_lags = np.roll(e_lags, 1)
        e_lags[0] = eps
        
    return {
        'p': state_dict['p'], 'q': state_dict['q'],
        'css': state_dict['css'] + eps*eps,
        'y_lags': list(y_lags),
        'e_lags': list(e_lags),
        'diff_lags': list(diff_lags)
    }

res = plpy.execute(f"SELECT * FROM model_pyautoarima_stats WHERE id = {stats_id}")
if not res: return True
centre = res[0]

new_c_state = transition(centre['incremental_state'], new_y, centre['phi'], centre['theta'], centre['c'])
is_ok = True

if is_incremental:
    v_rows = plpy.execute(f"SELECT * FROM pyautoarima_vertices WHERE stats_id = {stats_id}")
    css_values = []
    
    for row in v_rows:
        new_v_state = transition(row['incremental_state'], new_y, row['phi'], row['theta'], centre['c'])
        css_values.append(new_v_state['css'])
        plpy.execute(plpy.prepare(
            "UPDATE pyautoarima_vertices SET incremental_state = $1 WHERE id = $2",
            ["pyautoarima_incremental_state", "bigint"]
        ), [new_v_state, row['id']])
    
    if len(css_values) > 0 and min(css_values) < new_c_state['css'] * (1 - 1e-12):
        is_ok = False

plpy.execute(plpy.prepare(
    "UPDATE model_pyautoarima_stats SET incremental_state = $1 WHERE id = $2",
    ["pyautoarima_incremental_state", "bigint"]
), [new_c_state, stats_id])

return is_ok
$$;

CREATE OR REPLACE FUNCTION pyautoarima_fit_series(
    y DOUBLE PRECISION[]
)
RETURNS pyautoarima_fit_result
LANGUAGE plpython3u
AS $$
plpy.execute("SELECT activate_python_venv();")
import pmdarima as pm
import numpy as np

model = pm.auto_arima(
    y,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_p=5, max_q=5, max_d=2,
)

p, d, q = model.order
residuals = model.resid()
css = np.sum(residuals ** 2)

# with open("pyautoarima.log", "a") as f:
#     f.writelines([f"Trained pyautoarima with {len(y)} records, p = {p}, d = {d}, q = {q}, css = {css}\n"])

phi = []
theta = []
c = 0.0

for name, val in zip(model.arima_res_.param_names, model.arima_res_.params):
    if name in ("const", "intercept"):
        c = float(val)
    elif name.startswith("ar.L"):
        phi.append(float(val))
    elif name.startswith("ma.L"):
        theta.append(float(val))

return (p, d, q, phi, theta, c, css)
$$;

CREATE OR REPLACE FUNCTION pyautoarima_init_state_helper(
    ys DOUBLE PRECISION[], 
    phi DOUBLE PRECISION[], 
    theta DOUBLE PRECISION[], 
    c DOUBLE PRECISION, 
    p INT, 
    q INT, 
    d INT
)
RETURNS pyautoarima_incremental_state
LANGUAGE plpython3u AS $$
import numpy as np

def transition(state_dict, y_raw, phi_v, theta_v, c_val):
    diff_lags = np.array(state_dict['diff_lags'], dtype=float)
    y_lags = np.array(state_dict['y_lags'], dtype=float)
    e_lags = np.array(state_dict['e_lags'], dtype=float)
    
    x = y_raw
    for i in range(len(diff_lags)):
        delta = x - diff_lags[i]
        diff_lags[i] = x
        x = delta
    y = x
    
    y_hat = c_val
    if len(phi_v) > 0 and len(y_lags) > 0: 
        y_hat += np.dot(phi_v, y_lags)
    if len(theta_v) > 0 and len(e_lags) > 0: 
        y_hat += np.dot(theta_v, e_lags)
    
    eps = y - y_hat
    
    if len(y_lags) > 0:
        y_lags = np.roll(y_lags, 1); y_lags[0] = y
    if len(e_lags) > 0:
        e_lags = np.roll(e_lags, 1); e_lags[0] = eps
        
    return {
        'p': state_dict['p'], 'q': state_dict['q'],
        'css': state_dict['css'] + eps*eps, 
        'y_lags': list(y_lags), 
        'e_lags': list(e_lags), 
        'diff_lags': list(diff_lags)
    }

state = {'p': p, 'q': q, 'css': 0.0, 'y_lags': [0.0]*p, 'e_lags': [0.0]*q, 'diff_lags': [0.0]*d}
phi_arr = np.array(phi)
theta_arr = np.array(theta)

for y_val in ys:
    state = transition(state, y_val, phi_arr, theta_arr, c)

return state
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
BEGIN
    PERFORM set_config('search_path', 'public,pg_temp', true);
    
    -- Extract series manually to pass to helper
    EXECUTE format('SELECT array_agg(%I ORDER BY %I) FROM %I WHERE %I IS NOT NULL', 
                   value_col, date_col, input_tab, value_col) INTO v_series;
                   
    v_fit := pyautoarima_fit_series(v_series);

    SELECT m.id FROM models m INTO v_model_id WHERE m.model_type = 'pyautoarima' AND m.input_table = input_tab AND m.date_column = date_col AND m.value_column = value_col;

    v_inc_state := pyautoarima_init_state_helper(v_series, v_fit.phi, v_fit.theta, v_fit.c, v_fit.p, v_fit.q, v_fit.d);

    UPDATE model_pyautoarima_stats SET is_active = FALSE WHERE model_id = v_model_id;
    
    INSERT INTO model_pyautoarima_stats (model_id, phi, theta, d, c, incremental_state, is_incremental, is_active)
    VALUES (v_model_id, v_fit.phi, v_fit.theta, v_fit.d, v_fit.c, v_inc_state, is_incremental, TRUE)
    ON CONFLICT ON CONSTRAINT unique_model_pyautoarima_stats
    DO UPDATE SET is_active = TRUE;
    
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
    v_stats_id BIGINT;
    v_series DOUBLE PRECISION[];
BEGIN
    PERFORM set_config('search_path', 'public,pg_temp', true);

    INSERT INTO models (model_type, input_table, date_column, value_column)
    VALUES ('pyautoarima', input_table, date_column, value_column)
    RETURNING id INTO v_model_id;

    v_fit := pyautoarima_train(input_table, date_column, value_column, is_incremental);

    IF is_incremental THEN
        SELECT s.id FROM model_pyautoarima_stats s INTO v_stats_id WHERE s.model_id = v_model_id AND s.is_active = TRUE;
        EXECUTE format('SELECT array_agg(%I ORDER BY %I) FROM %I WHERE %I IS NOT NULL', 
                       value_column, date_column, input_table, value_column) INTO v_series;

        PERFORM pyautoarima_generate_vertices(
            v_stats_id, v_fit.p, v_fit.q, v_fit.d, v_fit.phi, v_fit.theta, v_fit.c, v_series
        );
    END IF;

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
    v_value DOUBLE PRECISION;
    v_series DOUBLE PRECISION[];
BEGIN
    PERFORM set_config('search_path', 'public,pg_temp', true);

    SELECT m.id, s.id AS stats_id, (s.incremental_state).p, s.d, (s.incremental_state).q, s.phi, s.theta, s.c
    INTO v_model
    FROM models m
    INNER JOIN model_pyautoarima_stats s ON s.model_id = m.id
    WHERE m.model_type = 'pyautoarima' AND m.input_table = v_input_table AND m.date_column = v_date_column AND m.value_column = v_value_column AND s.is_active = TRUE
    LIMIT 1;

    IF NOT FOUND THEN RETURN NULL; END IF;
    IF (v_model.p + v_model.q) = 0 THEN
        v_is_incremental := FALSE;
    END IF;

    IF v_is_incremental AND NOT EXISTS (SELECT 1 FROM pyautoarima_vertices WHERE stats_id = v_model.stats_id) THEN
        EXECUTE format('SELECT array_agg(%I ORDER BY %I) FROM %I WHERE %I IS NOT NULL', 
                       v_value_column, v_date_column, v_input_table, v_value_column) INTO v_series;
        PERFORM pyautoarima_generate_vertices(v_model.stats_id, v_model.p, v_model.q, v_model.d, v_model.phi, v_model.theta, v_model.c, v_series);
    END IF; 

    FOR v_value IN EXECUTE format('SELECT %I FROM new_rows ORDER BY %I', v_value_column, v_date_column) LOOP
        v_ok := pyautoarima_incremental_update(v_model.stats_id, v_is_incremental, v_value);
        IF NOT v_ok THEN EXIT; END IF;
    END LOOP;

    IF NOT v_ok OR NOT v_is_incremental THEN
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
import numpy as np

# -------------------------------
# Load model
# -------------------------------
sql = f"""
SELECT 
    (s.incremental_state).p,
    s.d,
    (s.incremental_state).q,
    s.phi, s.theta, s.c,
    s.incremental_state
FROM models m
JOIN model_pyautoarima_stats s ON s.model_id = m.id
WHERE m.model_type = 'pyautoarima'
  AND m.input_table = {input_table!r}
  AND m.date_column = {date_column!r}
  AND m.value_column = {value_column!r}
  AND s.is_active = TRUE
LIMIT 1
"""
model_row = plpy.execute(sql)
if not model_row:
    plpy.error(f"No trained pyautoarima model found for {input_table}, {date_column}, {value_column}")
m = model_row[0]

p, d, q = m["p"], m["d"], m["q"]
phi, theta, c = m["phi"], m["theta"], m["c"]
residuals = m["incremental_state"]["e_lags"]

# -------------------------------
# Load time series
# -------------------------------
arr_sql = f"""
SELECT array_agg({value_column} ORDER BY {date_column}) AS vals
FROM {input_table}
WHERE {value_column} IS NOT NULL
"""
arr_row = plpy.execute(arr_sql)
vals = arr_row[0]["vals"]
if not vals or len(vals) < 2:
    plpy.error(f"No data available for forecasting in {input_table}.{value_column}")

vals = np.array(vals, dtype=float)

# -------------------------------
# Difference if needed
# -------------------------------
if d > 0:
    initial_vals = vals[:d].copy()
    for i in range(d):
        vals[i+1:] = vals[i+1:] - vals[i:-1].copy()

# -------------------------------
# Select last values for forecast
# -------------------------------
ncond = max(p, q)
last_vals = vals[-ncond:] if ncond > 0 else np.array([], dtype=float)

# -------------------------------
# Forecast
# -------------------------------
forecast = np.zeros(horizon, dtype=float)

for t in range(horizon):
    yhat = c
    if p > 0:
        yhat += sum(phi[i]*last_vals[-(i+1)] for i in range(p))
    if q > 0:
        yhat += sum(theta[i]*residuals[-(i+1)] for i in range(q))
    eps = 0  # assume zero innovation for forecast
    forecast[t] = yhat
    # Shift last_vals and residuals
    if p > 0:
        last_vals = np.append(last_vals[1:], yhat)
    if q > 0:
        residuals = np.append(residuals[1:], eps)

# -------------------------------
# Re-integrate if differenced
# -------------------------------
if d > 0:
    for i in range(horizon):
        forecast[i] += vals[-d+i] if i < d else forecast[i-1]

# -------------------------------
# Compute forecast dates
# -------------------------------
date_sql = f"SELECT MAX({date_column}) AS last_date FROM {input_table}"
last_date = plpy.execute(date_sql)[0]["last_date"]
arr_forecasts = "{" + ",".join(str(float(v)) for v in forecast) + "}"

return plpy.execute(f"""
    SELECT
        '{last_date}'::timestamp + (i * '{forecast_step}'::interval) AS forecast_date,
        ('{arr_forecasts}'::double precision[])[i] AS forecast_value
    FROM generate_series(1, {horizon}) AS i
""")
$$;

COMMIT;