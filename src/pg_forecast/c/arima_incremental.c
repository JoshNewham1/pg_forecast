#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "utils.h"
#include "arima.h"
#include <string.h>
#include "constants.h"
#include <math.h>

PG_FUNCTION_INFO_V1(css_incremental_transition);
PG_FUNCTION_INFO_V1(_css_incremental_transition);
PG_FUNCTION_INFO_V1(_css_incremental_final);

/* -------------------------------------------------------------------------
 * State helpers
 * ------------------------------------------------------------------------- */

static arima_inc_state_t *
css_state_from_tuple(HeapTupleData *tup, TupleDesc desc)
{
    bool is_null;
    arima_inc_state_t *s = palloc0(sizeof(*s));

    s->t   = DatumGetInt32(heap_getattr(tup, 1, desc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: t cannot be NULL")));
    s->p   = DatumGetInt32(heap_getattr(tup, 2, desc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: p cannot be NULL")));
    s->q   = DatumGetInt32(heap_getattr(tup, 3, desc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: q cannot be NULL")));

    ArrayType *y_lags_arr = DatumGetArrayTypeP(heap_getattr(tup, 4, desc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: y_lags cannot be NULL")));
    int n_y = ArrayGetNItems(ARR_NDIM(y_lags_arr), ARR_DIMS(y_lags_arr));
    if (n_y > AR_MAX)
        ereport(ERROR, (errmsg("css_incremental: y_lags too long")));
    memcpy(s->y_lags, ARR_DATA_PTR(y_lags_arr), n_y * sizeof(double));

    ArrayType *e_lags_arr = DatumGetArrayTypeP(heap_getattr(tup, 5, desc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: e_lags cannot be NULL")));
    int n_e = ArrayGetNItems(ARR_NDIM(e_lags_arr), ARR_DIMS(e_lags_arr));
    if (n_e > MA_MAX)
        ereport(ERROR, (errmsg("css_incremental: e_lags too long")));
    memcpy(s->e_lags, ARR_DATA_PTR(e_lags_arr), n_e * sizeof(double));

    s->css = DatumGetFloat8(heap_getattr(tup, 6, desc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: css cannot be NULL")));

    s->d        = DatumGetInt32(heap_getattr(tup, 7, desc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: d cannot be NULL")));
    s->n_diff   = DatumGetInt32(heap_getattr(tup, 8, desc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: n_diff cannot be NULL")));

    ArrayType *diff_buf_arr = DatumGetArrayTypeP(heap_getattr(tup, 9, desc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: diff_buf cannot be NULL")));
    int n_diff_buf = ArrayGetNItems(ARR_NDIM(diff_buf_arr), ARR_DIMS(diff_buf_arr));
    if (n_diff_buf > D_MAX)
        ereport(ERROR, (errmsg("css_incremental: diff_buf too long")));
    memcpy(s->diff_buf, ARR_DATA_PTR(diff_buf_arr), n_diff_buf * sizeof(double));

    return s;
}

static Datum css_state_to_tuple(arima_inc_state_t *state, TupleDesc tupdesc)
{
    Datum values[9];
    bool nulls[9] = {false};

    values[0] = Int32GetDatum(state->t);
    values[1] = Int32GetDatum(state->p);
    values[2] = Int32GetDatum(state->q);
    values[3] = PointerGetDatum(build_float8_array(state->y_lags, state->p));
    values[4] = PointerGetDatum(build_float8_array(state->e_lags, state->q));
    values[5] = Float8GetDatum(state->css);
    values[6] = Int32GetDatum(state->d);
    values[7] = Int32GetDatum(state->n_diff);
    values[8] = PointerGetDatum(build_float8_array(state->diff_buf, state->d > 0 ? state->d + 1 : 0));

    HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    return HeapTupleGetDatum(tuple);
}

static arima_inc_state_t* css_incremental_init(int p, int q, int d)
{
    if (d < 0 || d > D_MAX)
        ereport(ERROR, (errmsg("css_incremental: d must be between 0 and %d", D_MAX)));

    arima_inc_state_t *state = palloc0(sizeof(arima_inc_state_t));

    state->t = 0;
    state->p = p;
    state->q = q;
    state->css = 0.0;
    for (int i = 0; i < AR_MAX; i++) state->y_lags[i] = 0.0;
    for (int i = 0; i < MA_MAX; i++) state->e_lags[i] = 0.0;

    state->d = d;
    state->n_diff = 0;
    for (int i = 0; i < D_MAX + 1; i++) state->diff_buf[i] = 0.0;

    return state;
}

/* -------------------------------------------------------------------------
 * Incremental differencing
 * ------------------------------------------------------------------------- */

static bool
arima_incremental_difference(arima_inc_state_t *state, double y, double *out)
{
    int d = state->d;

    // No differencing
    if (d == 0)
    {
        *out = y;
        return true;
    }

    // Warm-up: fill buffer
    if (state->n_diff < d + 1)
    {
        state->diff_buf[state->n_diff++] = y;

        if (state->n_diff < d + 1)
            return false;  // not ready yet
    }
    else
    {
        // Shift buffer to make room for newest
        memmove(state->diff_buf, state->diff_buf + 1, sizeof(double) * d);
        state->diff_buf[d] = y;
    }

    // Compute differenced value
    double curr[D_MAX + 1];
    double next[D_MAX + 1];

    for (int i = 0; i < D_MAX + 1; i++) curr[i] = 0.0;  // Initialise curr

    // Copy needed part of diff_buf into curr
    int len = d + 1;
    for (int i = 0; i < len; i++)
        curr[i] = state->diff_buf[i];

    for (int i = 0; i < d; i++)
    {
        // Compute first differences
        for (int j = 0; j < len - 1; j++)
            next[j] = curr[j + 1] - curr[j];

        // Swap buffers: copy next -> curr safely
        for (int j = 0; j < len - 1; j++)
            curr[j] = next[j];

        len--;
    }

    *out = curr[0];

    if (!isfinite(*out))
        elog(ERROR, "arima_incremental_difference produced non-finite value");

    return true;
}

static arima_inc_state_t*
css_incremental(arima_inc_state_t *state, double* phi, double* theta, double y_t, double c)
{
    int p = state->p;
    int q = state->q;
    int ncond = max(p, q);

    // Compute differenced value
    double y_diff;
    bool ready = arima_incremental_difference(state, y_t, &y_diff);
    double e_t = y_diff - c;

    elog(DEBUG1, "css_incremental after differencing: p = %d, q = %d", state->p, state->q);

    // Differencing warm-up: skip CSS and lag updates
    if (!ready)
        return state;

    // AR part
    for (int j = 0; j < p; j++)
    {
        e_t -= phi[j] * state->y_lags[j];
    }

    // MA part
    int max_ma = min(state->t - ncond, q);
    if (max_ma > 0)
    {
    for (int j = 0; j < max_ma; j++)
        {
            e_t -= theta[j] * state->e_lags[j];
        }
    }

    // Skip first ncond terms
    if (state->t >= ncond)
    {
        if (isfinite(e_t))
            state->css += e_t * e_t;
        else
            state->css = INFINITY;
    }
    state->t++;

    /* 
        Shift lag arrays forwards one (overwriting last element)
        and add y_t and e_t to front
    */
    if (p > 0)
    {
        memmove(&state->y_lags[1], // Dest
                &state->y_lags[0], // Source
                sizeof(double) * (p - 1)); // Copy first p-1 entries
        state->y_lags[0] = y_diff;
    }

    if (q > 0)
    {
        memmove(&state->e_lags[1],
                &state->e_lags[0],
                sizeof(double) * (q - 1));
        state->e_lags[0] = isfinite(e_t) ? e_t : 0.0;
    }

    if (!isfinite(state->css))
    {
        state->css = INFINITY;
        elog(WARNING, "css_incremental: CSS infinity at t=%d", state->t);
    }

    return state;
}

/* -------------------------------------------------------------------------
 * PG function handlers
 * ------------------------------------------------------------------------- */

PGDLLEXPORT Datum
css_incremental_transition(PG_FUNCTION_ARGS)
{
    HeapTupleHeader tup_header = PG_GETARG_HEAPTUPLEHEADER(0); // css_incremental_state arg
    Oid tup_type = HeapTupleHeaderGetTypeId(tup_header);
    int tup_typmod = HeapTupleHeaderGetTypMod(tup_header);
    TupleDesc tupdesc = lookup_rowtype_tupdesc(tup_type, tup_typmod);
    HeapTupleData tup_data;
    tup_data.t_len = HeapTupleHeaderGetDatumLength(tup_header);
    tup_data.t_data = tup_header;

    arima_inc_state_t *state = css_state_from_tuple(&tup_data, tupdesc);

    // Fetch other args (y, c, phi, theta)
    ArrayType *phi_arr, *theta_arr;
    float8 y_t = PG_GETARG_FLOAT8(1);
    phi_arr = PG_GETARG_ARRAYTYPE_P(2);
    theta_arr = PG_GETARG_ARRAYTYPE_P(3);
    float8 c = PG_GETARG_FLOAT8(4);

    int n_phi, n_theta;
    double *phi = get_1d_double_array(phi_arr, &n_phi, "css_incremental_transition");
    double *theta = get_1d_double_array(theta_arr, &n_theta, "css_incremental_transition");

    check_arima_dims(state->p, state->q, n_phi, n_theta, "css_incremental_transition");

    state = css_incremental(state, phi, theta, y_t, c);

    ReleaseTupleDesc(tupdesc);
    PG_RETURN_DATUM(css_state_to_tuple(state, tupdesc));
}

PGDLLEXPORT Datum
_css_incremental_transition(PG_FUNCTION_ARGS)
{
    ArrayType *phi_arr, *theta_arr;
    float8 y_t = PG_GETARG_FLOAT8(1);
    phi_arr = PG_GETARG_ARRAYTYPE_P(2);
    theta_arr = PG_GETARG_ARRAYTYPE_P(3);
    float8 c = PG_GETARG_FLOAT8(4);
    int d = PG_GETARG_INT32(5);

    int n_phi, n_theta;
    double *phi = get_1d_double_array(phi_arr, &n_phi, "_css_incremental_transition");
    double *theta = get_1d_double_array(theta_arr, &n_theta, "_css_incremental_transition");

    arima_inc_state_t *state;
    if (PG_ARGISNULL(0))
    {
        // Infer p and q from phi/theta arrays
        state = css_incremental_init(n_phi, n_theta, d);
    }
    else
    {
        state = (arima_inc_state_t *) PG_GETARG_POINTER(0);
        check_arima_dims(state->p, state->q, n_phi, n_theta, "_css_incremental_transition");
    }

    state = css_incremental(state, phi, theta, y_t, c);
    PG_RETURN_POINTER(state);
}

PGDLLEXPORT Datum
_css_incremental_final(PG_FUNCTION_ARGS)
{
    arima_inc_state_t *state = (arima_inc_state_t *) PG_GETARG_POINTER(0);
    TupleDesc tupdesc;

    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        elog(ERROR, "return type must be composite");

    PG_RETURN_DATUM(css_state_to_tuple(state, tupdesc));
}