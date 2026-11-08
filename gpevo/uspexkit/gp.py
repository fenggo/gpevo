"""Gaussian Process Regression via GPyTorch — sklearn-compatible wrapper.

Vendored from ``~/uspexkit/uspexkit/gp.py`` for the opt-in ``gpbackend : gpytorch``
INPUT.txt option.  It keeps the API the rest of uspexkit relies on:
``fit()``, ``predict(return_std=True)``, ``kernel_``,
``log_marginal_likelihood()`` and pickle serialization.

The sklearn composite kernel (``core.py``) is

    0.00581² · DotProduct(σ₀=0.412)
  + 0.35²    · Matern(ν=2.5, ARD lengthscales)   [density]
  + 0.35²    · RBF(ARD lengthscales)             [energy]
  + WhiteKernel(noise)

which maps to

    ScaleKernel(LinearKernel()) + ScaleKernel(MaternKernel | RBFKernel)

with observation noise handled by a ``GaussianLikelihood``.  ``core`` selects
between the Matern (density) and RBF (energy) variants so the two surrogate
models keep the same kernel families as the default sklearn backend.

Initial hyperparameters, their bounds and the noise constraint are read from
the sklearn kernel passed in, so edits to ``core.py``'s kernel carry over to
this backend.  DotProduct's ``sigma_0`` has no counterpart in gpytorch's
LinearKernel and is deliberately not mapped.

Importing this module never imports torch/gpytorch into the sklearn hot path:
``core.py`` imports ``GPyTorchRegressor`` lazily, inside the gpytorch branch.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import gpytorch
    _HAS_GPYTORCH = True
except ImportError:                                   # pragma: no cover
    torch = None
    gpytorch = None
    _HAS_GPYTORCH = False


# Used only when no sklearn kernel is supplied (e.g. direct unit tests).
_DEFAULT_NOISE = 0.031
_DEFAULT_BOUNDS = {
    'linear_scale': (1e-6, 1e-1),
    'core_scale': (1e-3, 1e1),
    'lengthscale': (1e-2, 1e2),
    'noise': (1e-8, 1e-1),
}


def _require_gpytorch():
    if not _HAS_GPYTORCH:
        raise RuntimeError(
            "gpbackend : gpytorch requires gpytorch; install it with "
            "`pip install gpytorch` (or `pip install gpevo[gpytorch]`).")


def _log_bounds(bounds, cap):
    """Finite, positive, capped (lo, hi) suitable for log-uniform sampling."""
    try:
        lo, hi = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError, IndexError):
        return cap
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        return cap
    lo, hi = max(lo, cap[0]), min(hi, cap[1])
    return (lo, hi) if lo < hi else cap


def _kernel_init(sklearn_kernel, n_features):
    """Initial hyperparameters and sampling bounds from the sklearn kernel.

    The composite kernel is walked by duck typing so this module never has to
    import sklearn's kernel classes.  ``constant_value``s are collected in
    ``get_params(deep=True)`` order: linear term first, non-linear term second.
    """
    init = {
        'linear_scale': 0.00581 ** 2,
        'core_scale': 0.35 ** 2,
        'lengthscale': np.full(n_features, 0.1, dtype=float),
        'noise': _DEFAULT_NOISE,
        'bounds': {k: tuple(v) for k, v in _DEFAULT_BOUNDS.items()},
    }
    if sklearn_kernel is None:
        return init

    objects = [sklearn_kernel]
    try:
        objects += [v for v in sklearn_kernel.get_params(deep=True).values()
                    if hasattr(v, 'get_params')]
    except Exception:
        pass

    constants = []
    for obj in objects:
        if hasattr(obj, 'noise_level'):
            init['noise'] = float(obj.noise_level)
            init['bounds']['noise'] = _log_bounds(
                getattr(obj, 'noise_level_bounds', None),
                _DEFAULT_BOUNDS['noise'])
        if hasattr(obj, 'length_scale'):
            try:
                ls = np.asarray(obj.length_scale, dtype=float).ravel()
                init['lengthscale'] = (np.full(n_features, ls[0])
                                       if ls.size == 1
                                       else np.resize(ls, n_features))
            except (TypeError, ValueError):
                pass
            init['bounds']['lengthscale'] = _log_bounds(
                getattr(obj, 'length_scale_bounds', None),
                _DEFAULT_BOUNDS['lengthscale'])
        if hasattr(obj, 'constant_value'):
            constants.append(float(obj.constant_value))
    if len(constants) >= 2:
        init['linear_scale'], init['core_scale'] = constants[0], constants[1]
    return init


def _sample_init(init, rng):
    """One log-uniform restart.  Restart 0 keeps the nominal initialisation."""
    out = {}
    for key in ('linear_scale', 'core_scale', 'noise'):
        lo, hi = init['bounds'][key]
        out[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    lo, hi = init['bounds']['lengthscale']
    out['lengthscale'] = np.exp(
        rng.uniform(np.log(lo), np.log(hi), size=len(init['lengthscale'])))
    # Interval.inverse_transform is undefined exactly at a bound.
    nlo, nhi = init['bounds']['noise']
    out['noise'] = min(max(out['noise'], nlo * 1.01), nhi * 0.99)
    return out


def _new_likelihood(noise_bounds=None):
    """Likelihood whose noise constraint matches the sklearn WhiteKernel."""
    if noise_bounds is None:
        constraint = gpytorch.constraints.GreaterThan(1e-8)
    else:
        lo, hi = noise_bounds
        constraint = gpytorch.constraints.Interval(lo, hi)
    return gpytorch.likelihoods.GaussianLikelihood(noise_constraint=constraint)


# ── kernel builder ──────────────────────────────────────────────────────────

def _build_kernel(n_features: int, core: str = "matern", init=None):
    """Build the gpytorch equivalent of the uspexkit composite kernel.

    ``core="rbf"`` uses RBFKernel (the energy model), otherwise MaternKernel
    ν=2.5 (the density model).  WhiteKernel becomes the likelihood noise.
    """
    _require_gpytorch()
    linear = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
    linear.outputscale = init['linear_scale'] if init else 0.00581 ** 2

    if core == "rbf":
        base = gpytorch.kernels.RBFKernel(ard_num_dims=n_features)
    else:
        base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_features)
    core_kernel = gpytorch.kernels.ScaleKernel(base)
    core_kernel.outputscale = init['core_scale'] if init else 0.35 ** 2
    core_kernel.base_kernel.lengthscale = (
        np.asarray(init['lengthscale']).reshape(1, -1)
        if init else 0.1)

    return linear + core_kernel


# ── internal GP model ───────────────────────────────────────────────────────

if _HAS_GPYTORCH:
    class _ExactGP(gpytorch.models.ExactGP):
        """Thin ExactGP wrapper used internally by GPyTorchRegressor."""

        def __init__(self, train_x, train_y, likelihood, kernel):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = kernel

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)


# ── sklearn-compatible regressor ────────────────────────────────────────────

class GPyTorchRegressor:
    """Gaussian Process Regressor backed by GPyTorch.

    Drop-in replacement for ``sklearn.gaussian_process.GaussianProcessRegressor``
    with the public API uspexkit consumes.  ``training_iter`` and
    ``learning_rate`` are exposed through INPUT.txt (``gpiter`` / ``gplr``).

    Unlike sklearn, ``log_marginal_likelihood()`` reports the TOTAL log marginal
    likelihood (sklearn's convention), not GPyTorch's per-point average.
    Passing ``theta`` is not supported and raises NotImplementedError.

    Parameters
    ----------
    kernel : sklearn kernel, optional
        Supplies the initial hyperparameters and their bounds.  When None the
        nominal uspexkit values are used.
    n_restarts_optimizer : int, default=0
        Number of extra L-BFGS restarts.  Restart 0 uses the nominal
        initialisation; the rest draw log-uniform hyperparameters from the
        kernel bounds using ``random_state``.
    alpha : float, default=1e-10
        Jitter added to the diagonal for numerical stability.
    normalize_y : bool, default=False
        Centre and scale *y* before fitting; predictions are un-scaled.
    random_state : int, optional
        Seeds an independent RNG for the restart draws; the global torch RNG
        is left untouched.
    training_iter : int, default=200
        Max L-BFGS iterations per restart.
    learning_rate : float, default=0.1
        L-BFGS step size.
    core : {"matern", "rbf"}, default="matern"
        Kernel family for the non-linear term.
    """

    def __init__(self, kernel=None, n_restarts_optimizer=0, alpha=1e-10,
                 normalize_y=False, random_state=None, training_iter=200,
                 learning_rate=0.1, core="matern"):
        self._sklearn_kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.training_iter = training_iter
        self.learning_rate = learning_rate
        self.core = core

        # Populated by fit()
        self.kernel_ = None       # gpytorch kernel (for API compat)
        self._model = None
        self._likelihood = None
        self._X_train = None      # torch tensor
        self._y_train = None      # torch tensor
        self._y_mean = 0.0
        self._y_std = 1.0
        self._lml = None
        self._fit_info = None     # training diagnostics, filled by fit()
        self._n_features = None
        self._noise_bounds = None

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(self, X, y):
        _require_gpytorch()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array of shape (n_samples, n_features), "
                f"got {X.shape}")
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        self._n_features = n_features

        # --- normalise y ---
        if self.normalize_y:
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y))
            if self._y_std < 1e-12:
                self._y_std = 1.0
            y_proc = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            y_proc = y.copy()

        train_x = torch.tensor(X, dtype=torch.float64)
        train_y = torch.tensor(y_proc, dtype=torch.float64)

        # --- initial hyperparameters, from the sklearn kernel when available ---
        init = _kernel_init(self._sklearn_kernel, n_features)
        self._noise_bounds = init['bounds']['noise']
        rng = np.random.default_rng(self.random_state)

        # --- multiple restarts ---
        n_restarts = max(1, self.n_restarts_optimizer + 1)
        best_lml = -float('inf')
        best_state = None
        last_error = None
        restart_info = []
        best_info = {'evals': 0}

        for restart in range(n_restarts):
            spec = init if restart == 0 else _sample_init(init, rng)
            kernel = _build_kernel(n_features, self.core, init=spec)
            likelihood = _new_likelihood(self._noise_bounds)
            nlo, nhi = self._noise_bounds
            likelihood.noise = min(max(spec['noise'], nlo * 1.01), nhi * 0.99)

            model = _ExactGP(train_x, train_y, likelihood, kernel)
            model.train()
            likelihood.train()

            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=self.learning_rate,
                max_iter=self.training_iter,
                line_search_fn='strong_wolfe',
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            state = {'evals': 0}

            def _closure():
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                state['evals'] += 1
                return loss

            try:
                with gpytorch.settings.cholesky_jitter(self.alpha):
                    optimizer.step(_closure)
            except Exception as exc:
                # Keep a trace: a failed restart used to vanish silently.
                last_error = exc
                restart_info.append({'lml': None, 'evals': state['evals'],
                                     'failed': True})
                continue

            # LML must be evaluated in train mode: in eval mode model(train_x)
            # returns the posterior (and warns, since input == train_inputs),
            # which inflates the LML and can pick the wrong restart.  GPyTorch
            # divides by n_samples; store the TOTAL to match sklearn.
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(self.alpha):
                output = model(train_x)
                lml_val = mll(output, train_y).item() * n_samples

            restart_info.append({'lml': lml_val, 'evals': state['evals'],
                                 'failed': False})
            if lml_val > best_lml:
                best_lml = lml_val
                best_info = restart_info[-1]
                best_state = {
                    'model': {k: v.clone() for k, v in model.state_dict().items()},
                    'likelihood': {k: v.clone()
                                   for k, v in likelihood.state_dict().items()},
                }

        if best_state is None:
            raise RuntimeError(
                f"GPyTorchRegressor.fit: all {n_restarts} L-BFGS restart(s) "
                f"failed (last error: {type(last_error).__name__}: "
                f"{last_error}). Check the training set for NaN/duplicate "
                f"rows or increase alpha.")

        kernel = _build_kernel(n_features, self.core)
        likelihood = _new_likelihood(self._noise_bounds)
        self._model = _ExactGP(train_x, train_y, likelihood, kernel)
        self._model.load_state_dict(best_state['model'])
        self._likelihood = likelihood
        self._likelihood.load_state_dict(best_state['likelihood'])
        self._model.eval()
        self._likelihood.eval()

        self._lml = best_lml
        self.kernel_ = self._model.covar_module
        self._X_train = train_x
        self._y_train = train_y
        self._fit_info = self._collect_fit_info(
            y, n_samples, n_features, n_restarts, restart_info, best_info)
        return self

    # ── diagnostics ──────────────────────────────────────────────────────

    def _collect_fit_info(self, y, n_samples, n_features, n_restarts,
                          restart_info, best_info):
        """Cheap, once-per-fit report; travels with the pickled model."""
        info = {
            'backend': 'gpytorch',
            'core': self.core,
            'n_train': int(n_samples),
            'n_features': int(n_features),
            'n_restarts': int(n_restarts),
            'training_iter': int(self.training_iter),
            'restart_lmls': [r['lml'] for r in restart_info],
            'failed_restarts': sum(r['failed'] for r in restart_info),
            'best_lml': float(self._lml),
            'best_evals': best_info['evals'],
            'y_mean': float(self._y_mean),
            'y_std': float(self._y_std),
        }
        try:
            linear, core_kernel = self.kernel_.kernels
            # Report hyperparameters on the ORIGINAL target scale so they are
            # comparable with train_RMSE / posterior_sigma.  LML is on the
            # normalised y scale; ARD lengthscales are on standardised X.
            info.update({
                'noise_sigma': float(self._likelihood.noise.item()) ** 0.5
                               * self._y_std,
                'mean_constant': float(self._model.mean_module.constant.item())
                                 * self._y_std + self._y_mean,
                'linear_outputscale': float(linear.outputscale.item())
                                      * self._y_std ** 2,
                'core_outputscale': float(core_kernel.outputscale.item())
                                    * self._y_std ** 2,
                'lengthscale': core_kernel.base_kernel.lengthscale.detach()
                                .cpu().numpy().ravel().tolist(),
            })
            # In-sample fit quality.  debug(False) suppresses the legitimate
            # GPInputWarning raised when predicting at train_inputs.
            with torch.no_grad(), gpytorch.settings.debug(False), \
                    gpytorch.settings.cholesky_jitter(self.alpha):
                posterior = self._model(self._X_train)
                mu = posterior.mean.cpu().numpy() * self._y_std + self._y_mean
                sd = posterior.stddev.cpu().numpy() * self._y_std
            resid = mu - y
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            info.update({
                'train_rmse': float(np.sqrt(np.mean(resid ** 2))),
                'train_r2': (float(1.0 - np.sum(resid ** 2) / ss_tot)
                             if ss_tot > 0 else float('nan')),
                'train_max_resid': float(np.max(np.abs(resid))),
                'train_posterior_sigma_mean': float(sd.mean()),
            })
        except Exception as exc:
            info['diagnostics_error'] = f'{type(exc).__name__}: {exc}'
        return info

    # ── predict ──────────────────────────────────────────────────────────

    def predict(self, X, return_std=False):
        _require_gpytorch()
        # A 1-D row means one sample; ExactGP.__call__ would otherwise treat
        # (n_features,) as n_features one-dimensional samples.
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        test_x = torch.tensor(X, dtype=torch.float64)

        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(self.alpha):
            posterior = self._model(test_x)
            mean = posterior.mean.cpu().numpy()
            std = posterior.stddev.cpu().numpy()

        # un-normalise
        mean = mean * self._y_std + self._y_mean
        std = std * self._y_std

        if return_std:
            return mean, std
        return mean

    # ── score / LML ──────────────────────────────────────────────────────

    def log_marginal_likelihood(self, theta=None):
        if theta is not None:
            raise NotImplementedError(
                "GPyTorchRegressor evaluates the marginal likelihood only at "
                "the fitted hyperparameters; theta is not supported.")
        if self._lml is None:
            raise RuntimeError('Model has not been fitted yet.')
        return self._lml

    def score(self, X, y):
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    # ── pickle support ───────────────────────────────────────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._model is not None:
            state['_model_state'] = {
                k: v.cpu().clone() for k, v in self._model.state_dict().items()
            }
            state['_likelihood_state'] = {
                k: v.cpu().clone() for k, v in self._likelihood.state_dict().items()
            }
        else:
            state['_model_state'] = None
            state['_likelihood_state'] = None

        if self._X_train is not None:
            state['_X_train_np'] = self._X_train.numpy()
            state['_y_train_np'] = self._y_train.numpy()
        else:
            state['_X_train_np'] = None
            state['_y_train_np'] = None

        state['_sklearn_kernel'] = None
        for key in ('_model', '_likelihood', 'kernel_', '_X_train', '_y_train'):
            state[key] = None
        return state

    def __setstate__(self, state):
        model_state = state.pop('_model_state', None)
        likelihood_state = state.pop('_likelihood_state', None)
        X_np = state.pop('_X_train_np', None)
        y_np = state.pop('_y_train_np', None)

        self.__dict__.update(state)

        if model_state is not None and self._n_features is not None:
            _require_gpytorch()
            if X_np is None or y_np is None:
                raise RuntimeError(
                    "pickled GPyTorchRegressor has no training data; refit it")
            kernel = _build_kernel(self._n_features, self.core)
            likelihood = _new_likelihood(getattr(self, '_noise_bounds', None))
            train_x = torch.tensor(X_np, dtype=torch.float64)
            train_y = torch.tensor(y_np, dtype=torch.float64)

            self._model = _ExactGP(train_x, train_y, likelihood, kernel)
            self._model.load_state_dict(model_state)
            self._likelihood = likelihood
            self._likelihood.load_state_dict(likelihood_state)
            self._model.eval()
            self._likelihood.eval()
            self.kernel_ = self._model.covar_module
            self._X_train = train_x
            self._y_train = train_y
