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


def _require_gpytorch():
    if not _HAS_GPYTORCH:
        raise RuntimeError(
            "gpbackend : gpytorch requires gpytorch; install it with "
            "`pip install gpytorch` (or `pip install gpevo[gpytorch]`).")


def _new_likelihood():
    """Likelihood with the same noise constraint used while fitting."""
    return gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-8))


# ── kernel builder ──────────────────────────────────────────────────────────

def _build_kernel(n_features: int, core: str = "matern"):
    """Build the gpytorch equivalent of the uspexkit composite kernel.

    ``core="rbf"`` uses RBFKernel (the energy model), otherwise MaternKernel
    ν=2.5 (the density model).  WhiteKernel becomes the likelihood noise.
    """
    _require_gpytorch()
    linear = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
    linear.outputscale = 0.00581 ** 2

    if core == "rbf":
        base = gpytorch.kernels.RBFKernel(ard_num_dims=n_features)
    else:
        base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_features)
    core_kernel = gpytorch.kernels.ScaleKernel(base)
    core_kernel.outputscale = 0.35 ** 2
    core_kernel.base_kernel.lengthscale = 0.1        # broadcast to all dims

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

    Parameters
    ----------
    kernel : sklearn kernel, optional
        Accepted for API compatibility; the equivalent gpytorch kernel is
        always built internally.  The WhiteKernel ``noise_level`` is used as
        the initial likelihood noise when available.
    n_restarts_optimizer : int, default=0
        Number of L-BFGS restarts with different initialisations.
    alpha : float, default=1e-10
        Jitter added to the diagonal for numerical stability.
    normalize_y : bool, default=False
        Centre and scale *y* before fitting; predictions are un-scaled.
    random_state : int, optional
        Seeds torch so repeated fits are reproducible (sklearn uses 0).
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

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(self, X, y):
        _require_gpytorch()
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
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

        # --- initial noise guess (from sklearn WhiteKernel if available) ---
        init_noise = 0.031
        if self._sklearn_kernel is not None:
            try:
                for k in [self._sklearn_kernel.k1, self._sklearn_kernel.k2]:
                    if hasattr(k, 'noise_level'):
                        init_noise = float(k.noise_level)
                        break
            except Exception:
                pass

        # --- multiple restarts ---
        n_restarts = max(1, self.n_restarts_optimizer + 1)
        best_lml = -float('inf')
        best_state = None
        model = None
        likelihood = None
        lml_val = float('-inf')
        restart_info = []
        best_info = {'evals': 0}

        for _ in range(n_restarts):
            kernel = _build_kernel(n_features, self.core)
            likelihood = _new_likelihood()
            likelihood.noise = init_noise

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
            except Exception:
                # A failed restart used to vanish silently; keep a trace of it.
                restart_info.append({'lml': None, 'evals': state['evals'],
                                     'failed': True})
                continue

            # LML must be evaluated in train mode: in eval mode model(train_x)
            # returns the posterior (and warns, since input == train_inputs),
            # which inflates the LML and can pick the wrong restart.
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(self.alpha):
                output = model(train_x)
                lml_val = mll(output, train_y).item()

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
            best_lml = lml_val
            best_state = {
                'model': {k: v.clone() for k, v in model.state_dict().items()},
                'likelihood': {k: v.clone()
                               for k, v in likelihood.state_dict().items()},
            }

        kernel = _build_kernel(n_features, self.core)
        likelihood = _new_likelihood()
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

        # Diagnostics are computed once here and travel with the pickled model,
        # so a cache hit in core._load_or_fit_gpr costs nothing extra.
        self._fit_info = {
            'backend': 'gpytorch',
            'core': self.core,
            'n_train': int(n_samples),
            'n_features': int(n_features),
            'n_restarts': int(n_restarts),
            'training_iter': int(self.training_iter),
            'restart_lmls': [r['lml'] for r in restart_info],
            'failed_restarts': sum(r['failed'] for r in restart_info),
            'best_lml': float(best_lml),
            'best_evals': best_info['evals'],
            'y_mean': float(self._y_mean),
            'y_std': float(self._y_std),
        }
        try:
            linear, core_kernel = self.kernel_.kernels
            # Report the hyperparameters on the ORIGINAL target scale, so they
            # are comparable with train_RMSE / posterior_sigma below.  LML and
            # ARD lengthscales stay on the normalised / standardised scales.
            self._fit_info.update({
                'noise_sigma': float(self._likelihood.noise.item()) ** 0.5
                               * self._y_std,
                'mean_constant': float(self._model.mean_module.constant.item())
                                 * self._y_std + self._y_mean,
                'linear_outputscale': float(linear.outputscale.item())
                                      * self._y_std ** 2,
                'core_outputscale': float(core_kernel.outputscale.item())
                                    * self._y_std ** 2,
                'lengthscale': core_kernel.base_kernel.lengthscale.detach()
                                .numpy().ravel().tolist(),
            })
            # In-sample fit quality.  debug(False) suppresses the legitimate
            # GPInputWarning raised when predicting at train_inputs.
            with torch.no_grad(), gpytorch.settings.debug(False), \
                    gpytorch.settings.cholesky_jitter(self.alpha):
                posterior = self._model(self._X_train)
                mu = posterior.mean.numpy() * self._y_std + self._y_mean
                sd = posterior.stddev.numpy() * self._y_std
            resid = mu - y
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            self._fit_info.update({
                'train_rmse': float(np.sqrt(np.mean(resid ** 2))),
                'train_r2': (float(1.0 - np.sum(resid ** 2) / ss_tot)
                             if ss_tot > 0 else float('nan')),
                'train_max_resid': float(np.max(np.abs(resid))),
                'train_posterior_sigma_mean': float(sd.mean()),
            })
        except Exception:
            pass

        return self

    # ── predict ──────────────────────────────────────────────────────────

    def predict(self, X, return_std=False):
        _require_gpytorch()
        X = np.asarray(X, dtype=np.float64)
        test_x = torch.tensor(X, dtype=torch.float64)

        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(self.alpha):
            posterior = self._model(test_x)
            mean = posterior.mean.numpy()
            std = posterior.stddev.numpy()

        # un-normalise
        mean = mean * self._y_std + self._y_mean
        std = std * self._y_std

        if return_std:
            return mean, std
        return mean

    # ── score / LML ──────────────────────────────────────────────────────

    def log_marginal_likelihood(self, theta=None):
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
            kernel = _build_kernel(self._n_features, self.core)
            likelihood = _new_likelihood()

            if X_np is not None:
                train_x = torch.tensor(X_np, dtype=torch.float64)
                train_y = torch.tensor(y_np, dtype=torch.float64)
            else:
                train_x = torch.zeros(1, self._n_features, dtype=torch.float64)
                train_y = torch.zeros(1, dtype=torch.float64)

            self._model = _ExactGP(train_x, train_y, likelihood, kernel)
            self._model.load_state_dict(model_state)
            self._likelihood = likelihood
            self._likelihood.load_state_dict(likelihood_state)
            self._model.eval()
            self._likelihood.eval()
            self.kernel_ = self._model.covar_module
            self._X_train = train_x
            self._y_train = train_y
