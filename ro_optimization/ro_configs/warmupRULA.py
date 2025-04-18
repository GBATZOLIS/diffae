CONFIG = {
    # ─── Common ─────────────────────────────────────────────────────────────
    "random_seed": 42,

    # classifier + regularization weights (shared by GD & ULA)
    "classifier_weight": 1.,
    "reg_norm_weight": 0.5,
    "reg_norm_type": "L2",

    # SNR & metric regularization
    "ro_SNR": 50,
    "reg_lambda": 1e-5,

    # how many GD steps to run before handing off to ULA
    "gd_warmup_steps": 2,

    # total “riemannian_steps” for whichever optimizer picks up last:
    #   - for GD warm‑up we override riemannian_steps→gd_warmup_steps
    #   - for ULA this is the number of SDE steps
    "riemannian_steps": 5,

    # ─── RGD (gradient_descent) warm‑up parameters ─────────────────────────
    "riemannian_lr_init": 5e-3,
    "line_search": "strong_wolfe",
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.7,
    "max_bracket": 11,
    "max_zoom": 10,
    "max_alpha": 300,
    "armijo_rho": 1e-6,
    "armijo_beta": 0.1,
    "retraction_operator": "denoiser",
    "use_momentum": False,
    "momentum_coeff": 0.6,

    # ─── Combined CG settings (for both GD warm‑up & ULA drift) ────────────
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 10,
    "cg_tol": 1e-6,
    "cg_max_iter": 20,

    # ─── ULA (brownian_dynamics) parameters ────────────────────────────────
    "optimizer_type": "RiemannianULA",  # final sampler
    "dt": 0.0001,
    "noise_scale": 1.0,
    "diffusion_approx": "lanczos",      # or "chebyshev"

    # if using lanczos:
    "lanczos_max_iter": 20,
    "lanczos_iter_logdet": 20,
    "lanczos_tol": 1e-7,
    "lanczos_reorth": True,

    # if using chebyshev:
    "chebyshev_order": 20,
    "chebyshev_bound_iter": 10,

    # ─── Logging / output ──────────────────────────────────────────────────
    "log_dir": "ro_optimization/ro_results/warmupULA",
    "plot_filename": "warmupULA.png",
}
