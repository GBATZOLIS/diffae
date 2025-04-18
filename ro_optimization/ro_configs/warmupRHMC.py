CONFIG = {
    # ─── Common ─────────────────────────────────────────────────────────────
    "random_seed": 42,

    # classifier + regularization weights (shared by GD & HMC)
    "classifier_weight": 1.0,
    "reg_norm_weight": 0.5,
    "reg_norm_type": "L2",

    # SNR & metric regularization
    "ro_SNR": 50,
    "reg_lambda": 1e-5,

    # how many GD steps to run before handing off to HMC
    "gd_warmup_steps": 2,

    # total “riemannian_steps” for whichever optim. picks up after warm‑up:
    #   - for GD warm‑up we’ll override riemannian_steps→gd_warmup_steps
    #   - for HMC this is the number of full HMC iterations
    "riemannian_steps": 2,

    # ─── RGD (gradient_descent) parameters ─────────────────────────────────
    "riemannian_lr_init": 5e-3,
    "optimizer_type": "RiemannianHMC",   # final optimizer
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
    

    # ─── RHMC (HMC) parameters ──────────────────────────────────────────────
    "leapfrog_steps": 2,
    "step_size": 5e-6,
    "momentum_method": "lanczos",
    "sign_for_logdet": -1.0,
    "lanczos_max_iter": 20,
    "lanczos_iter_logdet": 20,
    "lanczos_tol": 1e-8,
    "lanczos_reorth": True,

    # CG for metric inversions in RHMC and RGD
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 10,
    "cg_tol": 1e-6,
    "cg_max_iter": 20,

    # ─── Logging / output ──────────────────────────────────────────────────
    "log_dir": "ro_optimization/ro_results/warmupRHMC",
    "plot_filename": "warmupRHMC.png",
}
