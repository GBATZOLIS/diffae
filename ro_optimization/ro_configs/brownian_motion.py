CONFIG = {
    # Random seed for reproducibility
    "random_seed": 42,

    # Riemannian optimization parameters
    "ro_SNR": 20, #SNR at which Riemannian optimization takes place
    "reg_lambda": 1e-5,
    "riemannian_steps": 10,
    "riemannian_lr_init": 5e-3,
    
    # Optimizer selection:
    "optimizer_type": "brownian_dynamics",  # Choices:["gradient_descent", "trust_region", "brownian_dynamics"]

    # Brownian Motion (Itô SDE) parameters (added)
    "dt": 0.01,
    "noise_scale": 1.0,
    "diffusion_approx": "lanczos",  # Choose between "chebyshev" or "lanczos"
    
    # Lanczos solver settings (added)
    "lanczos_max_iter": 25,
    "lanczos_iter_logdet": 25,
    "lanczos_tol": 1e-7,
    "lanczos_reorth": True,

    # Chebyshev solver settings (added)
    "chebyshev_order": 20,
    "chebyshev_bound_iter": 10,

    # Trust-region parameters
    "trust_region_delta0": 0.1,
    "trust_region_eta_success": 0.75,
    "trust_region_eta_fail": 0.25,
    "trust_region_gamma_inc": 2.0,
    "trust_region_gamma_dec": 0.5,

    # Line search parameters (used by gradient descent branch)
    "line_search": "strong_wolfe",
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.7,
    "max_bracket": 11,
    "max_zoom": 10,
    "max_alpha": 300,
    "armijo_rho": 1e-6,
    "armijo_beta": 0.1,

    # Retraction operator options: "identity" or "denoiser"
    "retraction_operator": "denoiser",

    # Momentum settings (if used in gradient descent)
    "use_momentum": False,  # (set to False for now)
    "momentum_coeff": 0.6,

    # Settings for fast calculation of Riemannian gradient via CG
    "cg_preconditioner": 'diagonal',
    "cg_precond_diag_samples": 10, 
    "cg_tol": 1e-6, 
    "cg_max_iter": 20,

    # Logging
    "log_dir": "ro_optimization/ro_results/brownian_motion",
    "plot_filename": "brownian_motion.png",
}
