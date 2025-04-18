CONFIG = {
    # Random seed for reproducibility
    "random_seed": 42,
    
    # Optimization function
    "classifier_weight": 5.,
    "reg_norm_weight": 2.,
    "reg_norm_type": "L2",

    # Riemannian optimization parameters
    "ro_SNR": 50, #SNR at which Riemannian optimization takes place
    "reg_lambda": 1e-5, # Regularization for the metric g(x)=J(x)^TJ(x)+λI
    "riemannian_steps": 6, # HMC sampler settings: number of full HMC iterations        
    
    # Optimizer selection:
    "optimizer_type": "RiemannianHMC",  # Choices:["gradient_descent", "trust_region", "RiemannianULA", "RiemannianMALA", "RiemannianHMC"]

    # Leapfrog integrator parameters
    "leapfrog_steps": 5,            # Number of leapfrog sub-steps per HMC iteration
    "step_size": 0.0001,              # Leapfrog step size (epsilon)

    # Momentum sampling method (used for p ~ N(0, g(x)^(-1)))
    "momentum_method": "lanczos",
    # Momentum convention (here, we sample p ~ N(0, g(x)^(-1)))
    # With sign_for_logdet = -1.0, the Hamiltonian becomes:
    #   H(x, p) = U(x) - 0.5 * log|g(x)| + 0.5 p^T g(x) p
    "sign_for_logdet": -1.0,
    
    # Lanczos routine settings (used for both momentum sampling and logdet approximation)
    "lanczos_max_iter": 20,
    "lanczos_iter_logdet": 20,
    "lanczos_tol": 1e-7,
    "lanczos_reorth": True,
    
    # Conjugate gradient (CG) settings for solving metric inversions:
    "cg_max_iter": 20,
    "cg_tol": 1e-6,
    "cg_preconditioner": "diagonal",
    "cg_precond_diag_samples": 10,
    
    # Retraction/Geometry settings
    "retraction_operator": "denoiser",  # Must match your retraction and vector transport choices
    
    # Initial latent point: replace <your_initial_tensor_here> with your starting tensor of shape (B, 512)
    "initial_point": None,  #<your_initial_tensor_here>
    
    # Logging
    "log_dir": "ro_optimization/ro_results/hmc",
    "plot_filename": "hmc.png",
}
