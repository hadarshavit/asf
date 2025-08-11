import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from typing import List, Dict, Tuple, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from functools import partial

from asf.presolving.presolver import AbstractPresolver


class ASAPv2(AbstractPresolver):
    """
    ASAPv2 with differential evolution instead of CMA-ES.
    """
    
    def __init__(
        self,
        budget: float,
        presolver_cutoff: float,
        maximize: bool = False,
        size_preschedule: int = 3,
        max_runtime_preschedule: float = -1,
        regularization_weight: float = 0.0,
        variance_weight: float = 0.0,
        penalty_factor: float = 2.0,
        de_popsize: int = 15,
        de_maxiter: int = 100,
        seed: int = 42,
        verbosity: int = 0
    ):
        super().__init__(
            presolver_cutoff=presolver_cutoff, 
            budget=budget, 
            maximize=maximize
        )
        
        self.size_preschedule = size_preschedule
        self.max_runtime_preschedule = max_runtime_preschedule
        self.regularization_weight = regularization_weight
        self.variance_weight = variance_weight
        self.penalty_factor = penalty_factor
        self.de_popsize = de_popsize
        self.de_maxiter = de_maxiter
        self.seed = seed
        self.verbosity = verbosity
        
        # Will be set during fit
        self.algorithms: List[str] = []
        self.numAlg: int = 0
        self.preschedule_algorithms: List[str] = []
        self.ialgos_preschedule: np.ndarray = None
        self.runtimes_preschedule: np.ndarray = None
        self.features = None
        self.performance = None
        self.schedule: List[Tuple[str, float]] = []


    def fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        """Train the ASAP v2 presolver"""
        
        # Extract data
        self.features = features
        self.performance = performance
        self.algorithms = list(performance.columns)
        self.numAlg = len(self.algorithms)
        self.size_preschedule = min(self.size_preschedule, self.numAlg - 1)
        
        # Convert to numpy
        self.feature_train = features.values
        self.performance_train = performance.values
        
        if self.verbosity > 0:
            print()
            print("+" * 40)
            print(f"Training ASAP v2 with {len(self.algorithms)} algorithms")
            print(f"Preschedule size: {self.size_preschedule}")
        
        # 1. Identify algorithms for preschedule
        if self.size_preschedule > 0:
            self._identify_algorithms_for_preschedule()
        else:
            self.runtimes_preschedule = np.zeros((0,))
            self.ialgos_preschedule = np.zeros((0,)).astype(int)
        
        # 2. Optimize preschedule using differential evolution
        if self.size_preschedule > 1:
            self._optimize_preschedule_de()
        elif self.verbosity > 0:
            print("1-D schedule can't be optimized.")
            
        # 3. Build final schedule
        self._build_schedule()
    

    def _identify_algorithms_for_preschedule(self):
        """Select algorithms with highest avg. performance"""
        avg_performance = np.mean(self.performance_train, axis=0)
        best_alg_indices = np.argsort(avg_performance)[:self.size_preschedule]
        
        self.ialgos_preschedule = best_alg_indices
        self.preschedule_algorithms = [self.algorithms[i] for i in best_alg_indices]
        
        # Initialize with equal time distribution
        if self.max_runtime_preschedule > 0:
            total_time = min(self.max_runtime_preschedule, self.presolver_cutoff)
        else:
            total_time = self.presolver_cutoff
        
        self.runtimes_preschedule = np.full(
            self.size_preschedule, total_time / self.size_preschedule
        )
        
        if self.verbosity > 0:
            print(f"Selected preschedule algorithms: {self.preschedule_algorithms}")
            print(f"Initial runtimes: {self.runtimes_preschedule}")
    

    def _optimize_preschedule_de(self):
        """Optimize preschedule bdugets using differential evolution"""
        
        if self.verbosity > 0:
            print("Optimizing preschedule with differential evolution...")
        
        total_runtime_preschedule = np.sum(self.runtimes_preschedule)
        
        def encode_runtimes(rt):
            """Encode runtime for optimization"""
            if len(rt) <= 1:
                return np.array([])
            
            rt_normalized = rt / total_runtime_preschedule
            # Return all but the last element (last is determined by sum constraint)
            return rt_normalized[:-1]
        
        def decode_runtimes(x):
            """Decode runtime from optimization variables"""
            if len(x) == 0:
                return self.runtimes_preschedule
            
            x_ = np.abs(x)
            rt = np.zeros(len(x_) + 1)
            
            # Normalize if sum exceeds 1
            if np.sum(x_) > 1.0:
                x_ = x_ / np.sum(x_)
            
            rt[:-1] = x_ * total_runtime_preschedule
            rt[-1] = total_runtime_preschedule - np.sum(rt[:-1])
            
            # Ensure all times are non-negative
            rt = np.maximum(rt, 0.001)  # Minimum 1ms per algorithm
            
            return rt
        
        def objective_function(x):
            """Evaluate preschedule performance"""
            try:
                decoded_runtimes = decode_runtimes(x)
                
                total_cost = 0.0
                costs = []
                
                for i in range(len(self.performance_train)):
                    instance_cost = self._simulate_preschedule(
                        self.performance_train[i], 
                        decoded_runtimes
                    )
                    costs.append(instance_cost)
                    total_cost += instance_cost
                
                costs = np.array(costs)
                
                # Add regularization
                regularization = 0.0
                if self.regularization_weight > 0:
                    # Penalize uneven time distribution
                    rt_normalized = decoded_runtimes / np.sum(decoded_runtimes)
                    regularization = (self.regularization_weight * 
                                    len(self.performance_train) * 
                                    self.presolver_cutoff * 
                                    np.var(rt_normalized))
                
                # Add variance penalty
                variance_penalty = 0.0
                if self.variance_weight > 0:
                    solved_costs = costs[costs < self.presolver_cutoff * self.penalty_factor]
                    if len(solved_costs) > 1:
                        variance_penalty = self.variance_weight * np.var(solved_costs)
                
                return total_cost + regularization + variance_penalty
                
            except Exception as e:
                if self.verbosity > 1:
                    print(f"Error in objective function: {e}")
                return len(self.performance_train) * self.presolver_cutoff * self.penalty_factor
        

        # Encode initial guess
        initial_encoded = encode_runtimes(self.runtimes_preschedule)
        
        if len(initial_encoded) == 0:
            if self.verbosity > 0:
                print("No optimization needed for single algorithm preschedule")
            return
        
        # Set up bounds
        bounds = [(0.01, 0.99) for _ in range(len(initial_encoded))]
        
        # Run differential evolution
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                seed=self.seed,
                popsize=self.de_popsize,
                maxiter=self.de_maxiter,
                disp=self.verbosity > 0,
                x0=initial_encoded
            )
            optimized_runtimes = decode_runtimes(result.x)
            self.runtimes_preschedule = optimized_runtimes
            
            if self.verbosity > 0:
                print(f"Optimization completed. Final objective: {result.fun}")
                
        except Exception as e:
            if self.verbosity > 0:
                print(f"Optimization failed: {e}")
                print("Using initial runtimes")
        
        if self.verbosity > 0:
            print(f"Optimized preschedule times: {dict(zip(self.preschedule_algorithms, self.runtimes_preschedule))}")
    

    def _simulate_preschedule(self, instance_performance, runtimes):
        """Simulate preschedule execution for one instance"""
        total_time = 0.0
        
        # Execute preschedule
        for i, (alg_idx, time_limit) in enumerate(zip(self.ialgos_preschedule, runtimes)):
            if alg_idx >= len(instance_performance):
                continue
                
            alg_runtime = instance_performance[alg_idx]
            
            if alg_runtime <= time_limit:
                # Solved in preschedule
                return total_time + alg_runtime
            
            # Not solved, add full time and continue
            total_time += time_limit
        
        # Preschedule failed - return cost with penalty
        return self.presolver_cutoff * self.penalty_factor

    
    def _build_schedule(self):
        """Build the final schedule"""
        self.schedule = []
        
        # Add preschedule algorithms
        for i, alg in enumerate(self.preschedule_algorithms):
            if i < len(self.runtimes_preschedule):
                self.schedule.append((alg, float(self.runtimes_preschedule[i])))
        
        if self.verbosity > 0:
            print()
            print(f"Final schedule: {self.schedule}")
            print("+" * 40)


    def predict(self, features: Optional[pd.DataFrame] = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        Returns the optimized preschedule (same for all features).
        """
        if self.schedule is None:
            raise ValueError("Must call fit() before predict()")
        
        if features is None:
            return {"default": self.schedule}
        
        # Return same schedule for all instances
        result = {}
        for instance_id in features.index:
            result[instance_id] = self.schedule.copy()
        
        return result
    






    # def get_preschedule_config(self) -> Dict[str, float]:
    #     """Get the optimized preschedule configuration"""
    #     if self.preschedule_algorithms and self.runtimes_preschedule is not None:
    #         return dict(zip(self.preschedule_algorithms, self.runtimes_preschedule))
    #     return {}
    
    # def get_configuration(self) -> Dict:
    #     """Return configuration for compatibility with ASF selectors"""
    #     return {
    #         'algorithms': self.algorithms,
    #         'budget': self.budget,
    #         'presolver_cutoff': self.presolver_cutoff,
    #         'size_preschedule': self.size_preschedule,
    #         'preschedule_config': self.get_preschedule_config(),
    #         'regularization_weight': self.regularization_weight,
    #         'variance_weight': self.variance_weight,
    #         'penalty_factor': self.penalty_factor
    #     }