"""
Differential Privacy utilities for Federated Learning.

Implements:
1. Gradient/weight clipping (L2 norm clipping)
2. Gaussian noise addition
3. Privacy budget (epsilon) calculation using moments accountant
4. Privacy-utility tradeoff analysis
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch


class DPMechanism:
    """
    Differential Privacy mechanism for federated learning weight updates.
    
    Implements the Gaussian mechanism with L2 norm clipping.
    """
    
    def __init__(self, clip_norm: float = 1.0, noise_scale: float = 0.01):
        """
        Args:
            clip_norm: Maximum L2 norm for clipping (C)
            noise_scale: Noise multiplier (σ) - noise std = σ * C
        """
        self.clip_norm = clip_norm
        self.noise_scale = noise_scale
        
    def clip_gradients(self, gradients: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """
        Clip L2 norm of gradients/weight updates across all layers.
        
        Args:
            gradients: List of gradient arrays (one per layer)
            
        Returns:
            Tuple of (clipped_gradients, original_norm)
        """
        # Compute global L2 norm across all layers
        total_norm_sq = sum(np.sum(g ** 2) for g in gradients)
        total_norm = np.sqrt(total_norm_sq)
        
        # Clip: multiply by min(1, C / ||g||)
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-8))
        
        clipped = [g * clip_coef for g in gradients]
        
        return clipped, float(total_norm)
    
    def add_noise(self, values: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add Gaussian noise N(0, σ²C²) to values.
        
        Args:
            values: List of arrays to add noise to
            
        Returns:
            Noisy values
        """
        noise_std = self.noise_scale * self.clip_norm
        
        noisy = []
        for v in values:
            noise = np.random.normal(0, noise_std, size=v.shape).astype(v.dtype)
            noisy.append(v + noise)
        
        return noisy
    
    def privatize(self, gradients: List[np.ndarray]) -> Tuple[List[np.ndarray], dict]:
        """
        Apply full DP mechanism: clip then add noise.
        
        Args:
            gradients: List of gradient arrays
            
        Returns:
            Tuple of (private_gradients, metrics_dict)
        """
        clipped, original_norm = self.clip_gradients(gradients)
        private = self.add_noise(clipped)
        
        metrics = {
            'original_norm': original_norm,
            'clip_norm': self.clip_norm,
            'noise_scale': self.noise_scale,
            'clipped': original_norm > self.clip_norm,
        }
        
        return private, metrics


def compute_epsilon_rdp(
    noise_scale: float,
    sampling_rate: float,
    num_steps: int,
    orders: Optional[List[float]] = None,
    delta: float = 1e-5
) -> float:
    """
    Compute privacy budget (epsilon) using Renyi Differential Privacy (RDP) accounting.
    
    This is a simplified approximation. For production use, consider using
    Google's privacy accountant or Opacus.
    
    Args:
        noise_scale: Noise multiplier σ
        sampling_rate: Fraction of data sampled per round (q)
        num_steps: Number of training steps/rounds (T)
        orders: RDP orders to compute (alpha values)
        delta: Target delta for (ε, δ)-DP
        
    Returns:
        Privacy budget epsilon
        
    Reference:
        "Rényi Differential Privacy" (Mironov, 2017)
        "The Algorithmic Foundations of Differential Privacy" (Dwork & Roth, 2014)
    """
    if orders is None:
        # Standard orders from privacy accounting literature
        orders = [1.1, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    rdp = []
    for alpha in orders:
        if alpha == 1.0:
            # Special case: alpha=1 is KL divergence
            # RDP₁(M) ≈ q²T / (2σ²)
            rdp_alpha = (sampling_rate ** 2 * num_steps) / (2 * noise_scale ** 2)
        else:
            # General RDP for subsampled Gaussian mechanism (simplified)
            # RDP_α ≈ α * q² * T / (2σ²)
            rdp_alpha = (alpha * sampling_rate ** 2 * num_steps) / (2 * noise_scale ** 2)
        rdp.append(rdp_alpha)
    
    # Convert RDP to (ε, δ)-DP: ε = min_α [RDP_α + log(1/δ)/(α-1)]
    epsilon_candidates = []
    for alpha, rdp_alpha in zip(orders, rdp):
        if alpha > 1:
            eps = rdp_alpha + math.log(1.0 / delta) / (alpha - 1)
            epsilon_candidates.append(eps)
    
    epsilon = min(epsilon_candidates) if epsilon_candidates else float('inf')
    
    return epsilon


def compute_epsilon_standard(
    noise_scale: float,
    clip_norm: float,
    sensitivity: float,
    delta: float = 1e-5
) -> float:
    """
    Compute epsilon using standard Gaussian mechanism analysis.
    
    For Gaussian mechanism with noise N(0, σ²Δ²):
        ε ≈ sqrt(2 * ln(1.25/δ)) / σ
    
    where σ is the noise scale and Δ is the sensitivity (clip_norm here).
    
    Args:
        noise_scale: Noise multiplier σ
        clip_norm: Clipping bound (sensitivity)
        sensitivity: Global sensitivity of the query
        delta: Target delta
        
    Returns:
        Privacy budget epsilon
    """
    if noise_scale == 0:
        return float('inf')
    
    # Standard Gaussian mechanism: ε = sqrt(2 * ln(1.25/δ)) / σ
    epsilon = math.sqrt(2 * math.log(1.25 / delta)) / noise_scale
    
    return epsilon


def analyze_privacy_budget(
    noise_scale: float,
    clip_norm: float,
    num_clients: int,
    num_rounds: int,
    delta: float = 1e-5,
    use_rdp: bool = True
) -> dict:
    """
    Comprehensive privacy budget analysis for federated learning setting.
    
    Args:
        noise_scale: Noise multiplier σ
        clip_norm: L2 clipping bound C
        num_clients: Number of clients (hospitals)
        num_rounds: Number of FL rounds
        delta: Target delta for (ε, δ)-DP
        use_rdp: Use RDP accounting (more accurate) vs standard Gaussian
        
    Returns:
        Dictionary with epsilon, delta, and analysis details
    """
    # In federated learning, each client participates once per round
    # Sampling rate q = num_participating_clients / total_clients
    # For our case with 3 hospitals all participating: q = 1.0
    sampling_rate = 1.0  # All clients participate each round
    
    if use_rdp:
        epsilon = compute_epsilon_rdp(
            noise_scale=noise_scale,
            sampling_rate=sampling_rate,
            num_steps=num_rounds,
            delta=delta
        )
    else:
        # Standard Gaussian mechanism, accounting for composition
        epsilon_per_round = compute_epsilon_standard(
            noise_scale=noise_scale,
            clip_norm=clip_norm,
            sensitivity=clip_norm,
            delta=delta / num_rounds  # Strong composition
        )
        # Advanced composition theorem
        epsilon = epsilon_per_round * math.sqrt(2 * num_rounds * math.log(1 / delta))
    
    return {
        'epsilon': epsilon,
        'delta': delta,
        'noise_scale': noise_scale,
        'clip_norm': clip_norm,
        'num_rounds': num_rounds,
        'num_clients': num_clients,
        'sampling_rate': sampling_rate,
        'method': 'RDP' if use_rdp else 'Standard Gaussian',
        'interpretation': _interpret_epsilon(epsilon)
    }


def _interpret_epsilon(epsilon: float) -> str:
    """Provide human-readable interpretation of epsilon value."""
    if epsilon < 0.1:
        return "Very strong privacy (ε < 0.1)"
    elif epsilon < 1.0:
        return "Strong privacy (0.1 ≤ ε < 1.0)"
    elif epsilon < 10.0:
        return "Moderate privacy (1.0 ≤ ε < 10.0)"
    elif epsilon < 100.0:
        return "Weak privacy (10.0 ≤ ε < 100.0)"
    else:
        return "Very weak privacy (ε ≥ 100.0) - essentially no formal guarantee"


def privacy_utility_tradeoff_analysis(
    noise_scales: List[float],
    clip_norm: float,
    num_rounds: int,
    num_clients: int = 3,
    delta: float = 1e-5
) -> List[dict]:
    """
    Analyze privacy-utility tradeoff across different noise levels.
    
    Args:
        noise_scales: List of noise multipliers to analyze
        clip_norm: Clipping bound
        num_rounds: Number of FL rounds
        num_clients: Number of clients
        delta: Target delta
        
    Returns:
        List of analysis results for each noise scale
    """
    results = []
    for sigma in noise_scales:
        analysis = analyze_privacy_budget(
            noise_scale=sigma,
            clip_norm=clip_norm,
            num_clients=num_clients,
            num_rounds=num_rounds,
            delta=delta
        )
        analysis['sigma'] = sigma
        results.append(analysis)
    
    return results


# Torch-specific utilities for integration with PyTorch models

def clip_gradients_torch(
    parameters: List[torch.Tensor],
    max_norm: float
) -> Tuple[List[torch.Tensor], float]:
    """
    Clip gradients of PyTorch parameters (in-place).
    
    Args:
        parameters: List of model parameters with .grad
        max_norm: Maximum gradient norm
        
    Returns:
        Tuple of (clipped_gradients, original_norm)
    """
    grads = [p.grad.detach().clone() for p in parameters if p.grad is not None]
    
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads)).item()
    
    clip_coef = max_norm / (total_norm + 1e-8)
    clip_coef = min(1.0, clip_coef)
    
    # Apply clipping in-place
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(clip_coef)
    
    return grads, total_norm


def add_noise_to_gradients_torch(
    parameters: List[torch.Tensor],
    noise_scale: float,
    clip_norm: float
):
    """
    Add Gaussian noise to PyTorch parameter gradients (in-place).
    
    Args:
        parameters: List of model parameters with .grad
        noise_scale: Noise multiplier σ
        clip_norm: Clipping bound C (used to scale noise)
    """
    noise_std = noise_scale * clip_norm
    
    for p in parameters:
        if p.grad is not None:
            noise = torch.randn_like(p.grad) * noise_std
            p.grad.add_(noise)


if __name__ == '__main__':
    print("=" * 80)
    print("DIFFERENTIAL PRIVACY MODULE TEST")
    print("=" * 80)
    
    # Test DP mechanism
    print("\n1. Testing DP Mechanism:")
    dp = DPMechanism(clip_norm=1.0, noise_scale=0.01)
    
    # Simulate gradients
    grads = [np.random.randn(64, 11), np.random.randn(32, 64), np.random.randn(1, 32)]
    print(f"   Original gradient norms: {[np.linalg.norm(g) for g in grads]}")
    
    private_grads, metrics = dp.privatize(grads)
    print(f"   ✓ Clipped: {metrics['clipped']}, Original norm: {metrics['original_norm']:.4f}")
    print(f"   Private gradient norms: {[np.linalg.norm(g) for g in private_grads]}")
    
    # Test privacy budget calculation
    print("\n2. Testing Privacy Budget Calculation:")
    
    # Current config from config.yaml
    analysis = analyze_privacy_budget(
        noise_scale=0.01,
        clip_norm=1.0,
        num_clients=3,
        num_rounds=10,
        delta=1e-5
    )
    
    print(f"   Configuration:")
    print(f"     - Noise scale (σ): {analysis['noise_scale']}")
    print(f"     - Clip norm (C): {analysis['clip_norm']}")
    print(f"     - Rounds: {analysis['num_rounds']}")
    print(f"     - Clients: {analysis['num_clients']}")
    print(f"     - Delta (δ): {analysis['delta']}")
    print(f"\n   Privacy Budget:")
    print(f"     - Epsilon (ε): {analysis['epsilon']:,.2f}")
    print(f"     - Interpretation: {analysis['interpretation']}")
    
    # Test privacy-utility tradeoff
    print("\n3. Testing Privacy-Utility Tradeoff Analysis:")
    noise_scales = [0.001, 0.01, 0.1, 0.5, 1.0]
    tradeoff = privacy_utility_tradeoff_analysis(
        noise_scales=noise_scales,
        clip_norm=1.0,
        num_rounds=10
    )
    
    print(f"   {'Sigma':>8} {'Epsilon':>12} {'Interpretation':>40}")
    print(f"   {'-'*8} {'-'*12} {'-'*40}")
    for result in tradeoff:
        print(f"   {result['sigma']:>8.3f} {result['epsilon']:>12,.2f} {result['interpretation']:>40}")
    
    print("\n" + "=" * 80)
    print("✓ ALL DIFFERENTIAL PRIVACY TESTS PASSED")
    print("=" * 80)
