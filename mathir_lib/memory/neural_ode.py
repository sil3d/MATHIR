"""
Neural ODE Memory — Continuous-time memory evolution.
Memory evolves smoothly via an ODE instead of discrete steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NeuralODEMemory(nn.Module):
    """
    Memory that evolves continuously via Neural ODE.

    Dynamics: dm/dt = f_theta(m, x, t)

    Integrated via Euler or RK4.
    Adaptive time stepping.
    """

    def __init__(self, capacity: int = 1000, feature_dim: int = 272,
                 dt: float = 0.1, max_steps: int = 10, method: str = "euler"):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.dt = dt
        self.max_steps = max_steps
        self.method = method

        # Memory store
        self.register_buffer("values", torch.zeros(capacity, feature_dim))
        self.register_buffer("ages", torch.zeros(capacity))  # Time since last update
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        # ODE function: dm/dt = f(m, x, t)
        self.ode_func = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def dynamics(self, m: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute dm/dt.

        Args:
            m: [B, D] current memory state
            x: [B, D] input (broadcasted)
            t: [B, 1] current time

        Returns:
            dm/dt: [B, D]
        """
        inp = torch.cat([m, x, t], dim=-1)
        return self.ode_func(inp)

    def euler_step(self, m: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                   dt: float) -> torch.Tensor:
        """Euler integration step."""
        return m + dt * self.dynamics(m, x, t)

    def rk4_step(self, m: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                 dt: float) -> torch.Tensor:
        """RK4 integration step."""
        k1 = self.dynamics(m, x, t)
        k2 = self.dynamics(m + 0.5 * dt * k1, x, t + 0.5 * dt)
        k3 = self.dynamics(m + 0.5 * dt * k2, x, t + 0.5 * dt)
        k4 = self.dynamics(m + dt * k3, x, t + dt)
        return m + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def evolve(self, memory: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
               n_steps: int) -> torch.Tensor:
        """
        Evolve memory by n_steps from current state.

        Args:
            memory: [B, D] current memory
            x: [B, D] input
            t: [B, 1] current time
            n_steps: number of integration steps

        Returns:
            Evolved memory: [B, D]
        """
        dt = self.dt
        m = memory
        for step in range(n_steps):
            t_current = t + step * dt
            if self.method == "euler":
                m = self.euler_step(m, x, t_current, dt)
            elif self.method == "rk4":
                m = self.rk4_step(m, x, t_current, dt)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        return m

    def store(self, features: torch.Tensor) -> None:
        """Store features with initial age 0."""
        with torch.no_grad():
            ptr = int(self.ptr.item() if torch.is_tensor(self.ptr) else self.ptr)
            count = int(self.count.item() if torch.is_tensor(self.count) else self.count)

            idx = ptr % self.capacity
            self.values[idx] = features.detach().mean(0)
            self.ages[idx] = 0.0

            self.ptr = torch.tensor((ptr + 1) % self.capacity, dtype=torch.long, device=self.values.device)
            self.count = torch.tensor(min(count + 1, self.capacity), dtype=torch.long, device=self.values.device)

    def age_memory(self, dt: float = 1.0) -> None:
        """Age all memories by dt (used between time steps)."""
        with torch.no_grad():
            self.ages += dt

    def retrieve(self, query: torch.Tensor, n_steps: int = 3) -> torch.Tensor:
        """
        Retrieve by evolving memory toward query.

        Args:
            query: [B, D]
            n_steps: number of ODE steps to evolve

        Returns:
            [B, D] evolved memory + query
        """
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        if count < 1:
            return query

        # Take the most recent memory
        m = self.values[:count].mean(dim=0, keepdim=True).expand(query.size(0), -1)
        x = query
        t = torch.zeros(query.size(0), 1, device=query.device)

        evolved = self.evolve(m, x, t, n_steps)
        return evolved + query  # Residual

    def get_stats(self) -> dict:
        """Get statistics."""
        count = self.count.item() if torch.is_tensor(self.count) else self.count
        return {
            "count": count,
            "mean_age": self.ages[:count].mean().item() if count > 0 else 0,
            "max_age": self.ages[:count].max().item() if count > 0 else 0,
            "method": self.method,
        }
