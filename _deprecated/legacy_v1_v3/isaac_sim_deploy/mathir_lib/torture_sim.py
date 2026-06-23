"""
MATHIR ULTIMATE COGNITIVE TORTURE SIMULATOR
===========================================
Version 2.0 - Reinforced Concrete
Designed to break doped LSTMs and validate MATHIR superiority
"""

import torch
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
from enum import Enum, auto
import random
from collections import deque

IMG_H, IMG_W = 84, 84

# ==================== NEW SYSTEMS ====================

class MemoryChallengeType(Enum):
    """Sadistic memory challenge types"""
    DELAYED_MATCH_TO_SAMPLE = auto()
    SEQUENTIAL_PATTERN = auto()
    WORKING_MEMORY_LOAD = auto()
    INTERFERENCE_TEST = auto()
    CONTEXT_SWITCHING = auto()
    RETROACTIVE_INTERFERENCE = auto()
    PROACTIVE_INTERFERENCE = auto()

class DistractionClass(Enum):
    """Perverse distraction classes"""
    SEMANTIC_INTERFERENCE = auto()  # Similar to signal but false
    ATTENTION_CAPTURE = auto()      # Very salient but useless
    COGNITIVE_OVERLOAD = auto()     # Multiple simultaneous distractions
    TEMPORAL_DISTRACTION = auto()   # Distractions at critical moment
    CONTEXTUAL_DECEPTION = auto()   # Distraction linked to context but deceptive

@dataclass
class Neuromodulation:
    """Simulates neuromodulators affecting learning"""
    dopamine: float = 1.0      # Reward/attention
    norepinephrine: float = 1.0 # Alertness/arousal
    acetylcholine: float = 1.0  # Plasticity/attention
    serotonin: float = 1.0      # Inhibition/stability

# ==================== NEW COMPONENTS ====================

class DynamicRoadSystem:
    """Dynamic road that evolves during the episode"""
    def __init__(self):
        self.segments = []
        self.active_obstacles = []
        self.construction_zones = []
        self.dynamic_changes = deque(maxlen=10)
        
    def generate_torture_road(self, difficulty: str, episode_seed: int):
        """Generates a road designed to maximize cognitive load"""
        np.random.seed(episode_seed)
        
        # Road with dynamic changes
        segments = []
        total_length = 500 if difficulty == 'master' else 300
        
        # Diabolical patterns
        patterns = [
            ('spiral', 80, 'high_curvature'),      # Spiral to disorient
            ('zigzag', 60, 'rapid_switching'),     # Rapid switching
            ('chicane', 40, 'working_memory'),     # Sequence to memorize
            ('maze', 100, 'decision_cluster'),     # Multiple close decisions
            ('illusion', 50, 'perceptual_trap'),   # Optical illusions
            ('vanishing', 30, 'memory_trap')       # Vanishing road
        ]
        
        current_pos = 0
        for pattern, length, challenge in patterns:
            segments.append({
                'type': pattern,
                'length': length * (2 if difficulty == 'master' else 1),
                'challenge': challenge,
                'start': current_pos,
                'end': current_pos + length,
                'properties': self._generate_pattern_properties(pattern)
            })
            current_pos += length
            
        self.segments = segments
        return segments
    
    def _generate_pattern_properties(self, pattern: str) -> Dict:
        """Generates specific pattern properties"""
        props = {}
        if pattern == 'spiral':
            props['curvature_fn'] = lambda t: 0.5 * math.sin(t * 0.2)
            props['width_variation'] = lambda t: 15 + 5 * math.sin(t * 0.3)
        elif pattern == 'zigzag':
            props['frequency'] = random.uniform(0.1, 0.3)
            props['amplitude'] = random.uniform(20, 40)
        elif pattern == 'chicane':
            props['sequence'] = [1, -1, 1, -1, 1]  # Left, right, left...
            props['timing'] = random.randint(10, 20)
        return props

class CognitiveTortureEngine:
    """Cognitive torture engine - generates impossible challenges"""
    def __init__(self):
        self.active_challenges = []
        self.interference_tasks = []
        self.memory_traps = []
        self.distraction_orchestrator = DistractionOrchestrator()
        
    def generate_torture_sequence(self, step: int, vehicle_state: Dict) -> List[Dict]:
        """Generates an adaptive torture sequence"""
        challenges = []
        
        # 1. Working memory challenges
        if step % 150 == 0:  # Every 150 steps
            challenges.append(self._create_working_memory_challenge(step))
        
        # 2. Proactive/Retroactive interference
        if 100 < step < 300:
            challenges.append(self._create_interference_challenge(step))
        
        # 3. Temporal traps
        if random.random() < 0.01:  # 1% chance per step
            challenges.append(self._create_temporal_trap(step))
        
        # 4. Orchestrated distractions
        distractions = self.distraction_orchestrator.orchestrate_distractions(
            step, vehicle_state
        )
        challenges.extend(distractions)
        
        return challenges
    
    def _create_working_memory_challenge(self, step: int) -> Dict:
        """Creates a sadistic working memory challenge"""
        return {
            'type': 'working_memory',
            'subtype': random.choice(['n_back', 'complex_span', 'updating']),
            'load': random.randint(3, 7),  # 3-7 items to retain
            'duration': random.randint(50, 150),
            'interference': True,
            'reward': 25.0 if random.random() > 0.5 else -25.0
        }
    
    def _create_interference_challenge(self, step: int) -> Dict:
        """Creates an interference challenge"""
        return {
            'type': 'interference',
            'subtype': random.choice(['proactive', 'retroactive', 'both']),
            'strength': random.uniform(0.5, 1.0),
            'duration': random.randint(30, 100)
        }
    
    def _create_temporal_trap(self, step: int) -> Dict:
        """Creates a temporal trap"""
        return {
            'type': 'temporal_trap',
            'mechanism': random.choice(['delayed_gratification', 'premature_response']),
            'time_window': (step + random.randint(20, 80), step + random.randint(100, 200)),
            'penalty': -30.0
        }

class DistractionOrchestrator:
    """Orchestrates distractions to maximize interference"""
    def __init__(self):
        self.distraction_sequences = []
        self.semantic_interferers = []
        
    def orchestrate_distractions(self, step: int, vehicle_state: Dict) -> List[Dict]:
        """Orchestrates distractions strategically"""
        distractions = []
        
        # 1. Semantic interference (similar to signal but false)
        if step % 120 == 0:
            distractions.append(self._create_semantic_interference(step))
        
        # 2. Attention capture (very salient but useless)
        if random.random() < 0.02:
            distractions.append(self._create_attention_capture(step))
        
        # 3. Cognitive overload (multiple distractions)
        if 200 < step < 400 and step % 80 == 0:
            distractions.extend(self._create_cognitive_overload(step))
        
        # 4. Temporal distraction (at critical moment)
        if self._is_critical_moment(step, vehicle_state):
            distractions.append(self._create_temporal_distraction(step))
        
        return distractions
    
    def _create_semantic_interference(self, step: int) -> Dict:
        """Creates a distraction semantically similar to the signal"""
        fake_signals = [
            {'type': 'turn_hint', 'value': 1, 'color': 0.5, 'position': 'left'},  # Blue but false
            {'type': 'turn_hint', 'value': 2, 'color': 0.8, 'position': 'right'}, # Red but false
            {'type': 'speed_limit', 'value': 0, 'color': 0.3, 'position': 'center'}, # Deceptive
        ]
        signal = random.choice(fake_signals)
        signal['step'] = step
        signal['duration'] = random.randint(10, 30)
        signal['salience'] = random.uniform(0.7, 0.9)
        return signal
    
    def _create_attention_capture(self, step: int) -> Dict:
        """Creates a distraction that strongly captures attention"""
        return {
            'type': 'attention_capture',
            'subtype': random.choice(['flash', 'movement', 'contrast']),
            'intensity': random.uniform(0.8, 1.0),
            'duration': random.randint(5, 15),
            'position': (random.randint(0, IMG_W), random.randint(0, IMG_H))
        }
    
    def _create_cognitive_overload(self, step: int) -> List[Dict]:
        """Creates cognitive overload with multiple distractions"""
        distractions = []
        for _ in range(random.randint(3, 6)):
            distractions.append({
                'type': 'overload_element',
                'subtype': random.choice(['sign', 'object', 'texture']),
                'salience': random.uniform(0.4, 0.7),
                'duration': random.randint(20, 40),
                'position': (random.randint(0, IMG_W), random.randint(0, IMG_H))
            })
        return distractions
    
    def _create_temporal_distraction(self, step: int) -> Dict:
        """Creates a distraction at a critical moment"""
        return {
            'type': 'temporal_distraction',
            'timing': 'critical',
            'salience': 0.9,
            'duration': 10,
            'interferes_with': 'memory_retrieval'
        }
    
    def _is_critical_moment(self, step: int, vehicle_state: Dict) -> bool:
        """Detects critical moments for memory"""
        # Moments where the agent is about to make a decision
        steering = abs(vehicle_state.get('steering', 0))
        velocity = vehicle_state.get('velocity', 0)
        
        # Critical if: moderate speed + active steering + near intersection
        if 5 < velocity < 15 and steering > 0.2:
            # Check if near an intersection (simplified)
            road_pos = vehicle_state.get('x', 0)
            return any(90 < (road_pos % 200) < 110 for _ in range(3))
        return False

# ==================== MAIN SIMULATOR ====================

@dataclass
class UltimateVehicleState:
    """Extended vehicle state with dynamic parameters"""
    x: float = 0.0
    y: float = 0.0
    velocity: float = 0.0
    heading: float = 0.0
    steering: float = 0.0
    acceleration: float = 0.0
    slip_angle: float = 0.0
    traction: float = 1.0
    brake_temp: float = 25.0  # Brake temperature
    tire_wear: float = 1.0    # Tire wear (1.0 = new)
    
@dataclass  
class TortureEpisodeConfig:
    """Configuration for a torture episode"""
    difficulty: str = 'torture'
    memory_challenges: int = 5
    distraction_density: float = 0.8
    physics_realism: float = 0.9
    time_pressure: bool = True
    adaptive_difficulty: bool = True
    seed: int = None

class UltimateDrivingSimulator:
    """Ultimate simulator version - designed to break LSTMs"""
    
    def __init__(self, device='cuda', config: Any = None):
        self.device = device
        
        # Handle dictionary config (from YAML) or dataclass
        if isinstance(config, dict):
            # Override Global Constants if provided in YAML
            if 'perception' in config and 'camera_resolution' in config['perception']:
                global IMG_H, IMG_W
                IMG_H = config['perception']['camera_resolution'][0]
                IMG_W = config['perception']['camera_resolution'][1]
                print(f"🔧 TortureSim Configured Resolution: {IMG_W}x{IMG_H}")
            
            # Map YAML config to TortureEpisodeConfig if present
            env_conf = config.get('env_config', {})
            self.config = TortureEpisodeConfig(
                difficulty=env_conf.get('difficulty', 'torture'),
                # Add other mappings as needed
            )
            # Store raw config too
            self.raw_config = config
        else:
            self.config = config or TortureEpisodeConfig()
            self.raw_config = {}
        
        # Deterministic seed for reproducibility
        self.master_seed = self.config.seed or random.randint(0, 1000000)
        self.rng = np.random.RandomState(self.master_seed)
        
        # Main systems
        self.road_system = DynamicRoadSystem()
        self.torture_engine = CognitiveTortureEngine()
        self.neuromodulation = Neuromodulation()
        
        # Simulator state
        self.state = UltimateVehicleState()
        self.time_step = 0
        self.episode_score = 0.0
        self.cognitive_load = 0.0
        self.stress_level = 0.0
        
        # Advanced episodic memory
        self.memory_buffer = deque(maxlen=100)  # Recent memory buffer
        self.long_term_memory = []              # Long term memory
        self.active_memory_items = []           # Active items in working memory
        
        # Distractions and challenges
        self.active_distractions = []
        self.active_challenges = []
        self.pending_decisions = deque(maxlen=10)
        
        # Dynamic conditions
        self.weather_conditions = self._generate_weather_sequence()
        self.traffic_patterns = self._generate_traffic_patterns()
        self.time_of_day = 0.0
        self.road_conditions = self._generate_road_conditions()
        
        # Performance metrics
        self.performance_metrics = {
            'memory_accuracy': 0.0,
            'attention_efficiency': 0.0,
            'decision_quality': 0.0,
            'adaptation_speed': 0.0
        }
        
        self.reset()
    
    def _generate_weather_sequence(self) -> List[Dict]:
        """Generates dynamic weather sequence"""
        weather_types = ['clear', 'rain', 'fog', 'storm', 'snow']
        sequence = []
        
        # Weather changes during episode
        for i in range(5):  # 5 changes
            sequence.append({
                'type': self.rng.choice(weather_types),
                'start_step': self.rng.randint(0, 400),
                'duration': self.rng.randint(50, 150),
                'intensity': self.rng.uniform(0.3, 1.0)
            })
        
        return sequence
    
    def _generate_traffic_patterns(self) -> List[Dict]:
        """Generates complex traffic patterns"""
        patterns = []
        traffic_types = ['flowing', 'congested', 'erratic', 'platoon']
        
        for _ in range(3):
            patterns.append({
                'type': self.rng.choice(traffic_types),
                'density': self.rng.uniform(0.3, 0.9),
                'behavior': self.rng.choice(['cooperative', 'aggressive', 'unpredictable']),
                'spawn_rate': self.rng.uniform(0.1, 0.5)
            })
        
        return patterns
    
    def _generate_road_conditions(self) -> Dict:
        """Generates variable road conditions"""
        return {
            'friction': self.rng.uniform(0.3, 0.9),
            'damage': self.rng.uniform(0.0, 0.5),
            'obstacles': self.rng.randint(0, 10),
            'visibility': self.rng.uniform(0.5, 1.0)
        }
    
    def reset(self):
        """Resets the environment with a new sadistic configuration"""
        self.state = UltimateVehicleState()
        self.time_step = 0
        self.episode_score = 0.0
        self.cognitive_load = 0.0
        self.stress_level = 0.0
        
        # Generate a new torture road
        self.road_system.generate_torture_road(
            self.config.difficulty, 
            self.master_seed + self.time_step
        )
        
        # Reset memories
        self.memory_buffer.clear()
        self.long_term_memory.clear()
        self.active_memory_items.clear()
        
        # Reset challenges and distractions
        self.active_distractions.clear()
        self.active_challenges.clear()
        self.pending_decisions.clear()
        
        # Update conditions
        self.time_of_day = self.rng.random()
        self.road_conditions = self._generate_road_conditions()
        
        # Reset neuromodulation
        self.neuromodulation = Neuromodulation()
        
        return self._get_observation()
    
    def step(self, action: torch.Tensor):
        """Executes a simulation step with hyper-realistic physics"""
        dt = 0.1
        
        # Extract action
        steer_cmd = float(action[0, 0].item())
        accel_cmd = float(action[0, 1].item())
        brake_cmd = float(action[0, 2].item()) if action.shape[1] > 2 else 0.0
        
        # ========== ULTRA-REALISTIC PHYSICS ==========
        wheelbase = 2.5
        max_steer = 0.5
        
        # 1. Brake temperature effects
        brake_efficiency = max(0.3, 1.0 - (self.state.brake_temp - 100) / 400)
        self.state.brake_temp += abs(brake_cmd) * 2.0 - 0.5  # Cooling
        
        # 2. Tire wear
        tire_grip = self.state.tire_wear * self.road_conditions['friction']
        slip_reduction = 1.0 - (1.0 - self.state.tire_wear) * 0.5
        
        # 3. Steering model with mechanical play
        steering_play = 0.05 * (1.0 - tire_grip)
        effective_steer_cmd = steer_cmd + self.rng.uniform(-steering_play, steering_play)
        
        # 4. Dynamics with complex inertia
        steering_inertia = 0.3
        self.state.steering += (effective_steer_cmd * max_steer - self.state.steering) * steering_inertia
        
        # 5. Traction model with drift
        slip_ratio = self._calculate_advanced_slip()
        self.state.slip_angle = slip_ratio * (1 if self.rng.random() > 0.5 else -1)
        
        # 6. Acceleration with power limit
        power_limit = 100.0  # kW
        current_power = abs(accel_cmd) * power_limit * tire_grip
        
        # 7. Drag (aerodynamics, friction)
        drag = 0.5 * 0.3 * 1.2 * self.state.velocity ** 2  # Aerodynamic force
        rolling_resistance = 0.01 * self.state.velocity
        
        # 8. Net force calculation
        engine_force = current_power / max(1.0, self.state.velocity)
        resistance = drag + rolling_resistance
        
        net_acceleration = (engine_force - resistance - brake_cmd * brake_efficiency * 10.0) / 1500.0
        
        self.state.velocity = max(0.0, self.state.velocity + net_acceleration * dt)
        self.state.acceleration = net_acceleration
        
        # 9. Position update with drift
        effective_heading = self.state.heading + self.state.slip_angle
        
        self.state.heading += (self.state.velocity / wheelbase) * math.tan(self.state.steering) * dt
        self.state.x += self.state.velocity * math.cos(effective_heading) * dt
        self.state.y += self.state.velocity * math.sin(effective_heading) * dt
        
        # 10. Progressive tire wear
        wear_rate = 0.0001 * (abs(self.state.steering) + self.state.velocity / 30.0)
        self.state.tire_wear = max(0.3, self.state.tire_wear - wear_rate)
        
        self.time_step += 1
        
        # ========== ADVANCED COGNITIVE MANAGEMENT ==========
        self._update_cognitive_systems()
        self._manage_torture_challenges()
        
        # ========== PERFORMANCE EVALUATION ==========
        memory_reward = self._evaluate_memory_performance()
        attention_reward = self._evaluate_attention_performance()
        decision_reward = self._evaluate_decision_performance()
        
        # ========== OBSERVATION AND REWARD ==========
        observation = self._get_observation()
        total_reward = self._calculate_torture_reward(
            memory_reward, attention_reward, decision_reward
        )
        
        self.episode_score += total_reward.item()
        
        # ========== EPISODE END ==========
        done = self._check_episode_end()
        
        # Info dictionary with detailed metrics
        info = {
            'episode_score': self.episode_score,
            'cognitive_load': self.cognitive_load,
            'stress_level': self.stress_level,
            'memory_accuracy': self.performance_metrics['memory_accuracy'],
            'attention_efficiency': self.performance_metrics['attention_efficiency'],
            'vehicle_state': {
                'velocity': self.state.velocity,
                'steering': self.state.steering,
                'tire_wear': self.state.tire_wear,
                'brake_temp': self.state.brake_temp
            }
        }
        
        return observation, total_reward, done, info
    
    def _calculate_advanced_slip(self) -> float:
        """Calculates slip with advanced physical model"""
        if self.state.velocity < 0.1:
            return 0.0
        
        # Simplified Pacejka model
        B = 10.0  # Stiffness factor
        C = 1.9   # Shape factor
        D = 1.0   # Peak factor
        
        slip_angle = math.atan(3 * math.tan(self.state.steering) / 2)  # Slip angle
        
        # Load effect
        load = 1500.0  # kg
        load_effect = load / 1500.0
        
        # Temperature effect
        temp_effect = 1.0 - abs(self.state.brake_temp - 75) / 100.0
        
        # Wear effect
        wear_effect = self.state.tire_wear ** 0.5
        
        # Surface effect
        surface = self.road_conditions['friction']
        
        # Lateral force calculation
        lateral_force = D * math.sin(C * math.atan(B * slip_angle))
        lateral_force *= load_effect * temp_effect * wear_effect * surface
        
        # Conversion to slip ratio
        slip_ratio = lateral_force / max(0.1, self.state.velocity)
        
        return slip_ratio
    
    def _update_cognitive_systems(self):
        """Updates all cognitive systems"""
        # 1. Update neuromodulation based on performance
        self._update_neuromodulation()
        
        # 2. Generate new torture challenges
        vehicle_state_dict = {
            'x': self.state.x,
            'y': self.state.y,
            'velocity': self.state.velocity,
            'steering': self.state.steering,
            'heading': self.state.heading
        }
        
        new_challenges = self.torture_engine.generate_torture_sequence(
            self.time_step, vehicle_state_dict
        )
        self.active_challenges.extend(new_challenges)
        
        # 3. Manage working memory
        self._manage_working_memory()
        
        # 4. Calculate cognitive load
        self.cognitive_load = self._calculate_cognitive_load()
        self.stress_level = self._calculate_stress_level()
    
    def _update_neuromodulation(self):
        """Updates neuromodulator levels"""
        # Dopamine based on recent reward
        reward_window = list(self.memory_buffer)[-10:] if self.memory_buffer else []
        avg_reward = np.mean([r for _, r in reward_window]) if reward_window else 0.0
        
        self.neuromodulation.dopamine = 0.5 + 0.5 * math.tanh(avg_reward)
        
        # Norepinephrine based on novelty/uncertainty
        novelty = len(self.active_challenges) / 10.0
        self.neuromodulation.norepinephrine = 0.3 + 0.7 * novelty
        
        # Acetylcholine based on attention demand
        attention_demand = self.cognitive_load
        self.neuromodulation.acetylcholine = 0.4 + 0.6 * attention_demand
        
        # Serotonin based on stability
        stability = 1.0 - (len(self.active_distractions) / 20.0)
        self.neuromodulation.serotonin = max(0.3, stability)
    
    def _manage_working_memory(self):
        """Manages working memory with interference"""
        # Apply forgetting based on time and interference
        decay_rate = 0.95  # Decay rate per step
        
        for i, item in enumerate(self.active_memory_items):
            # Temporal decay
            if 'strength' in item:
                item['strength'] *= decay_rate
            
            # Distraction interference
            interference = len(self.active_distractions) * 0.05
            if 'strength' in item:
                item['strength'] *= (1.0 - interference)
            
            # Remove weak items
            if 'strength' in item and item['strength'] < 0.1:
                self.active_memory_items.pop(i)
                break
    
    def _calculate_cognitive_load(self) -> float:
        """Calculates current cognitive load"""
        load = 0.0
        
        # Working memory load
        load += len(self.active_memory_items) * 0.2
        
        # Attention load (distractions)
        load += len(self.active_distractions) * 0.15
        
        # Decision load
        load += len(self.pending_decisions) * 0.25
        
        # Sensory processing load
        load += len(self.active_challenges) * 0.1
        
        return min(1.0, load)
    
    def _calculate_stress_level(self) -> float:
        """Calculates stress level"""
        stress = 0.0
        
        # Stress due to speed
        stress += min(1.0, self.state.velocity / 30.0) * 0.3
        
        # Stress due to complexity
        stress += self.cognitive_load * 0.4
        
        # Stress due to conditions
        weather_stress = 0.0
        current_weather = self._get_current_weather()
        if current_weather['type'] in ['storm', 'snow']:
            weather_stress = current_weather['intensity'] * 0.3
        
        stress += weather_stress
        
        return min(1.0, stress)
    
    def _get_current_weather(self) -> Dict:
        """Returns current weather conditions"""
        for weather in self.weather_conditions:
            if weather['start_step'] <= self.time_step <= weather['start_step'] + weather['duration']:
                return weather
        return {'type': 'clear', 'intensity': 0.0}
    
    def _manage_torture_challenges(self):
        """Manages active torture challenges"""
        # Apply active challenges
        for challenge in list(self.active_challenges):
            # Check duration
            if 'duration' in challenge:
                challenge['duration'] -= 1
                if challenge['duration'] <= 0:
                    self.active_challenges.remove(challenge)
                    continue
            
            # Apply challenge effects
            if challenge['type'] == 'working_memory':
                self._apply_working_memory_challenge(challenge)
            elif challenge['type'] == 'interference':
                self._apply_interference_challenge(challenge)
            elif challenge['type'] == 'temporal_trap':
                self._apply_temporal_trap(challenge)
    
    def _apply_working_memory_challenge(self, challenge: Dict):
        """Applies a working memory challenge"""
        # Add items to working memory
        if 'load' in challenge and len(self.active_memory_items) < challenge['load']:
            new_item = {
                'id': len(self.active_memory_items),
                'type': 'memory_challenge',
                'value': self.rng.randint(0, 10),
                'strength': 1.0,
                'challenge_id': id(challenge)
            }
            self.active_memory_items.append(new_item)
    
    def _apply_interference_challenge(self, challenge: Dict):
        """Applies an interference challenge"""
        if challenge['subtype'] == 'proactive':
            # Proactive interference: old memories interfere with new ones
            for item in self.active_memory_items:
                if 'strength' in item:
                    item['strength'] *= (1.0 - challenge['strength'] * 0.1)
        
        elif challenge['subtype'] == 'retroactive':
            # Retroactive interference: new memories interfere with old ones
            # Harder retrieval of old items
            for item in self.active_memory_items:
                if 'strength' in item and item.get('age', 0) > 10:
                    item['strength'] *= (1.0 - challenge['strength'] * 0.15)
    
    def _apply_temporal_trap(self, challenge: Dict):
        """Applies a temporal trap"""
        time_window = challenge.get('time_window', (0, 0))
        
        if time_window[0] <= self.time_step <= time_window[1]:
            # Within time window
            if challenge['mechanism'] == 'premature_response':
                # Penalize premature responses
                if len(self.pending_decisions) > 0:
                    # Agent makes decisions too early
                    self.episode_score += challenge.get('penalty', -10.0)
    
    def _evaluate_memory_performance(self) -> float:
        """Evaluates memory performance"""
        if not self.active_memory_items:
            return 0.0
        
        # Calculate average memory strength
        total_strength = sum(item.get('strength', 0.0) for item in self.active_memory_items)
        avg_strength = total_strength / len(self.active_memory_items)
        
        # Update metric
        self.performance_metrics['memory_accuracy'] = 0.9 * self.performance_metrics['memory_accuracy'] + 0.1 * avg_strength
        
        # Reward proportional to memory strength
        return avg_strength * 5.0
    
    def _evaluate_attention_performance(self) -> float:
        """Evaluates attention performance"""
        # Calculate attention efficiency
        total_distractions = len(self.active_distractions)
        irrelevant_distractions = sum(1 for d in self.active_distractions if d.get('relevance', 0) < 0.2)
        
        if total_distractions > 0:
            attention_efficiency = 1.0 - (irrelevant_distractions / total_distractions)
        else:
            attention_efficiency = 1.0
        
        # Update metric
        self.performance_metrics['attention_efficiency'] = attention_efficiency
        
        # Reward efficient attention
        return attention_efficiency * 3.0
    
    def _evaluate_decision_performance(self) -> float:
        """Evaluates decision quality"""
        if not self.pending_decisions:
            return 0.0
        
        # Evaluate last decision
        last_decision = self.pending_decisions[-1] if self.pending_decisions else None
        
        if last_decision:
            # Quality based on timing and relevance
            timing_score = 1.0 - min(1.0, abs(last_decision['timing'] - self.time_step) / 50.0)
            relevance_score = last_decision.get('relevance', 0.5)
            
            decision_quality = (timing_score + relevance_score) / 2.0
            
            # Update metric
            self.performance_metrics['decision_quality'] = decision_quality
            
            return decision_quality * 4.0
        
        return 0.0
    
    def _get_observation(self) -> Dict[str, torch.Tensor]:
        """Generates super-complex observation"""
        # Main image with advanced effects
        img = torch.zeros(1, 1, IMG_H, IMG_W, device=self.device)
        
        # ========== ADVANCED ROAD DRAWING ==========
        current_segment = self._get_current_road_segment()
        
        if current_segment:
            # Draw according to segment type
            if current_segment['type'] == 'spiral':
                img = self._draw_spiral_road(img, current_segment)
            elif current_segment['type'] == 'zigzag':
                img = self._draw_zigzag_road(img, current_segment)
            elif current_segment['type'] == 'chicane':
                img = self._draw_chicane_road(img, current_segment)
            else:
                img = self._draw_standard_road(img)
        
        # ========== COGNITIVE SIGNAL (if present) ==========
        for memory_item in self.active_memory_items:
            if memory_item['type'] == 'memory_challenge':
                # Display memory signal
                self._draw_memory_signal(img, memory_item)
        
        # ========== DISTRACTIONS ==========
        for distraction in self.active_distractions:
            self._draw_distraction(img, distraction)
        
        # ========== WEATHER EFFECTS ==========
        current_weather = self._get_current_weather()
        img = self._apply_weather_effects(img, current_weather)
        
        # ========== NOISE AND PERTURBATIONS ==========
        img = self._apply_sensor_noise(img)
        
        # MATHIR V5 expects RGB (3 channels), assume Simulator is Grayscale -> RGB
        img = img.repeat(1, 3, 1, 1)
        
        # ========== EXTENDED STATE VECTOR ==========
        state_vec = torch.tensor([[
            # Vehicle data (normalized)
            self.state.velocity / 30.0,
            self.state.steering / 0.5,
            math.cos(self.state.heading),
            math.sin(self.state.heading),
            self.state.acceleration / 10.0,
            self.state.slip_angle / 0.3,
            self.state.tire_wear,
            self.state.brake_temp / 200.0,
            
            # Cognitive data
            self.cognitive_load,
            self.stress_level,
            len(self.active_memory_items) / 10.0,
            len(self.active_distractions) / 20.0,
            
            # Environmental data
            self.time_of_day,
            current_weather['intensity'],
            self.road_conditions['friction'],
            self.road_conditions['visibility'],
            
            # Neuromodulation
            self.neuromodulation.dopamine,
            self.neuromodulation.norepinephrine,
            self.neuromodulation.acetylcholine,
            self.neuromodulation.serotonin,
            
            # Performance
            self.performance_metrics['memory_accuracy'],
            self.performance_metrics['attention_efficiency'],
            self.performance_metrics['decision_quality']
        ]], device=self.device)
        
        return {
            'camera': img,
            'state': state_vec,
            'cognitive_load': torch.tensor([self.cognitive_load], device=self.device),
            'stress_level': torch.tensor([self.stress_level], device=self.device),
            'neuromodulation': torch.tensor([
                [self.neuromodulation.dopamine,
                 self.neuromodulation.norepinephrine,
                 self.neuromodulation.acetylcholine,
                 self.neuromodulation.serotonin]
            ], device=self.device)
        }
    
    def _draw_spiral_road(self, img: torch.Tensor, segment: Dict) -> torch.Tensor:
        """Draws a spiral road"""
        curvature_fn = segment['properties'].get('curvature_fn', lambda t: 0)
        width_fn = segment['properties'].get('width_variation', lambda t: 20)
        
        t = self.time_step * 0.01
        curvature = curvature_fn(t)
        road_width = width_fn(t)
        
        for y in range(IMG_H):
            center_x = IMG_W // 2 + curvature * y - self.state.steering * 20
            left = int(center_x - road_width / 2)
            right = int(center_x + road_width / 2)
            
            if 0 <= left < IMG_W:
                img[0, 0, y, max(0, left-2):left+2] = 0.8
            if 0 <= right < IMG_W:
                img[0, 0, y, max(0, right-2):right+2] = 0.8
        
        return img
    
    def _draw_zigzag_road(self, img: torch.Tensor, segment: Dict) -> torch.Tensor:
        """Draws a zigzag road"""
        freq = segment['properties'].get('frequency', 0.2)
        amp = segment['properties'].get('amplitude', 30)
        
        for y in range(IMG_H):
            # Sinusoidal oscillation
            offset = amp * math.sin(y * freq * 0.1)
            center_x = IMG_W // 2 + offset - self.state.steering * 15
            
            # Fixed width road
            left = int(center_x - 10)
            right = int(center_x + 10)
            
            if 0 <= left < IMG_W:
                img[0, 0, y, left:left+2] = 0.7
            if 0 <= right < IMG_W:
                img[0, 0, y, right-2:right] = 0.7
        
        return img
    
    def _draw_chicane_road(self, img: torch.Tensor, segment: Dict) -> torch.Tensor:
        """Draws a chicane (sequence of tight turns)"""
        sequence = segment['properties'].get('sequence', [1, -1])
        timing = segment['properties'].get('timing', 15)
        
        # Determine current phase of chicane
        phase = (self.time_step // timing) % len(sequence)
        direction = sequence[phase]
        
        for y in range(IMG_H):
            # Progressive offset according to direction
            offset = direction * 20 * min(1.0, (y % timing) / timing)
            center_x = IMG_W // 2 + offset - self.state.steering * 10
            
            left = int(center_x - 8)
            right = int(center_x + 8)
            
            if 0 <= left < IMG_W:
                img[0, 0, y, left:left+2] = 0.9
            if 0 <= right < IMG_W:
                img[0, 0, y, right-2:right] = 0.9
        
        return img
    
    def _draw_standard_road(self, img: torch.Tensor) -> torch.Tensor:
        """Draws a standard road"""
        curvature = math.sin(self.time_step * 0.01)
        
        for y in range(IMG_H):
            road_width = max(6, 15 - y * 0.1)
            center_x = IMG_W // 2 + curvature * y * 0.5 - self.state.steering * 20
            
            left = int(center_x - road_width)
            right = int(center_x + road_width)
            
            if 0 <= left < IMG_W:
                img[0, 0, y, left:left+2] = 0.6
            if 0 <= right < IMG_W:
                img[0, 0, y, right-2:right] = 0.6
        
        return img
    
    def _draw_memory_signal(self, img: torch.Tensor, memory_item: Dict):
        """Draws a memory signal"""
        value = memory_item.get('value', 0)
        strength = memory_item.get('strength', 0.5)
        
        # Random but consistent position for this signal
        signal_id = memory_item.get('id', 0)
        pos_y = 70 + (signal_id * 7) % 30
        pos_x = 10 + (signal_id * 13) % 60
        
        # Intensity proportional to memory strength
        intensity = 0.3 + 0.7 * strength
        
        if value % 2 == 0:  # Even signal = left
            img[0, 0, pos_y:pos_y+8, pos_x:pos_x+8] = intensity * 0.5  # Blue
        else:  # Odd signal = right
            img[0, 0, pos_y:pos_y+8, pos_x:pos_x+8] = intensity * 1.0  # Red
    
    def _draw_distraction(self, img: torch.Tensor, distraction: Dict):
        """Draws a distraction"""
        distraction_type = distraction.get('type', '')
        salience = distraction.get('salience', 0.5)
        
        if 'position' in distraction:
            x, y = distraction['position']
            x, y = int(x), int(y)
            
            if 0 <= x < IMG_W and 0 <= y < IMG_H:
                size = int(5 + salience * 10)
                
                if distraction_type == 'attention_capture':
                    # Flash or movement
                    if distraction.get('subtype') == 'flash':
                        img[0, 0, max(0, y-2):min(IMG_H, y+3), max(0, x-2):min(IMG_W, x+3)] = salience
                    elif distraction.get('subtype') == 'movement':
                        # Movement effect (trail)
                        for i in range(3):
                            yy = min(IMG_H-1, y + i)
                            xx = max(0, x - i)
                            img[0, 0, yy, xx:min(IMG_W, xx+3)] = salience * 0.8
                
                elif distraction_type == 'semantic_interference':
                    # Sign similar to signal
                    color = distraction.get('color', 0.5)
                    img[0, 0, y:min(IMG_H, y+10), x:min(IMG_W, x+10)] = color * salience
    
    def _apply_weather_effects(self, img: torch.Tensor, weather: Dict) -> torch.Tensor:
        """Applies weather effects"""
        weather_type = weather['type']
        intensity = weather['intensity']
        
        if weather_type == 'rain':
            # Rain - random drops
            for _ in range(int(30 * intensity)):
                rx = self.rng.randint(0, IMG_W-1)
                ry = self.rng.randint(0, IMG_H-1)
                img[0, 0, ry:min(IMG_H, ry+2), rx:min(IMG_W, rx+1)] += 0.3 * intensity
            
            # Contrast reduction
            img = img * (1.0 - 0.3 * intensity)
            
        elif weather_type == 'fog':
            # Fog - visibility reduction
            fog_density = 0.4 * intensity
            img = img * (1.0 - fog_density) + fog_density * 0.5
            
        elif weather_type == 'storm':
            # Storm - dramatic effects
            if self.rng.random() < 0.05 * intensity:
                # Lightning
                img = torch.clamp(img + 0.5, 0, 1)
            
            # Heavy rain
            img = self._apply_weather_effects(img, {'type': 'rain', 'intensity': intensity * 1.5})
            
        elif weather_type == 'snow':
            # Snow - flakes
            for _ in range(int(20 * intensity)):
                sx = self.rng.randint(0, IMG_W-1)
                sy = self.rng.randint(0, IMG_H-1)
                img[0, 0, sy:min(IMG_H, sy+1), sx:min(IMG_W, sx+1)] = 1.0
            
            # Contrast reduction
            img = img * (1.0 - 0.2 * intensity) + 0.2 * intensity
        
        return img
    
    def _apply_sensor_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Applies realistic sensor noise"""
        # Gaussian noise
        gaussian_noise = torch.randn_like(img) * 0.05
        
        # Impulse noise (sensor defects)
        impulse_noise = torch.zeros_like(img)
        if self.rng.random() < 0.01:
            impulse_noise[:, :, ::3, ::3] = self.rng.random() * 0.3
        
        # Quantization noise
        quantization_noise = torch.rand_like(img) * 0.02
        
        # Thermal noise (dependent on temperature)
        thermal_noise = torch.randn_like(img) * 0.01 * (self.state.brake_temp / 100.0)
        
        # Total noise application
        noisy_img = img + gaussian_noise + impulse_noise + quantization_noise + thermal_noise
        
        return torch.clamp(noisy_img, 0, 1)
    
    def _get_current_road_segment(self) -> Optional[Dict]:
        """Returns the current road segment"""
        road_pos = self.state.x
        
        for segment in self.road_system.segments:
            if segment['start'] <= road_pos <= segment['end']:
                return segment
        
        return None
    
    def _calculate_torture_reward(self, memory_reward: float, attention_reward: float, 
                                 decision_reward: float) -> torch.Tensor:
        """Calculates final torture reward"""
        # Base survival reward
        survival_reward = 0.1
        
        # High stress penalty
        stress_penalty = -self.stress_level * 2.0
        
        # Cognitive efficiency reward
        cognitive_efficiency = (memory_reward + attention_reward + decision_reward) / 3.0
        cognitive_bonus = cognitive_efficiency * 5.0
        
        # Lane deviation penalty
        lane_deviation = self._calculate_lane_deviation()
        lane_penalty = -lane_deviation * 10.0
        
        # Proper speed reward
        speed_reward = self._calculate_speed_reward()
        
        # Tire wear penalty
        wear_penalty = -(1.0 - self.state.tire_wear) * 5.0
        
        # Weighted total reward
        total_reward = (
            survival_reward * 0.1 +
            memory_reward * 0.3 +
            attention_reward * 0.2 +
            decision_reward * 0.2 +
            cognitive_bonus * 0.1 +
            speed_reward * 0.05 +
            stress_penalty * 0.2 +
            lane_penalty * 0.3 +
            wear_penalty * 0.1
        )
        
        # Neuromodulation application
        neuromod_multiplier = (
            self.neuromodulation.dopamine * 0.4 +
            self.neuromodulation.norepinephrine * 0.3 +
            self.neuromodulation.acetylcholine * 0.2 +
            self.neuromodulation.serotonin * 0.1
        )
        
        total_reward *= neuromod_multiplier
        
        return torch.tensor([total_reward], device=self.device)
    
    def _calculate_lane_deviation(self) -> float:
        """Calculates lane deviation"""
        current_segment = self._get_current_road_segment()
        
        if current_segment:
            # Ideal position at center of road
            ideal_x = IMG_W // 2
            
            # Current vehicle position in image
            vehicle_x = IMG_W // 2  # Approximation
            
            deviation = abs(vehicle_x - ideal_x) / (IMG_W // 2)
            return min(1.0, deviation)
        
        return 0.0
    
    def _calculate_speed_reward(self) -> float:
        """Calculates reward for appropriate speed"""
        ideal_speed = 15.0  # km/h
        
        speed_diff = abs(self.state.velocity - ideal_speed)
        speed_penalty = speed_diff / ideal_speed
        
        return 1.0 - min(1.0, speed_penalty)
    
    def _check_episode_end(self) -> bool:
        """Checks episode end conditions"""
        # End after 500 steps
        if self.time_step >= 500:
            return True
        
        # Severe lane deviation
        lane_deviation = self._calculate_lane_deviation()
        if lane_deviation > 0.9:
            return True
        
        # Virtual collision (too much stress)
        if self.stress_level > 0.95:
            return True
        
        # Total cognitive failure
        if len(self.active_memory_items) == 0 and self.time_step > 100:
            # No active memory left after some time
            return True
        
        # Prolonged zero speed
        if self.state.velocity < 0.1 and self.time_step > 50:
            # Stuck
            return True
        
        return False
