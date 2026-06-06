"""
MATHIR Cognitive Labyrinth Simulator
=====================================
Un environnement de conduite conçu pour tester les limites des architectures de mémoire.
Fusionne : physique imprévisible, distractions cognitives, et tests de mémoire contextuels.
"""

import torch
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from enum import Enum
import random

IMG_H, IMG_W = 84, 84

class WeatherCondition(Enum):
    CLEAR = 0
    RAIN = 1
    FOG = 2
    NIGHT = 3

class TrafficDensity(Enum):
    NONE = 0
    LIGHT = 1
    HEAVY = 2
    JAM = 3

@dataclass
class CognitiveSignal:
    """Signal mémoriel avec contexte temporel"""
    type: str  # 'turn_left', 'turn_right', 'stop_at_3rd', etc.
    value: int  # La direction ou action requise
    condition: str  # Condition d'activation
    seen_time: int = -1
    retention_deadline: int = 0
    distraction_immunity: bool = False  # Si le signal survit aux distractions

@dataclass  
class Distraction:
    """Élément conçu pour perturber l'attention"""
    type: str  # 'traffic_sign', 'pedestrian', 'animal', 'weather'
    position: Tuple[float, float]
    activation_time: Tuple[int, int]  # Fenêtre temporelle
    salience: float  # 0-1, comment il attire l'attention
    relevance: float  # 0-1, importance pour la tâche (0 = distraction pure)

@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    velocity: float = 0.0
    heading: float = 0.0
    steering: float = 0.0

class DrivingSimulator: # Renommage pour compatibilité scripts existants (était CognitiveDrivingSimulator)
    def __init__(self, device='cuda', difficulty='expert'):
        self.device = device
        self.difficulty = difficulty
        
        # Variables de système dynamique
        self.rng = np.random.default_rng()
        self.episode_conditions = {}
        
        # Mémoire épisodique étendue
        self.cognitive_signals: List[CognitiveSignal] = []
        self.active_distractions: List[Distraction] = []
        self.contextual_memory = {}
        self.pending_decisions = []
        
        # État du monde
        self.weather = WeatherCondition.CLEAR
        self.traffic = TrafficDensity.NONE
        self.time_of_day = 0.0  # 0=midnight, 0.5=noon
        self.road_complexity = 1.0
        
        # Paramètres de difficulté
        self._set_difficulty_params(difficulty)
        
        self.reset()
    
    def _set_difficulty_params(self, difficulty):
        """Configure les paramètres selon le niveau"""
        params = {
            'novice': {'memory_decay': 0.99, 'distraction_count': 2, 'physics_noise': 0.05},
            'expert': {'memory_decay': 0.95, 'distraction_count': 5, 'physics_noise': 0.1},
            'master': {'memory_decay': 0.90, 'distraction_count': 8, 'physics_noise': 0.2}
        }
        self.params = params.get(difficulty, params['expert'])
    
    def reset(self):
        """Réinitialise l'environnement avec des conditions aléatoires"""
        # Conditions météo aléatoires
        self.weather = random.choice(list(WeatherCondition))
        self.traffic = random.choice(list(TrafficDensity))
        self.time_of_day = self.rng.random()
        
        # Réinitialiser l'état du véhicule
        self.state = VehicleState()
        self.time_step = 0
        self.road_segments = []
        # Added for compatibility with other scripts referencing seeds
        self.road_seed = self.rng.integers(0, 1000)
        
        # Réinitialiser la mémoire cognitive
        self.cognitive_signals.clear()
        self.active_distractions.clear()
        self.contextual_memory.clear()
        self.pending_decisions.clear()
        
        # Générer une nouvelle route avec complexité variable
        self._generate_cognitive_road()
        
        # Définir les conditions d'épisode
        self.episode_conditions = {
            'signal_trigger': self._generate_signal_trigger(),
            'required_memory_items': self.rng.integers(1, 4),
            'distraction_pattern': self._generate_distraction_pattern()
        }
        
        return self._get_observation()
    
    def _generate_signal_trigger(self) -> Dict:
        """Génère une condition contextuelle pour l'apparition du signal"""
        triggers = [
            {'type': 'sequential', 'condition': 'after_3rd_red_light'},
            {'type': 'temporal', 'condition': 'random_between_50_150'},
            {'type': 'spatial', 'condition': 'after_curve_sequence'},
            {'type': 'event_based', 'condition': 'when_traffic_clears'}
        ]
        return random.choice(triggers)
    
    def _generate_distraction_pattern(self) -> List[Dict]:
        """Génère un modèle de distractions cohérent mais trompeur"""
        patterns = []
        for _ in range(self.params['distraction_count']):
            pattern = {
                'type': random.choice(['traffic_sign', 'pedestrian', 'animal', 'advertisement']),
                'timing': (self.rng.integers(20, 200), self.rng.integers(5, 30)),
                'salience': self.rng.random() * 0.8 + 0.2,
                'relevance': self.rng.random() * 0.3  # La plupart sont non pertinents
            }
            patterns.append(pattern)
        return patterns
    
    def _generate_cognitive_road(self):
        """Génère une route avec des défis cognitifs intégrés"""
        self.road_type = random.choice(['city_grid', 'highway', 'mountain', 'suburban'])
        
        # Définir la complexité basée sur la difficulté
        if self.difficulty == 'novice':
            self.road_complexity = 1.0
        elif self.difficulty == 'expert':
            self.road_complexity = 2.0
        else:
            self.road_complexity = 3.0
        
        # Générer les segments de route
        self.road_segments = []
        current_pos = 0
        
        # Ajouter des segments avec différents défis
        segment_types = [
            ('straight', 50, 'clear'),
            ('curve', 30, 'signal_zone'),
            ('intersection', 20, 'decision'),
            ('straight', 40, 'distraction_zone'),
            ('curve', 25, 'memory_test')
        ]
        
        for seg_type, length, challenge in segment_types:
            self.road_segments.append({
                'type': seg_type, # Fixed syntax error from prompt
                'length': length * self.road_complexity,
                'challenge': challenge,
                'start': current_pos,
                'end': current_pos + length * self.road_complexity
            })
            current_pos += length * self.road_complexity
    
    def step(self, action: torch.Tensor):
        """Exécute un pas de simulation avec physique avancée"""
        dt = 0.1
        
        # Décoder l'action
        steer_cmd = float(action[0, 0].item())
        accel_cmd = float(action[0, 1].item())
        
        # PHYSIQUE AVANCÉE AVEC PERTURBATIONS
        wheelbase = 2.5
        max_steer = 0.5
        
        # Effets météorologiques
        friction = self._get_weather_friction()
        steering_lag = self._get_steering_lag()
        traction = self._get_traction_factor()
        
        # Bruit de capteur (latence et imprécision)
        measured_steering = self.state.steering + self.rng.normal(0, 0.05)
        measured_velocity = self.state.velocity * (1 + self.rng.normal(0, 0.02))
        
        # Dynamique avec perturbations
        target_steering = steer_cmd * max_steer * traction
        self.state.steering += (target_steering - measured_steering) * steering_lag * dt
        
        # Friction variable
        effective_friction = friction * (1 + 0.5 * abs(self.state.steering))
        
        # Accélération avec perte de traction
        acceleration = accel_cmd * 5.0 * traction
        velocity_change = (acceleration - measured_velocity * effective_friction) * dt
        self.state.velocity = max(0.0, self.state.velocity + velocity_change)
        
        # Mise à jour de la position avec dérive
        slip_angle = self._calculate_slip_angle()
        effective_heading = self.state.heading + slip_angle
        
        self.state.heading += (self.state.velocity / wheelbase) * math.tan(self.state.steering) * dt
        self.state.x += self.state.velocity * math.cos(effective_heading) * dt
        self.state.y += self.state.velocity * math.sin(effective_heading) * dt
        
        self.time_step += 1
        
        # GESTION COGNITIVE DYNAMIQUE
        self._update_cognitive_state()
        self._manage_distractions()
        
        # VÉRIFICATION DES DÉCISIONS EN ATTENTE
        reward = self._check_pending_decisions()
        
        # OBSERVATION ET RÉCOMPENSE
        observation = self._get_observation()
        total_reward = self._calculate_reward(reward)
        
        # Vérifier la fin d'épisode
        done = self._check_episode_end()
        
        # Compatibility Info Dict
        info = {}
        
        return observation, total_reward, done, info
    
    def _update_cognitive_state(self):
        """Met à jour les signaux cognitifs et déclencheurs"""
        # Déclenchement contextuel des signaux
        trigger = self.episode_conditions['signal_trigger']
        
        if trigger['type'] == 'sequential':
            # Ex: "Au 3ème feu rouge"
            red_light_count = self.contextual_memory.get('red_lights_passed', 0)
            if red_light_count >= 3 and not any(s.type == 'turn_signal' for s in self.cognitive_signals):
                signal = CognitiveSignal(
                    type='turn_signal',
                    value=self.rng.choice([1, 2]),  # 1=gauche, 2=droite
                    condition='after_3rd_red_light',
                    seen_time=self.time_step,
                    retention_deadline=self.time_step + self.rng.integers(70, 130),
                    distraction_immunity=False
                )
                self.cognitive_signals.append(signal)
        
        elif trigger['type'] == 'event_based':
            # Ex: "Quand le trafic se dissipe"
            if self.traffic == TrafficDensity.NONE and len(self.cognitive_signals) == 0:
                signal = CognitiveSignal(
                    type='lane_change',
                    value=self.rng.choice([1, 2]),
                    condition='traffic_clear',
                    seen_time=self.time_step,
                    retention_deadline=self.time_step + self.rng.integers(50, 100)
                )
                self.cognitive_signals.append(signal)
        
        # Mettre à jour les délais de rétention
        for signal in self.cognitive_signals:
            if self.time_step > signal.retention_deadline:
                signal.distraction_immunity = False  # La mémoire s'affaiblit
    
    def _manage_distractions(self):
        """Gère l'apparition et la disparition des distractions"""
        # Appliquer le pattern de distractions
        for pattern in self.episode_conditions['distraction_pattern']:
            start, duration = pattern['timing']
            if start <= self.time_step < start + duration:
                # Créer une distraction si pas déjà présente
                if not any(d.type == pattern['type'] for d in self.active_distractions):
                    distraction = Distraction(
                        type=pattern['type'],
                        position=(self.rng.random() * 84, self.rng.random() * 84),
                        activation_time=(self.time_step, self.time_step + duration),
                        salience=pattern['salience'],
                        relevance=pattern['relevance']
                    )
                    self.active_distractions.append(distraction)
        
        # Retirer les distractions expirées
        self.active_distractions = [
            d for d in self.active_distractions
            if self.time_step < d.activation_time[1]
        ]
        
        # Distractions aléatoires additionnelles
        if self.rng.random() < 0.01:  # 1% de chance par pas
            distraction = Distraction(
                type='sudden_event',
                position=(self.rng.random() * 84, self.rng.random() * 84),
                activation_time=(self.time_step, self.time_step + 10),
                salience=0.9,
                relevance=0.0
            )
            self.active_distractions.append(distraction)
    
    def _check_pending_decisions(self) -> float:
        """Vérifie les décisions en attente et retourne la récompense"""
        reward = 0.0
        
        # Vérifier chaque signal mémorisé
        for signal in list(self.cognitive_signals):
            # Vérifier si nous sommes à un point de décision
            if self._is_decision_point(signal):
                # Observer l'action actuelle
                if signal.type == 'turn_signal':
                    required_action = 'left' if signal.value == 1 else 'right'
                    current_action = 'left' if self.state.steering < -0.3 else 'right' if self.state.steering > 0.3 else 'none'
                    
                    if current_action == required_action:
                        reward += 15.0  # Forte récompense pour mémoire correcte
                        print(f"✅ COGNITIVE SUCCESS: Remembered {required_action} after distractions")
                    else:
                        reward -= 15.0  # Forte pénalité pour échec
                        print(f"❌ COGNITIVE FAIL: Forgot {required_action}")
                
                # Retirer le signal traité
                if signal in self.cognitive_signals:
                    self.cognitive_signals.remove(signal)
        
        return reward
    
    def _is_decision_point(self, signal: CognitiveSignal) -> bool:
        """Détermine si c'est le moment de prendre une décision"""
        # Les points de décision sont maintenant contextuels
        road_pos = self.state.x  # Position approximative sur la route
        
        # Vérifier les intersections basées sur la route générée
        for segment in self.road_segments:
            if segment['challenge'] == 'decision':
                if segment['start'] <= road_pos <= segment['end']:
                    return True
        
        # Ou basé sur le délai de rétention
        if signal.retention_deadline - 10 <= self.time_step <= signal.retention_deadline + 10:
            return True
        
        return False
    
    def _get_weather_friction(self) -> float:
        """Retourne le coefficient de friction basé sur la météo"""
        base_friction = 0.05
        if self.weather == WeatherCondition.RAIN:
            return base_friction * 0.7
        elif self.weather == WeatherCondition.FOG:
            return base_friction * 0.9  # Humidité
        elif self.weather == WeatherCondition.NIGHT:
            return base_friction * 0.95  # Rosée nocturne
        return base_friction
    
    def _get_steering_lag(self) -> float:
        """Retourne le délai de direction (simule un système réel)"""
        base_lag = 0.2
        if self.traffic == TrafficDensity.HEAVY:
            return base_lag * 1.5  # Stress du conducteur
        return base_lag
    
    def _get_traction_factor(self) -> float:
        """Facteur d'adhérence variable"""
        base_traction = 1.0
        # Effets aléatoires (nids-de-poule, gravillons)
        if self.rng.random() < 0.01:
            return base_traction * self.rng.uniform(0.5, 0.9)
        return base_traction
    
    def _calculate_slip_angle(self) -> float:
        """Calcule l'angle de dérive du véhicule"""
        if self.state.velocity < 1.0:
            return 0.0
        
        # Modèle de dérive simplifié
        slip_ratio = abs(self.state.steering) * self.state.velocity / 30.0
        
        # Effets météorologiques
        if self.weather == WeatherCondition.RAIN:
            slip_ratio *= 1.5
        elif self.weather == WeatherCondition.FOG:
            slip_ratio *= 1.2
        
        # Bruit aléatoire
        slip_ratio += self.rng.normal(0, 0.02) * self.params['physics_noise']
        
        return slip_ratio * (1 if self.rng.random() > 0.5 else -1)
    
    def _get_observation(self) -> Dict[str, torch.Tensor]:
        """Génère une observation avec bruit et distractions"""
        # Image principale 84x84
        img = torch.zeros(1, 1, IMG_H, IMG_W, device=self.device)
        
        # ROUTE AVANCÉE
        curvature = math.sin(self.time_step * 0.01 + self.road_seed)
        curvature += 0.3 * math.sin(self.time_step * 0.03)  # Micro-variations
        
        # Dessiner la route avec effet météo
        road_visibility = 1.0
        if self.weather == WeatherCondition.FOG:
            road_visibility = 0.6
        elif self.weather == WeatherCondition.NIGHT:
            road_visibility = 0.4
        
        for y in range(84):
            # Route qui se rétrécit avec la distance
            road_width = max(5, 20 - (y * 0.15))
            center_x = 42 + (curvature * y * 0.5) - (self.state.steering * 20)
            
            # Lignes de voie avec imperfections
            left_lane = int(center_x - road_width)
            right_lane = int(center_x + road_width)
            
            if 0 <= left_lane < 84:
                img[0, 0, y, max(0, left_lane-1):left_lane+2] = 0.7 * road_visibility
            if 0 <= right_lane < 84:
                img[0, 0, y, max(0, right_lane-1):right_lane+2] = 0.7 * road_visibility
            
            # Surface de route avec texture
            if y % 4 == 0:  # Lignes de marquage discontinues
                road_center = int(center_x)
                if 0 <= road_center < 84:
                    img[0, 0, y, road_center:road_center+2] = 1.0 * road_visibility
        
        # SIGNAL COGNITIF (si visible)
        for signal in self.cognitive_signals:
            if signal.type == 'turn_signal':
                # Position aléatoire (pas toujours au même endroit)
                pos_y = 70 + self.rng.integers(-10, 10)
                if signal.value == 1:  # Gauche
                    pos_x = 10 + self.rng.integers(-5, 5)
                    img[0, 0, pos_y:pos_y+10, pos_x:pos_x+10] = 0.5
                else:  # Droite
                    pos_x = 64 + self.rng.integers(-5, 5)
                    img[0, 0, pos_y:pos_y+10, pos_x:pos_x+10] = 1.0
        
        # DISTRACTIONS
        for distraction in self.active_distractions:
            x, y = map(int, distraction.position)
            size = int(5 + distraction.salience * 10)
            
            if 0 <= x < 84 and 0 <= y < 84:
                if distraction.type == 'traffic_sign':
                    # Panneau circulaire
                    for i in range(size):
                        for j in range(size):
                            if (i-size/2)**2 + (j-size/2)**2 <= (size/2)**2:
                                img[0, 0, 
                                    min(83, max(0, y+i-size//2)), 
                                    min(83, max(0, x+j-size//2))] = 0.3 * distraction.salience
                
                elif distraction.type == 'pedestrian':
                    # Forme humaine
                    img[0, 0, y:min(84, y+size), x:min(84, x+size//2)] = 0.4 * distraction.salience
                
                elif distraction.type == 'animal':
                    # Forme animale
                    img[0, 0, y:min(84, y+size//2), x:min(84, x+size)] = 0.5 * distraction.salience
        
        # EFFETS MÉTÉO
        if self.weather == WeatherCondition.RAIN:
            # Gouttes de pluie aléatoires
            for _ in range(20):
                rx, ry = self.rng.integers(0, 84), self.rng.integers(0, 84)
                img[0, 0, ry:min(84, ry+2), rx:min(84, rx+1)] += 0.2
        
        elif self.weather == WeatherCondition.FOG:
            # Brouillard (réduction de contraste)
            img = img * 0.7 + 0.3
        
        elif self.weather == WeatherCondition.NIGHT:
            # Nuit (obscurité + phares)
            img = img * 0.5
            
            # Effet de phares
            light_center = 42 + int(self.state.steering * 15)
            for y in range(40, 84):
                brightness = 0.8 * (1 - (y-40)/44)
                light_width = 10 * (1 - (y-40)/44)
                left = max(0, int(light_center - light_width))
                right = min(84, int(light_center + light_width))
                img[0, 0, y, left:right] += brightness
        
        # BRUIT SENSORIEL
        noise = torch.randn_like(img) * 0.05 * self.params['physics_noise']
        img = torch.clamp(img + noise, 0, 1)
        
        # VECTEUR D'ÉTAT ÉTENDU (9 dimensions)
        state_vec = torch.tensor([[
            self.state.velocity / 30.0,
            self.state.steering / 0.5,
            math.cos(self.state.heading),
            math.sin(self.state.heading),
            self.time_of_day,
            len(self.cognitive_signals) / 3.0,  # Charge mémorielle
            len(self.active_distractions) / 10.0,  # Charge attentionnelle
            self.weather.value / 3.0,
            self.traffic.value / 3.0
        ]], device=self.device)
        
        return {
            'camera': img,
            'state': state_vec,
            'cognitive_load': torch.tensor([len(self.cognitive_signals)], device=self.device)
        }
    
    def _calculate_reward(self, memory_reward: float) -> torch.Tensor:
        """Calcule la récompense multi-objectifs"""
        # 1. Récompense de conduite sûre
        curvature = math.sin(self.time_step * 0.01 + self.road_seed)
        steering_err = abs(self.state.steering - curvature)
        
        # Pénalité pour sortie de route virtuelle
        road_center = 42
        car_pos = 42  # Approximation
        lane_deviation = abs(car_pos - road_center) / 42.0
        
        # 2. Récompense d'efficacité
        speed_efficiency = min(1.0, self.state.velocity / 20.0)
        
        # 3. Pénalité pour surcharge cognitive
        cognitive_penalty = -0.1 * len(self.cognitive_signals)
        
        # 4. Récompense pour filtrage des distractions
        distraction_resistance = 0.0
        irrelevant_distractions = [d for d in self.active_distractions if d.relevance < 0.2]
        if irrelevant_distractions:
            # Récompenser si l'IA ignore les distractions non pertinentes
            distraction_resistance = 0.05 * len(irrelevant_distractions)
        
        # Composition de la récompense totale
        base_reward = (1.0 - steering_err * 0.5) * 0.3
        lane_reward = (1.0 - lane_deviation) * 0.3
        efficiency_reward = speed_efficiency * 0.2
        cognitive_reward = memory_reward * 0.2
        
        total = base_reward + lane_reward + efficiency_reward + cognitive_reward
        total += cognitive_penalty + distraction_resistance
        
        return torch.tensor([total], device=self.device)
    
    def _check_episode_end(self) -> bool:
        """Vérifie les conditions de fin d'épisode"""
        # Fin après 500 pas ou sortie de route sévère
        if self.time_step >= 500:
            return True
        
        # Sortie de route
        if abs(self.state.x) > 100 or abs(self.state.y) > 100:
            return True
        
        # Échec cognitif grave (oubli de tous les signaux)
        if len(self.cognitive_signals) > 0 and self.time_step > 300:
            # Si un signal est en retard de plus de 50 pas
            for signal in self.cognitive_signals:
                if self.time_step > signal.retention_deadline + 50:
                    return True
        
        return False
