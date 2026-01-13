//! Soundscape mixer - Multi-layer audio control

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub name: &'static str,
    pub base_gain: f32,
    pub inhale_gain: f32,
    pub exhale_gain: f32,
}

const FOREST_LAYERS: &[LayerConfig] = &[
    LayerConfig {
        name: "birds",
        base_gain: 0.3,
        inhale_gain: 0.45,
        exhale_gain: 0.15,
    },
    LayerConfig {
        name: "wind",
        base_gain: 0.5,
        inhale_gain: 0.6,
        exhale_gain: 0.3,
    },
    LayerConfig {
        name: "creek",
        base_gain: 0.4,
        inhale_gain: 0.3,
        exhale_gain: 0.5,
    },
];

const OCEAN_LAYERS: &[LayerConfig] = &[
    LayerConfig {
        name: "waves",
        base_gain: 0.6,
        inhale_gain: 0.3,
        exhale_gain: 0.7,
    },
    LayerConfig {
        name: "seagulls",
        base_gain: 0.2,
        inhale_gain: 0.3,
        exhale_gain: 0.1,
    },
];

pub struct SoundscapeEngine {
    scene: String,
}

impl SoundscapeEngine {
    pub fn new(scene: &str) -> Self {
        Self {
            scene: scene.to_string(),
        }
    }

    pub fn compute_mix(
        &self,
        phase: &str,
        valence: f32,
        arousal: f32,
    ) -> HashMap<&'static str, f32> {
        let layers = match self.scene.as_str() {
            "forest" => FOREST_LAYERS,
            "ocean" => OCEAN_LAYERS,
            _ => &[],
        };

        layers
            .iter()
            .map(|layer| {
                let mut gain = match phase {
                    "inhale" => layer.inhale_gain,
                    "exhale" => layer.exhale_gain,
                    _ => layer.base_gain,
                };

                // Mood modulation
                if valence > 0.5 && layer.name.contains("bird") {
                    gain *= 1.2;
                }
                if arousal < 0.3 && layer.name.contains("wave") {
                    gain *= 1.1;
                }

                (layer.name, gain.clamp(0.0, 1.0))
            })
            .collect()
    }

    pub fn set_scene(&mut self, scene: &str) {
        self.scene = scene.to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forest_mix() {
        let engine = SoundscapeEngine::new("forest");
        let mix = engine.compute_mix("inhale", 0.5, 0.5);

        assert!(mix.contains_key("birds"));
        assert!(mix.contains_key("wind"));
        assert!(mix.contains_key("creek"));

        // Inhale should boost wind
        assert_eq!(mix["wind"], 0.6);
    }

    #[test]
    fn test_ocean_mix() {
        let engine = SoundscapeEngine::new("ocean");
        let mix = engine.compute_mix("exhale", 0.5, 0.5);

        assert!(mix.contains_key("waves"));
        assert!(mix.contains_key("seagulls"));

        // Exhale should boost waves
        assert_eq!(mix["waves"], 0.7);
    }
}
