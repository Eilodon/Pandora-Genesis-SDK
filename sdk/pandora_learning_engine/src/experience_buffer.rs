use pandora_core::ontology::EpistemologicalFlow;

#[derive(Debug, Clone)]
pub struct ExperienceSample {
    pub flow: EpistemologicalFlow,
    pub reward: f64,
}

#[derive(Debug, Default)]
pub struct ExperienceBuffer {
    samples: Vec<ExperienceSample>,
    capacity: usize,
}

impl ExperienceBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, sample: ExperienceSample) {
        if self.samples.len() >= self.capacity {
            self.samples.remove(0);
        }
        self.samples.push(sample);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = &ExperienceSample> {
        self.samples.iter()
    }
}

/// Priority sampling experience buffer (simplified SumTree-free version)
#[derive(Debug)]
pub struct PriorityExperienceBuffer {
    items: Vec<(ExperienceSample, f64)>, // (sample, priority)
    capacity: usize,
}

impl PriorityExperienceBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, sample: ExperienceSample, priority: f64) {
        if self.items.len() >= self.capacity {
            self.items.remove(0);
        }
        self.items.push((sample, priority.max(1e-6)));
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Sample an index proportionally to priority (O(n)).
    pub fn sample_index(&self, seed: u64) -> Option<usize> {
        if self.items.is_empty() {
            return None;
        }
        let total: f64 = self.items.iter().map(|(_, p)| *p).sum();
        if total <= 0.0 {
            return Some(0);
        }
        // simple LCG-based rng from seed
        let r = ((seed.wrapping_mul(6364136223846793005).wrapping_add(1)) >> 1) as f64
            / (u64::MAX as f64);
        let mut acc = 0.0;
        let target = r * total;
        for (i, (_, p)) in self.items.iter().enumerate() {
            acc += *p;
            if acc >= target {
                return Some(i);
            }
        }
        Some(self.items.len() - 1)
    }

    pub fn get(&self, index: usize) -> Option<&ExperienceSample> {
        self.items.get(index).map(|(s, _)| s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[test]
    fn buffer_respects_capacity() {
        let mut buf = ExperienceBuffer::with_capacity(2);
        let flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"a"));
        buf.push(ExperienceSample {
            flow: flow.clone(),
            reward: 1.0,
        });
        buf.push(ExperienceSample {
            flow: flow.clone(),
            reward: 2.0,
        });
        buf.push(ExperienceSample { flow, reward: 3.0 });
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn priority_sampling_prefers_higher_priority() {
        let mut buf = PriorityExperienceBuffer::with_capacity(3);
        let mk = |s: &str| EpistemologicalFlow::from_bytes(Bytes::copy_from_slice(s.as_bytes()));
        buf.push(
            ExperienceSample {
                flow: mk("a"),
                reward: 0.0,
            },
            1.0,
        );
        buf.push(
            ExperienceSample {
                flow: mk("b"),
                reward: 0.0,
            },
            10.0,
        );
        buf.push(
            ExperienceSample {
                flow: mk("c"),
                reward: 0.0,
            },
            1.0,
        );

        // sample multiple seeds, expect index 1 to appear often
        let mut hits_mid = 0;
        for k in 0..100 {
            if let Some(i) = buf.sample_index(k) {
                if i == 1 {
                    hits_mid += 1;
                }
            }
        }
        assert!(hits_mid > 40);
    }
}
