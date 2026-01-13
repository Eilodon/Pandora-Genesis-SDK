#![allow(clippy::single_component_path_imports)]
#![allow(clippy::single_match)]

use serde::{Deserialize, Serialize};
use zenb_core::domain::{ControlDecision, Envelope, Event};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Dashboard {
    pub session_active: bool,
    pub last_decision: Option<ControlDecision>,
    pub last_deny_reason: Option<String>,
    pub current_mode: Option<u8>,
    pub belief_conf: Option<f32>,
    pub belief_p: Option<[f32; 5]>,
    pub free_energy_ema: Option<f32>,
    pub lr: Option<f32>,
    pub resonance_score: Option<f32>,
}

impl Dashboard {
    pub fn apply(&mut self, e: &Envelope) {
        match &e.event {
            Event::SessionStarted { .. } => self.session_active = true,
            Event::ControlDecisionMade { decision } => {
                self.last_decision = Some(decision.clone());
                self.last_deny_reason = None;
            }
            Event::ControlDecisionDenied {
                reason,
                timestamp: _,
            } => {
                self.last_deny_reason = Some(reason.clone());
            }
            Event::SessionEnded { .. } => self.session_active = false,
            Event::BeliefUpdated { p, conf, mode } => {
                self.current_mode = Some(*mode);
                self.belief_conf = Some(*conf);
                self.belief_p = Some(*p);
            }
            Event::BeliefUpdatedV2 {
                p,
                conf,
                mode,
                free_energy_ema,
                lr,
                resonance_score,
            } => {
                self.current_mode = Some(*mode);
                self.belief_conf = Some(*conf);
                self.belief_p = Some(*p);
                self.free_energy_ema = Some(*free_energy_ema);
                self.lr = Some(*lr);
                self.resonance_score = Some(*resonance_score);
            }
            Event::PolicyChosen {
                mode,
                reason_bits: _,
                conf,
            } => {
                self.current_mode = Some(*mode);
                self.belief_conf = Some(*conf);
            }
            _ => {}
        }
    }
}

// Derive already provides the needed default values for all fields.

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StatsDaily {
    pub cycles: u64,
}

impl StatsDaily {
    pub fn apply(&mut self, e: &Envelope) {
        match &e.event {
            Event::CycleCompleted { cycles } => self.cycles += *cycles as u64,
            _ => {}
        }
    }
}
