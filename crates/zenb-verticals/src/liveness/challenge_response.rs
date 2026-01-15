//! Challenge-Response Verification
//!
//! Generates random challenges (blink, turn head, smile)
//! and verifies user compliance.

use rand::Rng;

/// Challenge types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeType {
    Blink,
    TurnLeft,
    TurnRight,
    Smile,
    OpenMouth,
    RaiseBrows,
}

/// Challenge state
#[derive(Debug, Clone)]
pub struct Challenge {
    pub challenge_type: ChallengeType,
    pub instruction: String,
    pub timeout_ms: u64,
    pub started_at_us: i64,
}

/// Challenge result
#[derive(Debug, Clone)]
pub struct ChallengeResult {
    pub passed: bool,
    pub challenge: ChallengeType,
    pub response_time_ms: f32,
}

/// Challenge Generator
pub struct ChallengeGenerator {
    current_challenge: Option<Challenge>,
    completed_challenges: Vec<ChallengeResult>,
    required_challenges: usize,
}

impl ChallengeGenerator {
    pub fn new() -> Self {
        Self {
            current_challenge: None,
            completed_challenges: Vec::new(),
            required_challenges: 2,
        }
    }

    /// Generate a new random challenge
    pub fn generate(&mut self, timestamp_us: i64) -> Challenge {
        let mut rng = rand::thread_rng();
        let challenge_type = match rng.gen_range(0..6) {
            0 => ChallengeType::Blink,
            1 => ChallengeType::TurnLeft,
            2 => ChallengeType::TurnRight,
            3 => ChallengeType::Smile,
            4 => ChallengeType::OpenMouth,
            _ => ChallengeType::RaiseBrows,
        };

        let instruction = match challenge_type {
            ChallengeType::Blink => "Please blink your eyes",
            ChallengeType::TurnLeft => "Please turn your head left",
            ChallengeType::TurnRight => "Please turn your head right",
            ChallengeType::Smile => "Please smile",
            ChallengeType::OpenMouth => "Please open your mouth",
            ChallengeType::RaiseBrows => "Please raise your eyebrows",
        }
        .to_string();

        let challenge = Challenge {
            challenge_type,
            instruction,
            timeout_ms: 5000,
            started_at_us: timestamp_us,
        };

        self.current_challenge = Some(challenge.clone());
        challenge
    }

    /// Verify if current challenge is completed
    pub fn verify(&mut self, landmarks: &[[f32; 2]]) -> bool {
        let challenge = match &self.current_challenge {
            Some(c) => c.clone(),
            None => return false,
        };

        let passed = match challenge.challenge_type {
            ChallengeType::Blink => self.detect_blink(landmarks),
            ChallengeType::TurnLeft => self.detect_head_turn(landmarks, -20.0),
            ChallengeType::TurnRight => self.detect_head_turn(landmarks, 20.0),
            ChallengeType::Smile => self.detect_smile(landmarks),
            ChallengeType::OpenMouth => self.detect_open_mouth(landmarks),
            ChallengeType::RaiseBrows => self.detect_raised_brows(landmarks),
        };

        if passed {
            self.completed_challenges.push(ChallengeResult {
                passed: true,
                challenge: challenge.challenge_type,
                response_time_ms: 0.0,
            });
            self.current_challenge = None;
        }

        passed
    }

    /// Check if all required challenges are completed
    pub fn is_complete(&self) -> bool {
        self.completed_challenges.len() >= self.required_challenges
    }

    fn detect_blink(&self, landmarks: &[[f32; 2]]) -> bool {
        if landmarks.len() < 468 {
            return false;
        }

        let left_top = landmarks[159];
        let left_bottom = landmarks[145];
        let left_inner = landmarks[133];
        let left_outer = landmarks[33];

        let v = ((left_top[0] - left_bottom[0]).powi(2)
            + (left_top[1] - left_bottom[1]).powi(2))
        .sqrt();
        let h = ((left_inner[0] - left_outer[0]).powi(2)
            + (left_inner[1] - left_outer[1]).powi(2))
        .sqrt();

        let ear = v / h.max(0.001);
        ear < 0.15
    }

    fn detect_head_turn(&self, landmarks: &[[f32; 2]], expected_yaw: f32) -> bool {
        if landmarks.len() < 468 {
            return false;
        }

        let nose = landmarks[4];
        let left_cheek = landmarks[234];
        let right_cheek = landmarks[454];

        let left_dist = ((nose[0] - left_cheek[0]).powi(2)
            + (nose[1] - left_cheek[1]).powi(2))
        .sqrt();
        let right_dist = ((nose[0] - right_cheek[0]).powi(2)
            + (nose[1] - right_cheek[1]).powi(2))
        .sqrt();

        let asymmetry = (right_dist - left_dist) / (left_dist + right_dist).max(0.001);
        let estimated_yaw = asymmetry * 45.0;

        if expected_yaw < 0.0 {
            estimated_yaw < -15.0
        } else {
            estimated_yaw > 15.0
        }
    }

    fn detect_smile(&self, landmarks: &[[f32; 2]]) -> bool {
        if landmarks.len() < 468 {
            return false;
        }

        let left_corner = landmarks[61];
        let right_corner = landmarks[291];
        let upper_lip = landmarks[0];
        let lower_lip = landmarks[17];

        let width = ((left_corner[0] - right_corner[0]).powi(2)
            + (left_corner[1] - right_corner[1]).powi(2))
        .sqrt();
        let height = ((upper_lip[0] - lower_lip[0]).powi(2)
            + (upper_lip[1] - lower_lip[1]).powi(2))
        .sqrt();

        width / height.max(0.001) > 3.0
    }

    fn detect_open_mouth(&self, landmarks: &[[f32; 2]]) -> bool {
        if landmarks.len() < 468 {
            return false;
        }

        let upper_lip = landmarks[0];
        let lower_lip = landmarks[17];

        let height = ((upper_lip[0] - lower_lip[0]).powi(2)
            + (upper_lip[1] - lower_lip[1]).powi(2))
        .sqrt();
        height > 0.05
    }

    fn detect_raised_brows(&self, landmarks: &[[f32; 2]]) -> bool {
        if landmarks.len() < 468 {
            return false;
        }

        let left_brow = landmarks[105];
        let left_eye = landmarks[159];
        let right_brow = landmarks[334];
        let right_eye = landmarks[386];

        let left_dist = (left_brow[1] - left_eye[1]).abs();
        let right_dist = (right_brow[1] - right_eye[1]).abs();

        (left_dist + right_dist) / 2.0 > 0.08
    }

    pub fn reset(&mut self) {
        self.current_challenge = None;
        self.completed_challenges.clear();
    }
}

impl Default for ChallengeGenerator {
    fn default() -> Self {
        Self::new()
    }
}
