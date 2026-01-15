//! Landmark-based ROI Extraction
//!
//! Extracts physiologically optimal ROIs (forehead, cheeks) from
//! MediaPipe Face Mesh 468 landmarks.
//!
//! # rPPG ROI Best Practices
//!
//! Research shows forehead and cheeks provide the strongest rPPG signals:
//! - Forehead: High blood perfusion, less facial movement
//! - Cheeks: Good signal, but affected by expressions
//!
//! # MediaPipe Landmark Indices
//!
//! - Forehead: 10, 67, 109, 151, 297, 338
//! - Left Cheek: 234, 227, 116, 123, 147, 187
//! - Right Cheek: 454, 447, 345, 352, 376, 411

/// Forehead landmark indices (MediaPipe Face Mesh)
pub const FOREHEAD_LANDMARKS: &[usize] = &[10, 67, 109, 151, 297, 338];

/// Left cheek landmark indices
pub const CHEEK_LEFT_LANDMARKS: &[usize] = &[234, 227, 116, 123, 147, 187];

/// Right cheek landmark indices
pub const CHEEK_RIGHT_LANDMARKS: &[usize] = &[454, 447, 345, 352, 376, 411];

/// Simple polygon for ROI definition
#[derive(Debug, Clone)]
pub struct Polygon {
    /// Vertices as (x, y) coordinates
    pub vertices: Vec<[f32; 2]>,
}

impl Polygon {
    /// Create new polygon from vertices
    pub fn new(vertices: Vec<[f32; 2]>) -> Self {
        Self { vertices }
    }

    /// Get bounding box [x_min, y_min, x_max, y_max]
    pub fn bounding_box(&self) -> [f32; 4] {
        if self.vertices.is_empty() {
            return [0.0, 0.0, 0.0, 0.0];
        }
        
        let mut x_min = f32::MAX;
        let mut y_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_max = f32::MIN;
        
        for [x, y] in &self.vertices {
            x_min = x_min.min(*x);
            y_min = y_min.min(*y);
            x_max = x_max.max(*x);
            y_max = y_max.max(*y);
        }
        
        [x_min, y_min, x_max, y_max]
    }

    /// Check if point is inside polygon (ray casting algorithm)
    pub fn contains(&self, x: f32, y: f32) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }
        
        let mut inside = false;
        let mut j = n - 1;
        
        for i in 0..n {
            let [xi, yi] = self.vertices[i];
            let [xj, yj] = self.vertices[j];
            
            if ((yi > y) != (yj > y))
                && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            {
                inside = !inside;
            }
            j = i;
        }
        
        inside
    }
}

/// Extract forehead ROI polygon from landmarks
///
/// # Arguments
/// * `landmarks` - 468 MediaPipe landmarks as [(x, y), ...]
///
/// # Returns
/// Polygon covering forehead region
pub fn forehead_roi(landmarks: &[[f32; 2]]) -> Option<Polygon> {
    if landmarks.len() < 468 {
        return None;
    }
    
    let vertices: Vec<[f32; 2]> = FOREHEAD_LANDMARKS
        .iter()
        .map(|&idx| landmarks[idx])
        .collect();
    
    Some(Polygon::new(vertices))
}

/// Extract left cheek ROI polygon from landmarks
pub fn left_cheek_roi(landmarks: &[[f32; 2]]) -> Option<Polygon> {
    if landmarks.len() < 468 {
        return None;
    }
    
    let vertices: Vec<[f32; 2]> = CHEEK_LEFT_LANDMARKS
        .iter()
        .map(|&idx| landmarks[idx])
        .collect();
    
    Some(Polygon::new(vertices))
}

/// Extract right cheek ROI polygon from landmarks
pub fn right_cheek_roi(landmarks: &[[f32; 2]]) -> Option<Polygon> {
    if landmarks.len() < 468 {
        return None;
    }
    
    let vertices: Vec<[f32; 2]> = CHEEK_RIGHT_LANDMARKS
        .iter()
        .map(|&idx| landmarks[idx])
        .collect();
    
    Some(Polygon::new(vertices))
}

/// Compute mean RGB within a polygon region
///
/// Uses bounding box optimization: only check pixels within bbox,
/// then apply point-in-polygon test.
///
/// # Arguments
/// * `frame` - Raw frame bytes (RGB888)
/// * `width` - Frame width
/// * `height` - Frame height
/// * `polygon` - ROI polygon
pub fn compute_polygon_mean_rgb(
    frame: &[u8],
    width: u32,
    height: u32,
    polygon: &Polygon,
) -> [f32; 3] {
    let [x_min, y_min, x_max, y_max] = polygon.bounding_box();
    
    let x_start = (x_min as u32).min(width);
    let y_start = (y_min as u32).min(height);
    let x_end = (x_max as u32).min(width);
    let y_end = (y_max as u32).min(height);
    
    let mut sum = [0.0f32; 3];
    let mut count = 0u32;
    
    let stride = (width * 3) as usize;
    
    for y in y_start..y_end {
        for x in x_start..x_end {
            if polygon.contains(x as f32, y as f32) {
                let idx = (y as usize * stride) + (x as usize * 3);
                if idx + 2 < frame.len() {
                    sum[0] += frame[idx] as f32;
                    sum[1] += frame[idx + 1] as f32;
                    sum[2] += frame[idx + 2] as f32;
                    count += 1;
                }
            }
        }
    }
    
    if count > 0 {
        let inv = 1.0 / count as f32;
        [sum[0] * inv, sum[1] * inv, sum[2] * inv]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_bounding_box() {
        let poly = Polygon::new(vec![
            [10.0, 20.0],
            [30.0, 20.0],
            [30.0, 40.0],
            [10.0, 40.0],
        ]);
        
        let bbox = poly.bounding_box();
        assert_eq!(bbox, [10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_polygon_contains() {
        let poly = Polygon::new(vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ]);
        
        assert!(poly.contains(5.0, 5.0));
        assert!(!poly.contains(15.0, 5.0));
        assert!(!poly.contains(-1.0, 5.0));
    }

    #[test]
    fn test_forehead_roi_requires_landmarks() {
        let too_few: Vec<[f32; 2]> = vec![[0.0, 0.0]; 100];
        assert!(forehead_roi(&too_few).is_none());
        
        let enough: Vec<[f32; 2]> = vec![[0.0, 0.0]; 468];
        assert!(forehead_roi(&enough).is_some());
    }
}
