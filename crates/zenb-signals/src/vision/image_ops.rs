//! Image Processing Operations
//!
//! Provides basic image operations when the `image-processing` feature is enabled:
//! - Load images from file or bytes
//! - Resize with interpolation
//! - Color conversion (RGB, Grayscale)
//! - Crop/ROI extraction
//! - Flip operations for front camera

#[cfg(feature = "image-processing")]
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba, RgbImage};

/// Image wrapper that provides unified access to pixel data
#[derive(Debug, Clone)]
pub struct Frame {
    /// Raw RGB8 pixel data (row-major)
    pub data: Vec<u8>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Timestamp in microseconds
    pub timestamp_us: i64,
}

impl Frame {
    /// Create a new frame from raw RGB8 data
    pub fn new(data: Vec<u8>, width: u32, height: u32, timestamp_us: i64) -> Self {
        assert_eq!(data.len(), (width * height * 3) as usize, "Data size mismatch");
        Self { data, width, height, timestamp_us }
    }

    /// Create an empty frame
    pub fn empty(width: u32, height: u32) -> Self {
        Self {
            data: vec![0u8; (width * height * 3) as usize],
            width,
            height,
            timestamp_us: 0,
        }
    }

    /// Get pixel at (x, y) as [R, G, B]
    #[inline]
    pub fn get_pixel(&self, x: u32, y: u32) -> [u8; 3] {
        if x >= self.width || y >= self.height {
            return [0, 0, 0];
        }
        let idx = ((y * self.width + x) * 3) as usize;
        [self.data[idx], self.data[idx + 1], self.data[idx + 2]]
    }

    /// Set pixel at (x, y)
    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, rgb: [u8; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = ((y * self.width + x) * 3) as usize;
        self.data[idx] = rgb[0];
        self.data[idx + 1] = rgb[1];
        self.data[idx + 2] = rgb[2];
    }

    /// Crop a rectangular region
    pub fn crop(&self, x: u32, y: u32, crop_w: u32, crop_h: u32) -> Frame {
        let mut cropped = Frame::empty(crop_w, crop_h);
        cropped.timestamp_us = self.timestamp_us;

        for dy in 0..crop_h {
            for dx in 0..crop_w {
                let src_x = x + dx;
                let src_y = y + dy;
                if src_x < self.width && src_y < self.height {
                    cropped.set_pixel(dx, dy, self.get_pixel(src_x, src_y));
                }
            }
        }
        cropped
    }

    /// Flip horizontally (mirror for front camera)
    pub fn flip_horizontal(&self) -> Frame {
        let mut flipped = Frame::empty(self.width, self.height);
        flipped.timestamp_us = self.timestamp_us;

        for y in 0..self.height {
            for x in 0..self.width {
                let src_x = self.width - 1 - x;
                flipped.set_pixel(x, y, self.get_pixel(src_x, y));
            }
        }
        flipped
    }

    /// Compute mean RGB values for the entire frame
    pub fn mean_rgb(&self) -> [f32; 3] {
        let mut sum = [0.0f64; 3];
        let count = (self.width * self.height) as f64;

        for y in 0..self.height {
            for x in 0..self.width {
                let px = self.get_pixel(x, y);
                sum[0] += px[0] as f64;
                sum[1] += px[1] as f64;
                sum[2] += px[2] as f64;
            }
        }

        [(sum[0] / count) as f32, (sum[1] / count) as f32, (sum[2] / count) as f32]
    }

    /// Compute mean RGB for a rectangular ROI
    pub fn roi_mean_rgb(&self, x: u32, y: u32, w: u32, h: u32) -> [f32; 3] {
        let mut sum = [0.0f64; 3];
        let mut count = 0u32;

        for dy in 0..h {
            for dx in 0..w {
                let px_x = x + dx;
                let px_y = y + dy;
                if px_x < self.width && px_y < self.height {
                    let px = self.get_pixel(px_x, px_y);
                    sum[0] += px[0] as f64;
                    sum[1] += px[1] as f64;
                    sum[2] += px[2] as f64;
                    count += 1;
                }
            }
        }

        if count > 0 {
            let inv = 1.0 / count as f64;
            [(sum[0] * inv) as f32, (sum[1] * inv) as f32, (sum[2] * inv) as f32]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Resize frame using nearest-neighbor interpolation
    pub fn resize(&self, new_width: u32, new_height: u32) -> Frame {
        let mut resized = Frame::empty(new_width, new_height);
        resized.timestamp_us = self.timestamp_us;

        let x_ratio = self.width as f32 / new_width as f32;
        let y_ratio = self.height as f32 / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = ((x as f32 * x_ratio) as u32).min(self.width - 1);
                let src_y = ((y as f32 * y_ratio) as u32).min(self.height - 1);
                resized.set_pixel(x, y, self.get_pixel(src_x, src_y));
            }
        }
        resized
    }

    /// Resize with bilinear interpolation (higher quality)
    pub fn resize_bilinear(&self, new_width: u32, new_height: u32) -> Frame {
        let mut resized = Frame::empty(new_width, new_height);
        resized.timestamp_us = self.timestamp_us;

        let x_ratio = (self.width as f32 - 1.0) / new_width as f32;
        let y_ratio = (self.height as f32 - 1.0) / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = x as f32 * x_ratio;
                let src_y = y as f32 * y_ratio;

                let x0 = src_x as u32;
                let y0 = src_y as u32;
                let x1 = (x0 + 1).min(self.width - 1);
                let y1 = (y0 + 1).min(self.height - 1);

                let x_frac = src_x - x0 as f32;
                let y_frac = src_y - y0 as f32;

                let p00 = self.get_pixel(x0, y0);
                let p10 = self.get_pixel(x1, y0);
                let p01 = self.get_pixel(x0, y1);
                let p11 = self.get_pixel(x1, y1);

                let mut result = [0u8; 3];
                for c in 0..3 {
                    let top = p00[c] as f32 * (1.0 - x_frac) + p10[c] as f32 * x_frac;
                    let bottom = p01[c] as f32 * (1.0 - x_frac) + p11[c] as f32 * x_frac;
                    result[c] = (top * (1.0 - y_frac) + bottom * y_frac) as u8;
                }
                resized.set_pixel(x, y, result);
            }
        }
        resized
    }

    /// Convert to grayscale
    pub fn to_grayscale(&self) -> Vec<u8> {
        let mut gray = Vec::with_capacity((self.width * self.height) as usize);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let px = self.get_pixel(x, y);
                // ITU-R BT.601 luma coefficients
                let luma = (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) as u8;
                gray.push(luma);
            }
        }
        gray
    }
}

// === Image crate integration (optional) ===

#[cfg(feature = "image-processing")]
impl Frame {
    /// Load frame from file path
    pub fn from_file(path: &str) -> Result<Self, String> {
        let img = image::open(path).map_err(|e| e.to_string())?;
        Ok(Self::from_dynamic_image(&img, 0))
    }

    /// Load frame from bytes (JPEG/PNG)
    pub fn from_bytes(bytes: &[u8], timestamp_us: i64) -> Result<Self, String> {
        let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
        Ok(Self::from_dynamic_image(&img, timestamp_us))
    }

    /// Convert from image crate DynamicImage
    pub fn from_dynamic_image(img: &DynamicImage, timestamp_us: i64) -> Self {
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        Self {
            data: rgb.into_raw(),
            width,
            height,
            timestamp_us,
        }
    }

    /// Convert to image crate RgbImage
    pub fn to_rgb_image(&self) -> RgbImage {
        ImageBuffer::from_raw(self.width, self.height, self.data.clone())
            .expect("Buffer size mismatch")
    }

    /// Save to file (JPEG/PNG based on extension)
    pub fn save(&self, path: &str) -> Result<(), String> {
        let img = self.to_rgb_image();
        img.save(path).map_err(|e| e.to_string())
    }
}

/// Convert RGBA to RGB (drop alpha channel)
pub fn rgba_to_rgb(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    
    for i in 0..pixel_count {
        let idx = i * 4;
        if idx + 2 < rgba.len() {
            rgb.push(rgba[idx]);
            rgb.push(rgba[idx + 1]);
            rgb.push(rgba[idx + 2]);
        }
    }
    rgb
}

/// Convert YUV (NV21) to RGB - common Android camera format
pub fn nv21_to_rgb(yuv: &[u8], width: u32, height: u32) -> Vec<u8> {
    let frame_size = (width * height) as usize;
    let mut rgb = vec![0u8; frame_size * 3];

    for y in 0..height {
        for x in 0..width {
            let y_idx = (y * width + x) as usize;
            let uv_idx = frame_size + (y / 2) as usize * width as usize + (x & !1) as usize;

            let y_val = yuv.get(y_idx).copied().unwrap_or(0) as i32;
            let v = yuv.get(uv_idx).copied().unwrap_or(128) as i32 - 128;
            let u = yuv.get(uv_idx + 1).copied().unwrap_or(128) as i32 - 128;

            let r = (y_val + (1.370705 * v as f32) as i32).clamp(0, 255) as u8;
            let g = (y_val - (0.337633 * u as f32) as i32 - (0.698001 * v as f32) as i32).clamp(0, 255) as u8;
            let b = (y_val + (1.732446 * u as f32) as i32).clamp(0, 255) as u8;

            let rgb_idx = y_idx * 3;
            rgb[rgb_idx] = r;
            rgb[rgb_idx + 1] = g;
            rgb[rgb_idx + 2] = b;
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_creation() {
        let frame = Frame::empty(100, 100);
        assert_eq!(frame.width, 100);
        assert_eq!(frame.height, 100);
        assert_eq!(frame.data.len(), 100 * 100 * 3);
    }

    #[test]
    fn test_pixel_operations() {
        let mut frame = Frame::empty(10, 10);
        frame.set_pixel(5, 5, [255, 128, 64]);
        let px = frame.get_pixel(5, 5);
        assert_eq!(px, [255, 128, 64]);
    }

    #[test]
    fn test_crop() {
        let mut frame = Frame::empty(100, 100);
        frame.set_pixel(50, 50, [255, 0, 0]);
        
        let cropped = frame.crop(40, 40, 20, 20);
        assert_eq!(cropped.width, 20);
        assert_eq!(cropped.height, 20);
        assert_eq!(cropped.get_pixel(10, 10), [255, 0, 0]);
    }

    #[test]
    fn test_flip_horizontal() {
        let mut frame = Frame::empty(100, 100);
        frame.set_pixel(0, 50, [255, 0, 0]);
        
        let flipped = frame.flip_horizontal();
        assert_eq!(flipped.get_pixel(99, 50), [255, 0, 0]);
    }

    #[test]
    fn test_resize() {
        let frame = Frame::empty(100, 100);
        let resized = frame.resize(50, 50);
        assert_eq!(resized.width, 50);
        assert_eq!(resized.height, 50);
    }

    #[test]
    fn test_mean_rgb() {
        let data = vec![128u8; 30 * 30 * 3];
        let frame = Frame::new(data, 30, 30, 0);
        let mean = frame.mean_rgb();
        assert!((mean[0] - 128.0).abs() < 0.01);
    }

    #[test]
    fn test_rgba_to_rgb() {
        let rgba = vec![255, 128, 64, 255, 100, 50, 25, 128];
        let rgb = rgba_to_rgb(&rgba, 2, 1);
        assert_eq!(rgb, vec![255, 128, 64, 100, 50, 25]);
    }

    #[test]
    fn test_grayscale() {
        let data = vec![255, 255, 255, 0, 0, 0]; // White + Black pixels
        let frame = Frame::new(data, 2, 1, 0);
        let gray = frame.to_grayscale();
        assert_eq!(gray[0], 255); // White
        assert_eq!(gray[1], 0);   // Black
    }
}
