#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub upper_left: Point,
    pub lower_right: Point,
}

impl Bbox {
    pub fn width(self) -> f32 {
        self.lower_right.x - self.upper_left.x
    }
    pub fn height(self) -> f32 {
        self.lower_right.y - self.upper_left.y
    }
    pub(crate) fn from_xywh(x: f32, y: f32, w: f32, h: f32) -> Self {
        let upper_left = Point { x, y };
        let lower_right = Point { x: x + w, y: y + h };
        Self {
            upper_left,
            lower_right,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FaceKeypoints {
    pub left_eye: Point,
    pub right_eye: Point,
    pub nose: Point,
    pub left_mouth: Point,
    pub right_mouth: Point,
}

#[derive(Debug, Clone, Copy)]
pub struct Face {
    pub probability: f32,
    pub keypoints: FaceKeypoints,
    pub bbox: Bbox,
}

#[derive(Debug, Clone, Copy)]
pub struct Threshold {
    pub score: f32,
    pub iou: f32,
}
