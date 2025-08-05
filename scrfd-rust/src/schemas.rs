#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
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
    pub score: f32,
    pub keypoints: FaceKeypoints,
    pub bbox: Bbox,
}

#[derive(Debug, Clone, Copy)]
pub struct Threshold {
    pub score: f32,
    pub iou: f32,
}
