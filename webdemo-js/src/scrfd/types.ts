export type Point = {
  x: number;
  y: number;
}

export class Bbox {
  upperLeft: Point;
  lowerRight: Point;

  constructor(upperLeft: Point, lowerRight: Point) {
      this.upperLeft = upperLeft;
      this.lowerRight = lowerRight;
  }

  width(): number {
    return this.lowerRight.x - this.upperLeft.x;
  }

  height(): number {
    return this.lowerRight.y - this.upperLeft.y;
  }

  static fromXywh(x: number, y: number, w: number, h: number): Bbox {
    const upperLeft: Point = { x, y };
    const lowerRight: Point = { x: x + w, y: y + h };
    return new Bbox(upperLeft, lowerRight);
  }
}

export type FaceKeypoints = {
  leftEye: Point;
  rightEye: Point;
  nose: Point;
  leftMouth: Point;
  rightMouth: Point;
};

export type Face = {
  probability: number;
  keypoints: FaceKeypoints;
  bbox: Bbox;
};

export type Threshold = {
  score: number;
  iou: number;
};
