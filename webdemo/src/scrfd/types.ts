type Point = {
     x: number;
     y: number;
}

class Bbox {
     upperLeft: Point;
     lowerRight: Point;

     constructor(upperLeft: Point, lowerRight: Point) {
         this.upperLeft = upperLeft;
	 this.lowerRight = lowerRight;
     }
}

