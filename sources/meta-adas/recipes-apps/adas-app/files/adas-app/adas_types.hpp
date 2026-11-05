#pragma once

struct BoundingBox {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float conf = 0;
    int   cls = 0;
};

struct VehicleDetection {
    BoundingBox box;
    bool far = false;
};
