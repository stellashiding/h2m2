// frontend/assets/loadingSketch.js

let angle = 0;

function setup() {
    let loadingDiv = select('#loading-animation');
    let canvas = createCanvas(80, 80);
    canvas.parent('loading-animation');
    angleMode(DEGREES);
    noStroke();
    fill(76, 175, 80); // Green
}

function draw() {
    background(255, 255, 255, 0); // Transparent background
    translate(width / 2, height / 2);
    rotate(angle);
    ellipse(0, -30, 20, 20);
    ellipse(0, 30, 20, 20);
    angle += 5;
}
