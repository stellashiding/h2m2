// frontend/assets/backgroundSketch.js

let particles = [];
let particleCount;

function setup() {
    let canvas = createCanvas(windowWidth, windowHeight);
    canvas.position(0, 0);
    canvas.style('z-index', '-2'); // Behind the container
    background(240, 242, 245);
    calculateParticleCount();
    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }
}

function draw() {
    background(240, 242, 245, 25); // Semi-transparent for trail effect
    particles.forEach(p => {
        p.update();
        p.show();
    });
}

function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
    calculateParticleCount();
    // Adjust particles based on new count
    while (particles.length < particleCount) {
        particles.push(new Particle());
    }
    while (particles.length > particleCount) {
        particles.pop();
    }
}

function calculateParticleCount() {
    particleCount = Math.floor((windowWidth * windowHeight) / 20000); // Adjust density as needed
    particleCount = constrain(particleCount, 50, 200); // Set min and max
}

class Particle {
    constructor() {
        this.x = random(width);
        this.y = random(height);
        this.size = random(3, 7);
        this.speedX = random(-0.5, 0.5);
        this.speedY = random(-0.5, 0.5);
        this.color = color(76, 175, 80, 150); // Green with transparency
    }

    update() {
        this.x += this.speedX;
        this.y += this.speedY;

        // Wrap around the edges
        if (this.x > width) this.x = 0;
        if (this.x < 0) this.x = width;
        if (this.y > height) this.y = 0;
        if (this.y < 0) this.y = height;
    }

    show() {
        noStroke();
        fill(this.color);
        ellipse(this.x, this.y, this.size);
    }
}
