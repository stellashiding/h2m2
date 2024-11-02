// frontend/assets/sentimentSketch.js

let sentimentSketchInstance;

/**
 * Initializes the Sentiment Analysis Pie Chart.
 * @param {string} containerId - The CSS selector for the container div.
 * @param {Object} data - The sentiment data containing positive, neutral, and negative scores.
 */
function initSentimentSketch(containerId, data) {
    // Remove existing sketch instance if it exists
    if (sentimentSketchInstance) {
        sentimentSketchInstance.remove();
    }

    // Remove existing tooltip if any
    let existingTooltip = document.querySelector('.tooltip');
    if (existingTooltip) {
        existingTooltip.remove();
    }

    // Create a new tooltip element
    const tooltip = document.createElement('div');
    tooltip.classList.add('tooltip');
    document.querySelector(containerId).appendChild(tooltip);

    // Initialize p5.js sketch
    sentimentSketchInstance = new p5((p) => {
        let sentimentData = { positive: 0, neutral: 0, negative: 0 };
        let total = 0;
        let angles = { positive: 0, neutral: 0, negative: 0 };
        let animationProgress = 0; // Tracks the animation progress
        let targetProgress = 100; // Final animation progress

        // Colors with gradient stops
        const colors = {
            positive: {
                start: p.color(76, 175, 80, 200), // Green
                end: p.color(102, 187, 106, 200)  // Light Green
            },
            neutral: {
                start: p.color(33, 150, 243, 200), // Blue
                end: p.color(65, 182, 230, 200)    // Light Blue
            },
            negative: {
                start: p.color(244, 67, 54, 200),  // Red
                end: p.color(255, 138, 128, 200)   // Light Red
            }
        };

        // Tooltip variables
        let hoveredSegment = null;

        p.setup = function() {
            let sentimentDiv = p.select(containerId);
            if (!sentimentDiv) {
                console.error(`Container ${containerId} not found`);
                return;
            }
            let canvas = p.createCanvas(300, 300);
            canvas.parent(containerId);
            p.angleMode(p.DEGREES);
            p.noStroke();
            p.textAlign(p.CENTER, p.CENTER);
            p.textSize(16);

            // Initialize sentiment data
            this.updateSentiment(data); // Correctly call the method using 'this'
        }

        p.draw = function() {
            p.background(255);
            p.translate(p.width / 2, p.height / 2); // Center the pie chart

            // Calculate total sentiment score
            total = sentimentData.positive + sentimentData.neutral + sentimentData.negative;

            if (total === 0) {
                p.fill(0);
                p.text('No Sentiment Data', 0, 0);
                return;
            }

            // Calculate angles for each sentiment
            angles.positive = p.map(sentimentData.positive, 0, total, 0, 360);
            angles.neutral = p.map(sentimentData.neutral, 0, total, 0, 360);
            angles.negative = p.map(sentimentData.negative, 0, total, 0, 360);

            // Animate drawing
            if (animationProgress < targetProgress) {
                animationProgress += 2; // Increment animation progress
                p.redraw(); // Trigger redraw for animation
            }

            // Draw positive sentiment segment
            drawSegment(0, angles.positive, colors.positive);

            // Draw neutral sentiment segment
            drawSegment(angles.positive, angles.neutral, colors.neutral);

            // Draw negative sentiment segment
            drawSegment(angles.positive + angles.neutral, angles.negative, colors.negative);

            // Detect hover over segments
            detectHover();

            // Draw legend
            drawLegend();
        }

        /**
         * Draws a pie segment with gradient fill and shadow.
         * @param {number} startAngle - Starting angle of the segment.
         * @param {number} arcAngle - Angle span of the segment.
         * @param {Object} colorObj - Object containing start and end colors for the gradient.
         */
        function drawSegment(startAngle, arcAngle, colorObj) {
            p.push();
            // Shadow effect
            p.fill(0, 0, 0, 20);
            p.arc(5, 5, 200, 200, startAngle, startAngle + arcAngle, p.PIE);
            p.pop();

            // Create gradient
            let gradientSteps = 20;
            for (let i = 0; i < gradientSteps; i++) {
                let inter = p.map(i, 0, gradientSteps, 0, 1);
                let currentColor = p.lerpColor(colorObj.start, colorObj.end, inter);
                p.fill(currentColor);
                p.arc(0, 0, 200, 200, startAngle + (arcAngle * (i / gradientSteps)), startAngle + (arcAngle * ((i + 1) / gradientSteps)), p.PIE);
            }

            // Animate the drawing based on animationProgress
            if (animationProgress < targetProgress) {
                let limitedAngle = arcAngle * (animationProgress / targetProgress);
                p.push();
                p.fill(255, 255, 255, 0); // Transparent fill to mask the remaining part
                p.arc(0, 0, 200, 200, startAngle, startAngle + limitedAngle, p.PIE);
                p.pop();
            }
        }

        /**
         * Updates the sentiment data and resets animation.
         * @param {Object} data - New sentiment data.
         */
        p.updateSentiment = function(data) {
            sentimentData = data;
            animationProgress = 0;
            p.redraw();
        }

        /**
         * Detects if the mouse is hovering over any pie segment.
         */
        function detectHover() {
            let mouseX = p.mouseX - p.width / 2;
            let mouseY = p.mouseY - p.height / 2;
            let distance = p.dist(mouseX, mouseY, 0, 0);

            if (distance <= 100) { // Within the pie chart
                let angle = p.degrees(p.atan2(mouseY, mouseX));
                angle = angle < 0 ? angle + 360 : angle;

                if (angle < angles.positive) {
                    hoveredSegment = 'Positive';
                } else if (angle < angles.positive + angles.neutral) {
                    hoveredSegment = 'Neutral';
                } else {
                    hoveredSegment = 'Negative';
                }

                // Set tooltip
                if (hoveredSegment) {
                    let percentage = ((sentimentData[hoveredSegment.toLowerCase()] / total) * 100).toFixed(1);
                    tooltip.innerHTML = `${hoveredSegment}: ${percentage}%`;
                    tooltip.classList.add('visible');
                }
            } else {
                hoveredSegment = null;
                tooltip.classList.remove('visible');
            }
        }

        /**
         * Draws the legend with improved layout and spacing.
         */
        function drawLegend() {
            p.push();
            p.resetMatrix();
            p.translate(-p.width / 2, -p.height / 2); // Position legend at top-left
            
            // Legend container
            p.fill(0);
            p.textSize(14);
            p.textAlign(p.LEFT, p.CENTER);

            let legendX = 20;
            let legendY = p.height - 100;
            let legendSpacing = 30;

            // Positive Legend
            p.fill(colors.positive.start);
            p.rect(legendX, legendY, 20, 20);
            p.fill(0);
            p.text('Positive', legendX + 30, legendY + 10);
            legendY += legendSpacing;

            // Neutral Legend
            p.fill(colors.neutral.start);
            p.rect(legendX, legendY, 20, 20);
            p.fill(0);
            p.text('Neutral', legendX + 30, legendY + 10);
            legendY += legendSpacing;

            // Negative Legend
            p.fill(colors.negative.start);
            p.rect(legendX, legendY, 20, 20);
            p.fill(0);
            p.text('Negative', legendX + 30, legendY + 10);
            legendY += legendSpacing;

            p.pop();
        }

        p.mouseMoved = function() {
            p.redraw();
        }

        p.mouseClicked = function() {
            if (hoveredSegment) {
                alert(`${tooltip.innerHTML}`);
            }
        }
    });

    // Initialize with data
    if (sentimentSketchInstance && typeof sentimentSketchInstance.updateSentiment === 'function') {
        sentimentSketchInstance.updateSentiment(data);
    }
}
