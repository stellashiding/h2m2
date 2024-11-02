// frontend/sketch.js

document.addEventListener('DOMContentLoaded', () => {
    const analyzeButton = document.getElementById('analyze-button');
    const textInput = document.getElementById('text-input');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');

    analyzeButton.addEventListener('click', () => {
        const selectedOptions = [];
        if (document.getElementById('summary').checked) selectedOptions.push('summary');
        if (document.getElementById('key_points').checked) selectedOptions.push('key_points');
        if (document.getElementById('sentiment_analysis').checked) selectedOptions.push('sentiment_analysis');
        if (document.getElementById('paraphrasing').checked) selectedOptions.push('paraphrasing');
        if (document.getElementById('question_extraction').checked) selectedOptions.push('question_extraction');

        const userText = textInput.value.trim();
        if (userText === '') {
            alert('Please enter some text to analyze.');
            return;
        }

        if (selectedOptions.length === 0) {
            alert('Please select at least one analysis option.');
            return;
        }

        // Show loading indicator
        loadingDiv.classList.remove('hidden');
        resultsDiv.innerHTML = '';

        // Prepare payload
        const payload = {
            text: userText,
            analyses: selectedOptions
        };

        // Send POST request to backend
        fetch('http://127.0.0.1:5000/analyze_text', { // Ensure backend is running on this URL
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            loadingDiv.classList.add('hidden');
            if (data.error) {
                alert(`Error: ${data.error}`);
                return;
            }
            displayResults(data);
        })
        .catch(error => {
            loadingDiv.classList.add('hidden');
            console.error('Error:', error);
            alert('An error occurred while processing your request.');
        });
    });

    function displayResults(data) {
        if (data.summary) {
            const summaryHeader = document.createElement('h2');
            summaryHeader.textContent = 'Summary:';
            resultsDiv.appendChild(summaryHeader);

            // Create a container for the summary
            const summaryContainer = document.createElement('div');
            summaryContainer.classList.add('result-card');
            summaryContainer.id = 'summary-container';

            const summaryPara = document.createElement('p');
            summaryPara.textContent = data.summary;
            summaryContainer.appendChild(summaryPara);
            resultsDiv.appendChild(summaryContainer);

            // Animate the card
            setTimeout(() => {
                summaryContainer.classList.add('visible');
            }, 100);
        }

        if (data.key_points) {
            const keyPointsHeader = document.createElement('h2');
            keyPointsHeader.textContent = 'Key Points:';
            resultsDiv.appendChild(keyPointsHeader);

            const keyPointsCard = document.createElement('div');
            keyPointsCard.classList.add('result-card');
            const keyPointsList = document.createElement('ul');
            data.key_points.forEach(point => {
                const li = document.createElement('li');
                li.textContent = point;
                keyPointsList.appendChild(li);
            });
            keyPointsCard.appendChild(keyPointsList);
            resultsDiv.appendChild(keyPointsCard);

            // Animate the card
            setTimeout(() => {
                keyPointsCard.classList.add('visible');
            }, 200);
        }

        if (data.sentiment) {
            const sentimentHeader = document.createElement('h2');
            sentimentHeader.textContent = 'Sentiment Analysis:';
            resultsDiv.appendChild(sentimentHeader);

            // Create a container for the sentiment visualization
            const sentimentContainer = document.createElement('div');
            sentimentContainer.classList.add('result-card');
            sentimentContainer.id = 'sentiment-analysis-container';
            resultsDiv.appendChild(sentimentContainer);

            // Initialize P5.js sentiment sketch
            if (typeof initSentimentSketch === 'function') {
                initSentimentSketch('#sentiment-analysis-container', data.sentiment);
            }

            // Animate the sentiment container
            setTimeout(() => {
                sentimentContainer.classList.add('visible');
            }, 300);
        }

        if (data.paraphrased_text) {
            const paraphraseHeader = document.createElement('h2');
            paraphraseHeader.textContent = 'Paraphrased Text:';
            resultsDiv.appendChild(paraphraseHeader);

            const paraphraseCard = document.createElement('div');
            paraphraseCard.classList.add('result-card');
            const paraphrasePara = document.createElement('p');
            paraphrasePara.textContent = data.paraphrased_text;
            paraphraseCard.appendChild(paraphrasePara);
            resultsDiv.appendChild(paraphraseCard);

            // Animate the card
            setTimeout(() => {
                paraphraseCard.classList.add('visible');
            }, 400);
        }

        if (data.generated_questions) {
            const questionsHeader = document.createElement('h2');
            questionsHeader.textContent = 'Generated Questions:';
            resultsDiv.appendChild(questionsHeader);

            const questionsCard = document.createElement('div');
            questionsCard.classList.add('result-card');
            const questionsList = document.createElement('ul');
            data.generated_questions.forEach(question => {
                const li = document.createElement('li');
                li.textContent = question;
                li.classList.add('clickable-question');
                li.addEventListener('click', () => {
                    getAnswerForQuestion(question);
                });
                questionsList.appendChild(li);
            });
            questionsCard.appendChild(questionsList);
            resultsDiv.appendChild(questionsCard);

            // Animate the card
            setTimeout(() => {
                questionsCard.classList.add('visible');
            }, 500);
        }
    }

    function getAnswerForQuestion(question) {
        const userText = textInput.value.trim();
        if (!userText) {
            alert('Please enter some text.');
            return;
        }

        // Prepare payload
        const payload = {
            text: userText,
            question: question
        };

        // Send POST request to backend
        fetch('http://127.0.0.1:5000/answer_question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            displayAnswer(question, data.answer);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while fetching the answer.');
        });
    }

    function displayAnswer(question, answer) {
        const questionItems = document.querySelectorAll('.clickable-question');
        questionItems.forEach(item => {
            if (item.textContent === question) {
                // Check if the answer is already displayed
                if (item.nextElementSibling && item.nextElementSibling.classList.contains('answer')) {
                    // Toggle visibility
                    item.nextElementSibling.classList.toggle('hidden');
                } else {
                    // Create an answer element
                    const answerElement = document.createElement('div');
                    answerElement.classList.add('answer');
                    answerElement.textContent = answer;
                    // Insert after the question item
                    item.parentNode.insertBefore(answerElement, item.nextSibling);
                }
            }
        });
    }
});
