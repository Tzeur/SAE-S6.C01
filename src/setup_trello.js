const https = require('https');

const key = process.env.TRELLO_API_KEY || "";
const token = process.env.TRELLO_TOKEN || "";

const boardName = "S6.C01 - Analyse Yelp";
const lists = [
    "Done",
    "In Progress",
    "Phase C: GenAI & Agents",
    "Phase B: Prediction Models",
    "Phase A: Data Analysis",
    "Backlog"
];

const cards = {
    "Phase A: Data Analysis": [
        "Merge and clean dataset (review, user, business)",
        "Analyze distribution by business category",
        "Analyze correlation: Review Count vs Average Rating",
        "Analyze 'Big Reviewers' severity",
        "Analyze Review Length vs Sentiment & User Experience",
        "Vocabulary Analysis (TF-IDF Top 10 words)",
        "Analyze impact of photos on ratings"
    ],
    "Phase B: Prediction Models": [
        "Data Prep: Bag-of-Words, TF-IDF, Embeddings (BERT/LLM)",
        "Task 1: Polarity Classification (>3 Pos, <3 Neg, =3 Neu)",
        "Task 2: Rating Prediction (1-5)",
        "Implement Classic ML (LogReg, SVM)",
        "Implement Deep Learning (MLP, CNN)",
        "Implement Transformers (Fine-tuned or Pre-trained)",
        "Compare results"
    ],
    "Phase C: GenAI & Agents": [
        "Zero-shot/Few-shot classification with LLM",
        "Aspect-Based Sentiment Analysis (ABSA) with LangChain/LlamaIndex",
        "Build structured output pipeline (Aspect -> Sentiment)"
    ]
};

function trelloRequest(method, path, body = null) {
    return new Promise((resolve, reject) => {
        const options = {
            hostname: 'api.trello.com',
            path: `/1${path}${path.includes('?') ? '&' : '?'}key=${key}&token=${token}`,
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const req = https.request(options, res => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    resolve(JSON.parse(data));
                } else {
                    reject(new Error(`Request failed (${res.statusCode}): ${data}`));
                }
            });
        });

        req.on('error', reject);
        if (body) req.write(JSON.stringify(body));
        req.end();
    });
}

async function setup() {
    try {
        console.log(`Creating board "${boardName}"...`);
        const board = await trelloRequest('POST', `/boards/?name=${encodeURIComponent(boardName)}&defaultLists=false`);
        console.log(`Board created: ${board.id}`);

        // Create lists (in reverse order so they appear correctly left-to-right)
        const listIds = {};
        for (const listName of lists) {
            console.log(`Creating list "${listName}"...`);
            const list = await trelloRequest('POST', `/lists?name=${encodeURIComponent(listName)}&idBoard=${board.id}`);
            listIds[listName] = list.id;
        }

        // Create cards
        for (const [listName, cardTitles] of Object.entries(cards)) {
            const listId = listIds[listName];
            if (!listId) continue;
            for (const title of cardTitles) {
                console.log(`Creating card "${title}" in "${listName}"...`);
                await trelloRequest('POST', `/cards?name=${encodeURIComponent(title)}&idList=${listId}`);
            }
        }

        console.log("Trello setup complete!");
    } catch (error) {
        console.error("Error setting up Trello:", error);
    }
}

setup();
