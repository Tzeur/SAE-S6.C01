
const key = process.env.TRELLO_API_KEY || "";
const token = process.env.TRELLO_TOKEN || "";
const url = `https://api.trello.com/1/members/me/boards?key=${key}&token=${token}`;

console.log("Testing Trello connection...");
fetch(url)
    .then(async res => {
        console.log(`Status: ${res.status} ${res.statusText}`);
        const text = await res.text();
        console.log("Response:", text.substring(0, 200));
    })
    .catch(err => {
        console.error("Error:", err);
    });
