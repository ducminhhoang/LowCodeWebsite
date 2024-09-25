const { execSync } = require("child_process");
const fs = require("fs");

const changelog = execSync("npx conventional-changelog -p angular -i CHANGELOG.md -s").toString();
fs.writeFileSync("CHANGELOG.md", changelog);
console.log("Changelog updated!");