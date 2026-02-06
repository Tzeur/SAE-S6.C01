$root = "c:\Users\anas\Desktop\Cours\SAE"
cd $root

# Create directories
New-Item -ItemType Directory -Force -Path "data"
New-Item -ItemType Directory -Force -Path "notebooks"
New-Item -ItemType Directory -Force -Path "src"
New-Item -ItemType Directory -Force -Path "references"

# Move Data files (flatten structure)
$dataFiles = @("yelp_academic_dataset_business.json", "yelp_academic_dataset_user4students.jsonl", "yelp_academic_reviews4students.jsonl")
foreach ($name in $dataFiles) {
    if (Test-Path "Data\$name\$name") {
        Move-Item "Data\$name\$name" "data\$name" -Force
    }
}

# Move Resources (Notebooks)
Get-ChildItem "Ressources\*.ipynb" | Move-Item -Destination "notebooks" -Force

# Move Subject and Evaluation
if (Test-Path "Sujet") { Move-Item "Sujet\*" "references" -Force }
if (Test-Path "evaluation") { Move-Item "evaluation\*" "references" -Force }

# Move scripts
Move-Item "test_trello.js" "src\test_trello.js" -Force
Move-Item "setup_trello.js" "src\setup_trello.js" -Force
Move-Item "extract_pdf.py" "src\extract_pdf.py" -Force
Move-Item "pdf_content.txt" "references\pdf_content.txt" -Force

# Cleanup empty old folders
Remove-Item "Data" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "Ressources" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "Sujet" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "evaluation" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Project structure reorganized successfully."
